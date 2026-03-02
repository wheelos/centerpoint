# CenterPointTRT (Apollo) — Training/Export Handoff Notes

This document captures the current state of the `center_point_trt_mmdet3d` toolchain, and what to watch out for when continuing with **MMDetection3D training code**.

It is written for future new teammates to pick up quickly.

## Goal & Current Status

- Goal: reproduce Apollo’s **two-stage ONNX** contract for CenterPointTRT:
  - `cpdet_pfe.onnx`: `voxels -> pillar_feature`
  - `cpdet_backbone.onnx`: `canvas_feature -> (scores, bbox_preds, dir_scores)`
- Status: the exported empty ONNX (random weights) is aligned in **graph structure** with Apollo’s shipped ONNX, including:
  - first `1x1 Conv` stem in backbone
  - Apollo-style neck (downsample + lateral + upsample+BN) + concat
  - multi-task head fan-out and concat packing
  - output order `scores, bbox_preds, dir_scores`

Reference “source of truth” ONNX:
- `model_reference/cpdet_pfe.onnx`
- `model_reference/cpdet_backbone.onnx`

## Apollo Runtime Contract (Shapes / Names / Packing)

From Apollo centerpoint model parameters:

- **PFE**
  - Input: `voxels` shape `[N, 9]` (export uses `[N, 1, 9, 1]` for compatibility)
  - Output: `pillar_feature` shape `[N, 48]`
- **Backbone**
  - Input: `canvas_feature` shape `[1, 64, 512, 512]` (48 + 16 channels)
  - Output (order matters for Netron/readability; runtime should bind by name):
    - `scores` shape `[1, 4, 128, 128]`
    - `bbox_preds` shape `[1, 24, 128, 128]`
    - `dir_scores` shape `[1, 8, 128, 128]`

Channel packing expectations:
- 4 tasks × 1 class each: `Car/Pedestrian/Bicycle/TrafficCone`
- `dir_scores`: `rot(2)` per task => `2 * 4 = 8`
- `bbox_preds`: per task `[reg(2), hei(1), dim(3)]` => `6 * 4 = 24`
- `bbox_preds` is packed **per task**:
  - `[t0.reg, t0.hei, t0.dim, t1.reg, t1.hei, t1.dim, t2..., t3...]`

## “Model” vs “Pipeline” (What’s inside ONNX vs outside)

Important: Apollo’s implementation keeps **voxelization / scatter / BEV extra features** out of ONNX.

- Outside ONNX (C++/CUDA in car-side inference; PyTorch for training-side):
  - point filtering to grid
  - per-point voxel feature build (9D)
  - scatter-max to BEV (48 channels)
  - optional extra BEV features (“cnnseg features”, 16 channels)
- Inside ONNX:
  - PFE: Linear/BN/ReLU
  - Backbone/neck/head: conv stack + CenterHead-style heads

## PointCloud -> 9D Voxel Feature (Apollo-aligned)

This is the most important “silent mismatch” risk when writing training code: if your 9D feature math differs slightly from Apollo runtime, the model can train and converge but won’t match car-side behavior.

Source of truth in this repo (of course these code can not be compiled, only for reference):
- `modules/perception/lidar/lib/detector/center_point_trt/center_point_trt.cu`
  - `Point2GridKernel`, `PointCloudSumKernel`, `VoxelFeatureKernel`
- `modules/perception/lidar/lib/detector/center_point_trt/center_point_trt.cc`
  - `x_offset_ / y_offset_` definition in `CenterPointTRT::LoadParams()`

### 0) Input fields

Each point provides at least:
- `x, y, z`
- `intensity` (typically 0~255 in Apollo pipelines)

### 1) Optional 45° rotation (before grid + feature)

If `enable_rotate_45degree=true`, Apollo uses rotated `(px, py)` everywhere in voxelization:

```
px = 0.707107 * x - 0.707107 * y
py = 0.707107 * x + 0.707107 * y
```

`z` and `intensity` are not rotated.

### 2) Point -> pillar grid index (2D pillar grid)

Apollo uses a 2D pillar grid (no z-indexing for voxel id). A point is discarded if:
- `z < z_min_range` or `z > z_max_range`
- rotated/unrotated `(px,py)` falls outside `[x_min_range,x_max_range] × [y_min_range,y_max_range]`

Grid coordinates:

```
coord_x = int((px - x_min_range) / voxel_x_size)
coord_y = int((py - y_min_range) / voxel_y_size)
grid_idx = coord_y * grid_x_size + coord_x
```

### 3) Per-pillar mean (cluster mean)

For each `grid_idx`, Apollo accumulates (using the same `px/py` from above):
- `sum_x += px`, `sum_y += py`, `sum_z += z`, `count += 1`

Then:

```
x_mean = sum_x / max(count, 1)
y_mean = sum_y / max(count, 1)
z_mean = sum_z / max(count, 1)
```

### 4) 9D feature definition (per point)

Apollo writes features in this exact order (`voxel_feature_dim` is typically 9):

**Dims 0~3: base coords + intensity**
- if `use_input_norm=true`:
  - `f0 = px / x_max_range`
  - `f1 = py / y_max_range`
  - `f2 = z  / z_max_range`
  - `f3 = intensity / 255.0`
- else:
  - `f0 = px`
  - `f1 = py`
  - `f2 = z`
  - `f3 = intensity`

**Dims 4~6: cluster offset**
```
f4 = px - x_mean
f5 = py - y_mean
f6 = z  - z_mean
```

**Dims 7~8: pillar-center offset**

Apollo defines:
```
x_offset = voxel_x_size / 2 + x_min_range
y_offset = voxel_y_size / 2 + y_min_range
```

So each pillar’s geometric center is:
```
pillar_center_x = coord_x * voxel_x_size + x_offset
pillar_center_y = coord_y * voxel_y_size + y_offset
```

And:
```
f7 = px - pillar_center_x
f8 = py - pillar_center_y
```

### 5) Tensor shape fed into PFE ONNX

Apollo runtime forms `voxels` as a per-point feature table (not “T points per voxel”):
- logical: `[N, 9]`
- export tooling uses: `[N, 1, 9, 1]` (equivalent, easier to match Apollo ONNX ops)

## Code Layout

Core modules (PyTorch side):
- `apollo_centerpoint_trt/pfe.py`:
  - `ApolloPFE`: `Linear(bias=False) + BN1d + ReLU`
  - accepts `[N,1,9,1]` or `[N,9]`
- `apollo_centerpoint_trt/bev_feature.py`:
  - `ApolloBevFeatureGenerator`: feature math aligned to Apollo C++
  - uses scatter-max; “cnnseg features” are `detach()`’d (treated as fixed features)
- `apollo_centerpoint_trt/backbone.py`:
  - `ApolloSecondBackboneLite`: custom Conv+ReLU backbone (no BN) matching shipped ONNX
- `apollo_centerpoint_trt/neck.py`:
  - `ApolloNeckLite`: custom 3-branch neck matching shipped ONNX
  - ConvTranspose bias is **disabled** to match Apollo’s `ConvTranspose_38` (W only)
- `apollo_centerpoint_trt/export_wrappers.py`:
  - `BackboneHeadExportWrapper`: normalizes CenterHead outputs across MMDet3D versions and packs outputs into Apollo contract
  - returns `(scores, bbox_preds, dir_scores)` in this order
- `apollo_centerpoint_trt/mmdet3d_centerpoint_trt.py`:
  - `CenterPointTRTDetector`: MMEngine `BaseModel`-compatible wrapper intended for training
  - rotates GT boxes when `enable_rotate_45degree` is enabled
- `apollo_centerpoint_trt/export_model.py`:
  - `CenterPointTRTExportModel`: minimal container for ONNX export (backbone+neck+head+pfe)

Export / debug scripts:
- `tools/export_onnx.py`: exports two ONNX files; handles CPU-only torch environments; optional Identity stripping; optional output reordering
- `tools/diff_onnx.py`: prints structural diffs (Conv prefix, task_heads count, output shapes, op stats)

## Critical Detail: `enable_rotate_45degree`

Apollo’s preprocessing can rotate xy by +45° before voxelization/scatter:

- points rotation used in `ApolloBevFeatureGenerator`:
  - `px = (x - y) / sqrt(2)`
  - `py = (x + y) / sqrt(2)`
- implication for training:
  - if preprocessing rotates points, **GT boxes must be rotated by +45°** in the same frame before computing targets/loss
  - predictions/decoded boxes should be rotated back by **-45°** for evaluation/visualization in original lidar frame

`CenterPointTRTDetector` currently performs the GT rotation around z when enabled.

## MMDetection3D Training: Recommended Approach

Do NOT attempt to train from the exported ONNX directly. Use PyTorch modules and export ONNX only after training.

Recommended training path:

1) Use MMDet3D `tools/train.py` + MMEngine runner
2) Register this project as a plugin via `custom_imports`
3) Set `model.type = "CenterPointTRTDetector"` (not `CenterPointTRTExportModel`)
4) Compose:
   - Apollo preprocessing (`ApolloBevFeatureGenerator`)
   - PFE (`ApolloPFE`)
   - backbone/neck/head (Apollo-lite or MMDet3D modules if compatible)

### What data the model expects

`CenterPointTRTDetector` is designed to accept:

- `inputs`: a dict containing `points`, where `points` is `List[Tensor]` (one tensor per sample)
  - each points tensor: `[..., >=4]` columns, using `x,y,z,intensity` from the first 4 dims
- `data_samples`: list of `Det3DDataSample`-like objects containing:
  - `gt_instances_3d.bboxes_3d`
  - `gt_instances_3d.labels_3d`
  - metainfo (if your head’s coder/predict needs it)

You will need to implement a dataset + pipeline that yields the above.

### Registry / scope gotchas (MMDet3D 1.4.0)

- Some losses are under MMDet scope; configs should set:
  - `loss_cls=dict(_scope_='mmdet', type='GaussianFocalLoss', ...)`
  - `loss_bbox=dict(_scope_='mmdet', type='L1Loss', ...)`
- When building from plain python configs (empty export path), we call `mmdet3d.utils.register_all_modules()` best-effort so registries contain MMDet components.

## Coordinate Frames & Conventions

This toolchain is **sensor-frame first**: CenterPointTRT voxelization and inference use `LidarFrame::cloud` as-is (no IMU/novatel transform applied before voxelization).

### Frames you will see in Apollo lidar perception

- **Lidar / Sensor frame** (a.k.a. “local frame”)
  - Source: `drivers::PointCloud` fields `x/y/z` copied into `LidarFrame::cloud`
  - Used by: HDMap ROI filtering (after polygons are transformed), voxel feature build (9D), PFE/backbone inference, and raw detection boxes (`cluster.x/y/z/yaw`).
  - Common convention in Apollo lidar point clouds is right-handed **x forward, y left, z up**.
  - If your platform uses a different convention, you must keep it consistent end-to-end (or explicitly convert both points and GT boxes).

- **World / Map frame**
  - Source: `LidarFrame::lidar2world_pose`
  - Used by: HDMap polygons (originally in world), `LidarFrame::world_cloud`, and tracking output alignment.

- **IMU / Novatel frame**
  - Source: `LidarFrame::lidar2novatel_extrinsics` + `novatel2world_pose`
  - Not used by CenterPointTRT inference by default. Only apply this if you intentionally want to canonicalize points/boxes into novatel frame.

### MMDet3D compatibility note

Many MMDet3D datasets/tools assume a particular lidar axis convention. The safest approach for “match car-side inference” is:
- keep the **same axis convention as `LidarFrame::cloud`** in your dataset/pipeline, and
- avoid extra axis swaps unless you also update `ApolloBevFeatureGenerator` (and GT box transforms) accordingly.

## Export Workflow (after training)

Export ONNX (two files):

```bash
python3 modules/perception/lidar/tools/center_point_trt_mmdet3d/tools/export_onnx.py \
  --config /path/to/your_mmdet3d_cfg.py \
  --checkpoint /path/to/epoch_xx.pth \
  --out-dir /tmp/center_point_trt_onnx \
  --strip-identity
```

Then compare exported backbone with Apollo:

```bash
python3 tools/diff_onnx.py \
  --a model_reference/cpdet_backbone.onnx \
  --b /tmp/center_point_trt_onnx/cpdet_backbone.onnx
```

Expected summary for a fully aligned backbone:
- `task_heads: 4 ids=['0','1','2','3']`
- outputs `scores(1,4,128,128) bbox_preds(1,24,128,128) dir_scores(1,8,128,128)`
- first Conv `w=(64,64,1,1)`
s
## Known Pitfalls / Lessons Learned (from this integration)

- MMDet3D version drift:
  - `CenterPoint` detector init args vary a lot (`pts_voxel_layer` etc.). Avoid depending on the full detector wrapper for Apollo TRT.
- `CenterHead.forward()` return shape varies:
  - may be `list[dict]`, `(list[dict], aux...)`, or `tuple(list[dict], list[dict], ...)` split by `multi_apply`.
  - `export_wrappers.py` normalizes these forms.
- CPU-only torch in a “temporary env”:
  - `export_onnx.py` auto-falls back to CPU if CUDA isn’t available.
- Netron graph cosmetics:
  - exporter may insert `Identity` nodes; `--strip-identity` removes them post-export (requires `onnx` python package).
