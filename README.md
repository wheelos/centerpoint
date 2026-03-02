# CenterPointTRT — MMDetection3D Training + ONNX Export

This folder provides a **training + export scaffold** for the Wheel.OS bussiness version
`CenterPointTRT` inference contract used in this repo:

- `cpdet_pfe.onnx`: point-wise PFE, **`voxels -> pillar_feature`**
- `cpdet_backbone.onnx`: BEV backbone+head, **`canvas_feature -> (bbox_preds, scores, dir_scores)`**

Important: **voxelization / scatter are *not* inside ONNX** in Apollo's
implementation. They are done by C++/CUDA pre/post-processing.

## Contract (must match car-side inference)

From `modules/perception/lidar_cpdet_detection/data/cpdet_param.pb.txt` on Apollo bussiness version and the
current C++ inference:

- PFE input `voxels`: `[N, 9]` (or `[N, 1, 9, 1]`) where each row is a **point**
  feature:
  - `[x, y, z, i, dx_mean, dy_mean, dz_mean, dx_center, dy_center]`
  - `use_input_norm=true` means `x/y/z` are range-normalized and `i` is scaled
    by `1/255` (matching C++).
- PFE output `pillar_feature`: `[N, 48]`
- Backbone input `canvas_feature`: `[1, 64, 512, 512]` (48 + 16 channels)
  - 48 from max-pool scatter of `pillar_feature`
  - 16 from "cnnseg" style extra BEV features (max/mean height, intensity,
    count/nonempty, height bins)
- Backbone outputs:
  - `bbox_preds`: `[1, 24, 128, 128]` (`(reg 2 + hei 1 + dim 3) * num_tasks(4)`)
  - `scores`: `[1, 4, 128, 128]`
  - `dir_scores`: `[1, 8, 128, 128]` (`rot 2 * num_tasks(4)`)

This matches the current pipeline config:
`modules/perception/pipeline/config/lidar_detection_pipeline_trt.pb.txt`.

## What this scaffold gives you

- `apollo_centerpoint_trt/pfe.py`: a small PFE module that matches the existing
  `cpdet_pfe.onnx` structure (Linear/MatMul + BatchNorm + ReLU).
- `apollo_centerpoint_trt/bev_feature.py`: PyTorch feature generation that
  matches the C++ preprocessing contract (voxel feature + scatter-max + cnnseg
  extra features).
- `tools/export_onnx.py`: export script that produces **two ONNX files** with
  the exact tensor names expected by Apollo C++.

## Dependencies (training-side)

You run this in your **training environment**, not inside Apollo.

- PyTorch >= 2.0 is recommended (for `scatter_reduce`).
  If you're on PyTorch 1.x, install `torch-scatter` and the code will use it.
- MMDetection3D installed (version depends on your stack). The export script
  expects a CenterPoint-like model with `pts_backbone/pts_neck/pts_bbox_head`.

## Usage

### 1) Train (MMDetection3D)

Integrate `apollo_centerpoint_trt` as a plugin in your MMDetection3D repo and
create a model that:

- generates `canvas_feature` using `ApolloBevFeatureGenerator`
- feeds it into a CenterPoint backbone+neck+head
- uses 4 tasks with 1 class each: `Car/Pedestrian/Bicycle/TrafficCone`

If you are using MMEngine "lazy config", **do not** import custom `torch.nn.Module`
classes directly in the config file. Use registry strings and `custom_imports`:

```python
custom_imports = dict(imports=["apollo_centerpoint_trt"], allow_failed_imports=False)
model = dict(type="CenterPointTRTDetector", ...)
```

This repo does **not** include a full runnable dataset config, since datasets
are project-specific. The model-side contract above is the key.

### 2) Export ONNX

After training, export with:

```bash
python3 modules/perception/lidar/tools/center_point_trt_mmdet3d/tools/export_onnx.py \
  --config /path/to/your_mmdet3d_cfg.py \
  --checkpoint /path/to/epoch_xx.pth \
  --strip-identity \
  --out-dir /tmp/center_point_trt_onnx
```

Copy the exported files into Apollo work_root, e.g.:

`data/perception/lidar/models/detection/center_point_trt/cpdet_pfe.onnx`
`data/perception/lidar/models/detection/center_point_trt/cpdet_backbone.onnx`

Then clear TensorRT engine cache if needed (or use distinct cache suffixes).

### Export "empty" (untrained) ONNX for Netron inspection

If you only want to inspect the graph structure in Netron, you can export
without a checkpoint (random weights):

```bash
python3 modules/perception/lidar/tools/center_point_trt_mmdet3d/tools/export_onnx.py \
  --config /path/to/your_mmdet3d_cfg.py \
  --strip-identity \
  --out-dir /tmp/center_point_trt_onnx
```

### Diff Apollo ONNX vs exported ONNX (structure check)

Install `onnx` in the same env, then run:

```bash
python3 modules/perception/lidar/tools/center_point_trt_mmdet3d/tools/diff_onnx.py \
  --a modules/perception/production/data/perception/lidar/models/detection/center_point_trt/cpdet_backbone.onnx \
  --b /tmp/center_point_trt_onnx/cpdet_backbone.onnx
```
