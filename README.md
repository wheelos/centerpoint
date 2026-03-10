# CenterPointTRT — MMDetection3D Training + ONNX Export

This folder provides a **training + export scaffold** for the Wheel.OS business version
`CenterPointTRT` inference contract used in this repo:

- `cpdet_pfe.onnx`: point-wise PFE, **`voxels -> pillar_feature`**
- `cpdet_backbone.onnx`: BEV backbone+head, **`canvas_feature -> (scores, bbox_preds, dir_scores)`**

Important: **voxelization / scatter are *not* inside ONNX** in Apollo's
implementation. They are done by C++/CUDA pre/post-processing.

## Contract (must match car-side inference)

From `modules/perception/lidar_cpdet_detection/data/cpdet_param.pb.txt` on Apollo business version and the
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

Recommended pinned environment for this repo:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The pinned stack is:

- `Python 3.10.12`
- `PyTorch 2.1.2 + cu121`
- `MMEngine 0.10.5`
- `MMCV 2.1.0`
- `MMDetection 3.3.0`
- `MMDetection3D 1.4.0`

The top-level [requirements.txt](/home/soon/Desktop/center_point_trt_mmdet3d/requirements.txt) points to the pinned CUDA 12.1 stack in [requirements-cu121.txt](/home/soon/Desktop/center_point_trt_mmdet3d/requirements/requirements-cu121.txt).

If you prefer to install the pinned file directly:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/requirements-cu121.txt
```

## Usage

### 1) Train (MMDetection3D / MMEngine)

Before training on nuScenes, prepare the MMDetection3D info `.pkl` files with
the installed official converter:

```bash
bash tools/prepare_nuscenes_data.sh \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag nuscenes
```

This wrapper uses the currently active `python` / venv to locate the installed
`mmdet3d` package and then calls its official `create_data.py`.

For `v1.0-mini`, pass the version through to the same wrapper:

```bash
bash tools/prepare_nuscenes_data.sh \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini
```

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

This repo includes a local training entrypoint:

```bash
python3 tools/train.py \
  mmdet3d_example_configs/centerpoint_trt_nuscenes_4task_train.py
```

For a `v1.0-mini` smoke run with the same model / schedule:

```bash
python3 tools/train.py \
  mmdet3d_example_configs/centerpoint_trt_nuscenes_4task_mini.py
```

The bundled nuScenes config is a runnable scaffold for MMDetection3D 1.x, but
you still need to edit dataset paths / info files for your environment.

Common variants:

```bash
python3 tools/train.py \
  mmdet3d_example_configs/centerpoint_trt_nuscenes_4task_train.py \
  --amp
```

```bash
python3 tools/train.py \
  mmdet3d_example_configs/centerpoint_trt_nuscenes_4task_train.py \
  --resume
```

Do not mix up these three categories:

- `--amp`
  - training acceleration only
  - switches `optim_wrapper` from `OptimWrapper` to `AmpOptimWrapper`
  - reduces memory / may improve throughput
  - does **not** change points, labels, targets, or model semantics
- data augmentation in the dataset pipeline
  - current nuScenes example uses `GlobalRotScaleTrans`, `RandomFlip3D`, `PointShuffle`
  - only active in the training pipeline
  - changes the sampled training data distribution on purpose to improve robustness
- Apollo-aligned preprocessing
  - implemented by `ApolloBevFeatureGenerator` inside the model
  - includes 45 degree xy rotation, 9D voxel feature construction, scatter-max, and 16-channel cnnseg features
  - must stay aligned with car-side inference if you want train/inference consistency

In short:

- `--amp` = faster / lighter training
- data augmentation = perturb training samples
- Apollo preprocessing = inference contract you are trying to learn against

Important training details in the bundled config:

- fixed Apollo-aligned preprocessing (`enable_rotate_45degree=True`, 9D voxel feature, 48+16 BEV channels)
- 4 tasks / 4 output heads: `car`, `pedestrian`, `bicycle`, `traffic_cone`
- nuScenes 10-class -> 4-task class remapping via `ApolloMapClasses3D`
- internal merged-class BEV mAP validation via `ApolloMergedClassMetric3D`
- training-time augmentation in the example config:
  - `GlobalRotScaleTrans`: random rotation / scale perturbation
  - `RandomFlip3D`: random BEV flips
  - `PointShuffle`: random point order shuffle
- default optimizer / schedule:
  - `AdamW(lr=2e-4, betas=(0.95, 0.99), weight_decay=0.01)`
  - `20` epochs
  - linear warmup for first `1000` iters + cosine annealing
- early stopping:
  - monitors `apollo/mAP`
  - starts checking from epoch `5`
  - stops if there is no improvement for `5` validation epochs
  - best checkpoint is still tracked by `CheckpointHook(save_best="apollo/mAP")`

### 2) Export ONNX

After training, export with:

```bash
python3 tools/export_onnx.py \
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
python3 tools/export_onnx.py \
  --config /path/to/your_mmdet3d_cfg.py \
  --strip-identity \
  --out-dir /tmp/center_point_trt_onnx
```

### Diff Apollo ONNX vs exported ONNX (structure check)

Install `onnx` in the same env, then run:

```bash
python3 tools/diff_onnx.py \
  --a modules/perception/production/data/perception/lidar/models/detection/center_point_trt/cpdet_backbone.onnx \
  --b /tmp/center_point_trt_onnx/cpdet_backbone.onnx
```
