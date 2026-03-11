# Custom LiDAR Dataset Format

本文档定义一套**仓库内的中间数据格式**，用于接入业务录制点云数据。

目标不是复刻 nuScenes 原始格式，而是提供一套更小、更稳定的格式，满足：

- 原始 `pcd` / 标注文件可转换进来
- 可继续生成训练/验证/测试 `info pkl`
- 可与 nuScenes 一起混训
- 可保留 Apollo 训练所需的坐标、强度、类别语义

## 1. 设计原则

- 原始数据与训练数据解耦  
  原始 `pcd`、标注工具导出的 json/xml，不直接作为训练输入。
- 统一到仓库内部格式  
  所有业务数据先转成同一套中间格式，再写 dataset / info builder。
- 坐标系显式记录  
  不假设所有业务数据天然是 Apollo 的 `x前 y左 z上`。
- 语义统一到 4 类  
  `car / pedestrian / bicycle / traffic_cone`
- scene 级切分  
  train/val/test 必须按场景切，不按帧随机切。

## 2. 目录结构

```text
data/custom_lidar/
  raw/
    scene_0001/
      lidar/
        000000.pcd
        000001.pcd
      labels/
        000000.json
        000001.json
      calib.json
      scene_meta.json
    scene_0002/
      ...
  converted/
    points/
      scene_0001/
        000000.bin
        000001.bin
    labels/
      scene_0001/
        000000.json
        000001.json
    infos/
      custom_infos_train.pkl
      custom_infos_val.pkl
      custom_infos_test.pkl
    splits/
      train.txt
      val.txt
      test.txt
```

说明：

- `raw/` 保留原始文件，便于回溯
- `converted/points/*.bin` 作为训练实际读取的点云
- `converted/labels/*.json` 作为统一后的标注
- `converted/infos/*.pkl` 对应训练入口实际使用的索引文件

## 3. 点云文件格式

训练侧统一使用 `.bin`，每个点一行 `float32`，推荐字段：

```text
[x, y, z, intensity]
```

即：

- shape: `[N, 4]`
- dtype: `float32`

### intensity 约定

统一要求写成：

- `0 ~ 255`

原因：

- 当前 Apollo 前处理默认按 `intensity_scale=255.0` 使用
- 这样与当前训练实现保持一致

如果原始数据是 `0 ~ 1`：

- 转换脚本里直接乘 `255`

如果原始点云还带其他字段，如：

- `ring`
- `timestamp`
- `elongation`

先不进入训练主输入；需要时可额外保存在原始 meta 中。

## 4. 坐标系约定

中间格式里的训练坐标系统一定义为：

- `x`: forward
- `y`: left
- `z`: up

也就是 Apollo / 常见 LiDAR 检测坐标系。

### 如果原始数据不是这个坐标系

转换阶段必须做显式变换，并在 `scene_meta.json` 或 annotation meta 里记录：

- `source_coord`
- `target_coord`
- `transform_matrix`

推荐支持这些源坐标描述：

- `apollo_lidar`
- `nuscenes_lidar`
- `sensor_native`
- `unknown`

如果是 `unknown`，就必须在转换脚本参数里手工指定变换。

## 5. 标注文件格式

每一帧一个 json，结构暂定如下：

```json
{
  "frame_id": "000000",
  "scene_id": "scene_0001",
  "timestamp_us": 1710000000000,
  "point_cloud_path": "converted/points/scene_0001/000000.bin",
  "coord_system": "apollo_lidar",
  "objects": [
    {
      "id": "obj_001",
      "class_name": "car",
      "bbox_3d": {
        "center": [12.34, -1.25, -0.80],
        "size": [4.20, 1.85, 1.62],
        "yaw": 1.57
      },
      "num_lidar_pts": 128,
      "attributes": {
        "source_label": "car"
      }
    }
  ]
}
```

### 字段定义

- `frame_id`
  - 当前帧编号，字符串，建议与文件名一致
- `scene_id`
  - 场景编号
- `timestamp_us`
  - 微秒时间戳，可选但强烈推荐
- `point_cloud_path`
  - 指向转换后的 `.bin`
- `coord_system`
  - 该标注当前所在坐标系，期望为 `apollo_lidar`
- `objects`
  - 当前帧全部 3D 标注框

### `bbox_3d` 定义

- `center = [x, y, z]`
  - 建议为**框几何中心**
- `size = [length, width, height]`
  - 顺序固定：`l, w, h`
- `yaw`
  - 绕 `z` 轴旋转，弧度制

## 6. 类别定义

训练目标统一映射到：

- `car`
- `pedestrian`
- `bicycle`
- `traffic_cone`

### 语义映射

- `car / truck / bus / trailer / construction_vehicle -> car`
- `pedestrian -> pedestrian`
- `bicycle / motorcycle -> bicycle`
- `traffic_cone -> traffic_cone`

不并入：

- `barrier`

因为它和 `traffic_cone` 语义差异太大。

### 标注工具导出保留原始类

建议 `attributes.source_label` 保留标注工具原始类别，映射在转换阶段做：

- 原始标签保留
- 训练标签另算

## 7. 可选多帧 / sweep 支持

如果业务数据有连续帧，建议在 annotation 或 info 里补：

```json
"sweeps": [
  {
    "time_lag": -0.1,
    "point_cloud_path": "converted/points/scene_0001/000000_prev1.bin",
    "transform_matrix": [[...], [...], [...], [...]]
  }
]
```

约定：

- `time_lag < 0` 表示过去帧
- `transform_matrix` 表示 sweep 点到当前主帧坐标系的变换

如果没有可靠 ego pose / 外参：

- 第一版只支持单帧训练

## 8. Scene 元信息文件

每个 scene 建议有一个 `scene_meta.json`：

```json
{
  "scene_id": "scene_0001",
  "sensor_name": "lidar_top",
  "coord_system": "apollo_lidar",
  "has_sweeps": true,
  "frame_count": 324,
  "source_dataset": "business_recording_v1"
}
```

这个文件主要用于：

- 切分 train/val/test
- 追踪来源
- 后续统计数据质量

## 9. train/val/test 切分规范

统一用 scene 级 txt 文件：

```text
converted/splits/train.txt
converted/splits/val.txt
converted/splits/test.txt
```

内容示例：

```text
scene_0001
scene_0003
scene_0008
```

要求：

- 同一 scene 只能出现在一个 split
- 不允许按帧随机切分

## 10. 训练 info 的目标字段

最终写进 `custom_infos_*.pkl` 的每条样本，建议至少包含：

- `sample_idx`
- `scene_id`
- `timestamp`
- `lidar_points.lidar_path`
- `instances`
  - `bbox_3d`
  - `bbox_label_3d`
  - `bbox_label_name`
  - `num_lidar_pts`
- `sweeps`（可选）
- `metainfo`
  - `source_dataset`
  - `coord_system`

其中 `instances[*].bbox_3d` 统一用：

```text
[x, y, z, l, w, h, yaw]
```

## 11. 与 nuScenes 混训的建议

- nuScenes 维持现有 `NuScenesDataset`
- 业务数据走 `CustomLidarDataset`
- 通过：
  - `ConcatDataset`
  - 或加一层自定义混合采样

## 12. 目前支持的功能

- 单帧点云
- `.pcd -> .bin`
- 每帧 json 标注
- 4 类标签映射
- scene 级 split
- 生成 `custom_infos_train.pkl / val.pkl / test.pkl`

后续考虑支持：

- 复杂属性
- 地图
- Apollo 原始 proto 数据直读

## 13. 后续仓库新增文件 for 混合训练

- `custom_lidar_tool/label_adapters/*.py`
- `custom_lidar_tool/info_builder.py`
- `apollo_centerpoint_trt/datasets/custom_lidar_dataset.py`
- `apollo_centerpoint_trt/transforms/coord_convert_3d.py`

## 15. 使用说明

命令行入口：

- `custom_lidar_dataset.py`

### 初始化目录和 scene meta

```bash
python3 custom_lidar_dataset.py init \
  --root data/custom_lidar \
  --scene scene_0001 \
  --source-coord sensor_native \
  --target-coord apollo_lidar
```

如果需要坐标变换，还支持：

- `--transform-file path/to/transform.json`
- `--transform-values "16个逗号分隔浮点数"`

### 点云编码

把原始 `.pcd` 转成训练使用的 `[x,y,z,intensity] float32` `.bin`：

```bash
python3 custom_lidar_dataset.py encode-pcd \
  --root data/custom_lidar \
  --scene scene_0001 \
  --input /path/to/pcd_dir
```

当前实现支持：

- `ascii` PCD
- `binary` PCD

暂不支持：

- `binary_compressed` PCD

### 标注转换接口（当前为占位）

```bash
python3 custom_lidar_dataset.py encode-label \
  --root data/custom_lidar \
  --scene scene_0001 \
  --input /path/to/label_export \
  --label-format labelcloud
```

当前这个命令只创建 adapter stub，保留接口，等待后续确定标注工具后再补真正转换逻辑。

### 生成 info pkl 与 split

```bash
python3 custom_lidar_dataset.py convert \
  --root data/custom_lidar \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 0
```

这个命令会生成：

- `converted/splits/train.txt`
- `converted/splits/val.txt`
- `converted/splits/test.txt`
- `converted/infos/custom_infos_train.pkl`
- `converted/infos/custom_infos_val.pkl`
- `converted/infos/custom_infos_test.pkl`

### 接进训练配置

仓库已经提供了最小训练配置入口：

- `mmdet3d_example_configs/centerpoint_trt_custom_4task_train.py`

对应的数据集类已经注册到插件里：

- `apollo_centerpoint_trt/datasets/custom_lidar_dataset.py`

所以后续只要 `custom_infos_*.pkl` 已经生成，就可以直接训练：

```bash
python3 tools/train.py \
  mmdet3d_example_configs/centerpoint_trt_custom_4task_train.py
```

这份配置当前假设：

- `data_root = data/custom_lidar/`
- 点云输入是 `load_dim=4, use_dim=4`
- 不使用 sweeps
- 直接复用 Apollo 前处理和 4-task head
