# Training Recipe

本文档说明当前仓库的模型训练细节，重点是：

- 当前训练到底做了哪些事情
- 每一项的作用是什么
- 哪些项是为了 **Apollo 车端一致性**
- 哪些项是为了 **提升训练效果**
- 哪些项只是为了 **调试和观察训练状态**

对应主配置文件：

- `mmdet3d_example_configs/centerpoint_trt_nuscenes_4task_train.py`


## 1. 训练目标

当前目标不是复刻官方 nuScenes CenterPoint 的完整训练结构，而是：

- 保持 **Apollo 风格前处理 / PFE / BEV 语义**
- 在 MMDetection3D 1.x 框架里训练
- 最终仍然服务于 Apollo 的 4-task 推理项

当前 4 个 task 是：

- `car`
- `pedestrian`
- `bicycle`
- `traffic_cone`

nuScenes 与 Apollo 当前类别映射是：

- `car <- car / truck / construction_vehicle / bus / trailer`
- `pedestrian <- pedestrian`
- `bicycle <- bicycle / motorcycle`
- `traffic_cone <- traffic_cone`

注意：

- `barrier` 当前 **不并入** `traffic_cone`
- 这是有意保持 task 纯度，避免污染 `traffic_cone` 的训练和评估


## 2. Apollo 一致性相关项

这些配置首先服务于 **车端前处理一致性**，不是单纯为了追求 nuScenes 指标。

### 2.1 Apollo BEV 特征生成

训练时并不是直接用官方 PointPillars 的标准 voxelization，而是先做 Apollo 风格的特征构造：

- 点云过滤
- 45° 旋转坐标
- 手工 9 维 voxel 特征
- CNNSeg 风格额外 BEV 特征

对应模块：

- `apollo_centerpoint_trt/bev_feature.py`

作用：

- 保证训练时看到的特征分布，尽量接近 Apollo 车端 C++/CUDA 前处理
- 降低“训练好、上线分布不一致”的风险


### 2.2 固定 45° 旋转

当前配置中：

- `enable_rotate_45degree=True`

作用：

- 对齐 Apollo 车端的坐标处理方式
- 训练时 GT 会同步旋转
- 推理输出时会再旋回 lidar 坐标

这项主要是 **车端一致性项**。


### 2.3 Apollo-lite PFE / Backbone / Neck

当前用的是自定义模块：

- `ApolloPFE`
- `ApolloSecondBackboneLite`
- `ApolloNeckLite`

对应文件：

- `apollo_centerpoint_trt/pfe.py`
- `apollo_centerpoint_trt/backbone.py`
- `apollo_centerpoint_trt/neck.py`

作用：

- 尽量贴 Apollo 导出结构和推理合同
- 让后续 ONNX 导出、Apollo 侧对接更直接

代价：

- 训练效果上限通常不如官方成熟的 nuScenes CenterPoint 结构


## 3. 数据与训练增强

这部分主要服务于 **提升训练效果**，与 nuScenes CenterPoint 对齐。

### 3.1 单帧 + 多 sweeps

当前训练和验证都使用：

- 当前帧点云
- 额外 `9` 个 sweeps

作用：

- 提高时序信息密度
- 提升远距离和稀疏目标稳定性
- 对行人、自行车、小目标通常有帮助


### 3.2 GT Database Sampling

当前启用了：

- `ObjectSample`
- `db_sampler`

使用的数据来自：

- `nuscenes_dbinfos_train.pkl`
- `nuscenes_gt_database/`

作用：

- 给长尾类补样本
- 提高 `pedestrian / bicycle / traffic_cone` 的出现频率
- 缓解训练集类不平衡


### 3.3 CBGSDataset

当前训练集外层包了：

- `CBGSDataset`

作用：

- 做类均衡采样
- 降低长尾类被高频类淹没的问题

这是当前相对接近官方 CenterPoint nuScenes 训练的一项。


### 3.4 几何增强

当前启用了：

- `GlobalRotScaleTrans`
- `RandomFlip3D`
- `PointShuffle`

作用：

- 增强朝向、尺度、左右场景鲁棒性
- 提升泛化能力

注意：

- 这些增强属于 **训练增强**
- 不是 Apollo 车端前处理的一部分


### 3.5 Range Filter

当前启用了：

- `PointsRangeFilter`
- `ObjectRangeFilter`

作用：

- 限定训练和评估的空间范围
- 保持训练、验证和模型检测范围口径一致

注意：

- 这属于“检测范围裁剪”
- 不等于 Apollo 车端完整 ROI filter


## 4. 模型头与任务定义

### 4.1 4-task CenterHead

当前 head 仍然是 CenterPoint 风格的 `CenterHead`，但 task 被固定为 4 个：

- `car`
- `pedestrian`
- `bicycle`
- `traffic_cone`

作用：

- 保持 Apollo 推理代码不变
- 避免推理侧需要改成 4 以上 task

代价：

- `car` task 内部会混合大车和小车
- `bicycle` task 内部会混合自行车和摩托车
- 类内分布更复杂，训练会更难


### 4.2 velocity 分支

当前 head 保留了 `vel` 分支。

作用：

- 兼容 nuScenes 训练中的字段要求
- 避免 `CenterHead` 在训练时缺少 `vel` 而报错

注意：

- 导出模型给 Apollo 时不会直接把 `vel` 当成最终推理项输出


## 5. 优化器与学习率策略

### 5.1 Optimizer

当前使用：

- `AdamW`

作用：

- 对检测训练通常比较稳定
- 对当前这类从零训练结构更友好


### 5.2 官方风格 cyclic 调度

当前已改为更接近官方 CenterPoint nuScenes 的两段调度：

- 前半段 cosine LR
- 后半段 cosine LR
- 同时配合 cosine momentum

作用：

- 前期快速收敛
- 后期细化收敛
- 比简单 warmup + cosine 更接近官方模型训练效果


### 5.3 Batch Size

当前训练 `batch_size=4`。

作用：

- 比 `batch_size=2` 梯度更稳
- 更接近官方 nuScenes 配置

注意：

- 显存不够时，可以退回 `2`


### 5.4 Gradient Clipping

当前启用了：

- `clip_grad`

作用：

- 防止梯度爆炸
- 在自定义 Apollo 前处理特征下，训练更稳


## 6. 推理后处理相关

### 6.1 test_cfg

当前调了：

- `min_radius`
- `pre_max_size`
- `post_max_size`
- `nms_thr`

作用：

- 控制 decode 和 NMS 行为
- 直接影响 AP、召回、误检数量

说明：

- 对齐 CenterPoint nuScenes 的 config


## 7. 评估指标

当前不是只看一个指标，而是同时看两类。

### 7.1 ApolloMergedClassMetric3D

输出：

- `apollo/AP/...`
- `apollo/mAP`

作用：

- 看排序、框质量、NMS 之后的整体检测表现

注意：

- 这是项目内部 4-task 指标
- **不是**官方 nuScenes metric


### 7.2 ApolloCenterRecallMetric3D

输出：

- `Recall@0.5m/...`
- `mRecall@0.5m`

作用：

- 看“目标中心点有没有被检出来”
- 比 BEV AP 更接近业务上“车能不能检测出来”的问题

适用场景：

- 当业务允许几十厘米误差时，这个指标往往比 AP 更有解释力


### 7.3 训练集评估

当前额外启用了：

- `ApolloTrainEvalHook`

作用：

- 每个 epoch 后，用训练集原始分布再评一次
- 输出：
  - `train_apollo/...`
  - `train_center/...`

意义：

- debug: 区分“训练流程没学会（训练流程本身有问题）”还是“泛化差（数据、超参需要调整）”
- 如果训练集 Recall 都很低，就不是单纯的数据泛化问题


## 8. 早停与 checkpoint

当前启用了：

- `ApolloEarlyStoppingHook`
- `CheckpointHook(save_best="apollo/mAP")`

作用：

- 当验证集 `apollo/mAP` 长时间不涨时提前停止
- 自动保存 best checkpoint

说明：

- 当前早停监控的是验证集 `apollo/mAP`
- 不是训练集指标


## 10. 当前最值得关注的指标

业务更关心：

- 车能不能被检测出来
- 中心点偏几十厘米是否可接受

那最该重点看的是：

- `train_center/Recall@0.5m/car`
- `apollo_center/Recall@0.5m/car`

其次再看：

- `train_apollo/AP/car`
- `apollo/AP/car`
