# Example *training* config scaffold for MMDetection3D 1.x on nuScenes.
#
# This file is intentionally self-contained (plain python) so it can also be
# used by `tools/export_onnx.py` (empty-export path) without MMEngine lazy/base
# indirections.
#
# You still need to set:
# - `data_root`
# - `ann_file` paths (nuscenes infos pkl)

custom_imports = dict(
    imports=[
        "apollo_centerpoint_trt",
    ],
    allow_failed_imports=False,
)

# -------------------------
# Apollo-aligned preprocessing
# -------------------------
bev_feature_cfg = dict(
    min_x_range=-51.2,
    max_x_range=51.2,
    min_y_range=-51.2,
    max_y_range=51.2,
    min_z_range=-3.5,
    max_z_range=3.5,
    voxel_x_size=0.2,
    voxel_y_size=0.2,
    voxel_z_size=7.0,
    enable_rotate_45degree=True,
    use_input_norm=True,
    intensity_scale=255.0,  # nuScenes lidar intensity is 0..255 in many dumps
    use_cnnseg_features=True,
    height_bin_min_height=-3.0,
    height_bin_max_height=2.0,
    height_bin_voxel_size=0.5,
    pillar_feature_dim=48,
    cnnseg_feature_dim=16,
)

pfe_cfg = dict(in_channels=9, out_channels=48)
point_cloud_range = [-51.2, -51.2, -3.5, 51.2, 51.2, 3.5]

# -------------------------
# CenterPoint backbone/neck/head (Apollo-lite)
# -------------------------
pts_backbone = dict(
    type="ApolloSecondBackboneLite",
    in_channels=64,
    stem_out_channels=64,
    stage_channels=[64, 128, 256],
    stage_blocks=[3, 5, 5],
)

pts_neck = dict(
    type="ApolloNeckLite",
    in_channels=[64, 128, 256],
    out_channels=128,
)

head_train_cfg = dict(
    pts=dict(
        point_cloud_range=point_cloud_range,
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 7.0],
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
    )
)

head_test_cfg = dict(
    pts=dict(
        point_cloud_range=point_cloud_range,
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 0.85, 0.85, 0.175],
        score_threshold=0.1,
        pc_range=[-51.2, -51.2],
        out_size_factor=4,
        voxel_size=[0.2, 0.2],
        nms_type="rotate",
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
    )
)

# Target task order matches Apollo TRT contract: [car, ped, bicycle, traffic_cone]
tasks = [
    dict(num_class=1, num_classes=1, class_names=["car"]),
    dict(num_class=1, num_classes=1, class_names=["pedestrian"]),
    dict(num_class=1, num_classes=1, class_names=["bicycle"]),
    dict(num_class=1, num_classes=1, class_names=["traffic_cone"]),
]

pts_bbox_head = dict(
    type="CenterHead",
    in_channels=384,
    tasks=tasks,
    # nuScenes training in MMDet3D expects a velocity branch. Apollo export
    # still drops `vel` and only packs reg/height/dim/rot-derived outputs.
    common_heads=dict(
        reg=(2, 2),
        height=(1, 2),
        dim=(3, 2),
        rot=(2, 2),
        vel=(2, 2),
    ),
    share_conv_channel=64,
    bbox_coder=dict(
        type="CenterPointBBoxCoder",
        pc_range=[-51.2, -51.2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2],
        code_size=9,
    ),
    separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
    loss_cls=dict(_scope_="mmdet", type="GaussianFocalLoss", reduction="mean"),
    loss_bbox=dict(_scope_="mmdet", type="L1Loss", reduction="mean", loss_weight=0.25),
    norm_bbox=True,
    norm_cfg=None,
    train_cfg=head_train_cfg.get("pts"),
    test_cfg=head_test_cfg.get("pts"),
)

model = dict(
    type="CenterPointTRTDetector",
    bev_feature_cfg=bev_feature_cfg,
    pfe_cfg=pfe_cfg,
    centerpoint=dict(
        pts_backbone=pts_backbone,
        pts_neck=pts_neck,
        pts_bbox_head=pts_bbox_head,
    ),
    # Keep default preprocessor behavior (turn on/off in your stack as needed).
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
)

# -------------------------
# nuScenes dataset scaffold
# -------------------------
#
# Expected layout for this config:
# data/
#   nuscenes_infos_train.pkl
#   nuscenes_infos_val.pkl
#   nuscenes_dbinfos_train.pkl
#   nuscenes_gt_database/
#   samples/
#   sweeps/
#   maps/
#   v1.0-*/
data_root = "data/"

# nuScenes canonical 10 classes (MMDet3D nuScenes converters usually follow this order).
nuscenes_classes = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Map nuScenes classes -> Apollo 4-task label space [0..3].
# Adjust this mapping based on your intended semantics.
class_mapping = {
    "car": 0,
    "truck": 0,
    "construction_vehicle": 0,
    "bus": 0,
    "trailer": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "motorcycle": 2,
    "traffic_cone": 3,
}

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "nuscenes_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            construction_vehicle=5,
            bus=5,
            trailer=5,
            pedestrian=5,
            motorcycle=5,
            bicycle=5,
            traffic_cone=5,
        ),
    ),
    classes=[
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
    ],
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        pedestrian=6,
        motorcycle=6,
        bicycle=6,
        traffic_cone=2,
    ),
    points_loader=dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
)

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="ObjectSample", db_sampler=db_sampler),
    # Training-time augmentation. These are not part of Apollo car-side preprocessing.
    dict(
        type="ApolloRangeFilter3D",
        point_cloud_range=point_cloud_range,
        enable_rotate_45degree=bev_feature_cfg["enable_rotate_45degree"],
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.0, 0.0, 0.0],
    ),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(type="ApolloMapClasses3D", mapping=class_mapping, src_classes=nuscenes_classes),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

val_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="ApolloRangeFilter3D",
        point_cloud_range=point_cloud_range,
        enable_rotate_45degree=bev_feature_cfg["enable_rotate_45degree"],
    ),
    dict(type="ApolloMapClasses3D", mapping=class_mapping, src_classes=nuscenes_classes),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,
    ),
    dict(
        type="ApolloRangeFilter3D",
        point_cloud_range=point_cloud_range,
        enable_rotate_45degree=bev_feature_cfg["enable_rotate_45degree"],
    ),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            type="NuScenesDataset",
            data_root=data_root,
            ann_file="nuscenes_infos_train.pkl",
            data_prefix=dict(
                pts="samples/LIDAR_TOP",
                sweeps="sweeps/LIDAR_TOP",
            ),
            metainfo=dict(classes=nuscenes_classes),
            pipeline=train_pipeline,
            test_mode=False,
            use_valid_flag=True,
        ),
    ),
    collate_fn=dict(type="pseudo_collate"),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="NuScenesDataset",
        data_root=data_root,
        ann_file="nuscenes_infos_val.pkl",
        data_prefix=dict(
            pts="samples/LIDAR_TOP",
            sweeps="sweeps/LIDAR_TOP",
        ),
        metainfo=dict(classes=nuscenes_classes),
        pipeline=val_pipeline,
        test_mode=False,
        use_valid_flag=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="NuScenesDataset",
        data_root=data_root,
        ann_file="nuscenes_infos_val.pkl",
        data_prefix=dict(
            pts="samples/LIDAR_TOP",
            sweeps="sweeps/LIDAR_TOP",
        ),
        metainfo=dict(classes=nuscenes_classes),
        # Keep GT in the test loop because ApolloMergedClassMetric3D matches
        # predictions against packed `gt_instances_3d`.
        pipeline=val_pipeline,
        test_mode=False,
        use_valid_flag=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

# -------------------------
# Evaluator / metric
# -------------------------
#
# IMPORTANT:
# - nuScenes official metric is defined on 10 canonical classes.
# - This example collapses many classes into 4 tasks, so we provide a simple
#   internal BEV mAP metric as a sanity check (not the official nuScenes score).
#
val_evaluator = [
    dict(
        type="ApolloMergedClassMetric3D",
        class_names=["car", "pedestrian", "bicycle", "traffic_cone"],
        iou_thr=[0.5, 0.25, 0.25, 0.25],
        max_dets=500,
        score_thr=0.0,
        center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    ),
    dict(
        type="ApolloCenterRecallMetric3D",
        prefix="apollo_center",
        class_names=["car", "pedestrian", "bicycle", "traffic_cone"],
        distance_thr=0.5,
        max_dets=500,
        score_thr=0.0,
        center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    ),
]

test_evaluator = val_evaluator

train_eval_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="NuScenesDataset",
        data_root=data_root,
        ann_file="nuscenes_infos_train.pkl",
        data_prefix=dict(
            pts="samples/LIDAR_TOP",
            sweeps="sweeps/LIDAR_TOP",
        ),
        metainfo=dict(classes=nuscenes_classes),
        pipeline=val_pipeline,
        test_mode=False,
        use_valid_flag=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

train_evaluator = [
    dict(
        type="ApolloMergedClassMetric3D",
        prefix="train_apollo",
        class_names=["car", "pedestrian", "bicycle", "traffic_cone"],
        iou_thr=[0.5, 0.25, 0.25, 0.25],
        max_dets=500,
        score_thr=0.0,
        center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    ),
    dict(
        type="ApolloCenterRecallMetric3D",
        prefix="train_center",
        class_names=["car", "pedestrian", "bicycle", "traffic_cone"],
        distance_thr=0.5,
        max_dets=500,
        score_thr=0.0,
        center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    ),
]

# -------------------------
# Runtime / optimization
# -------------------------
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=20, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    # `tools/train.py --amp` upgrades this wrapper to `AmpOptimWrapper`.
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=8,
        eta_min=1e-4 * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=12,
        eta_min=1e-4 * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=8,
        eta_min=0.8947368421052632,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=12,
        eta_min=1.0,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        save_best="apollo/mAP",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

custom_hooks = [
    dict(
        type="ApolloTrainEvalHook",
        dataloader=train_eval_dataloader,
        evaluator=train_evaluator,
        interval=1,
        start_epoch=1,
    ),
    dict(
        type="ApolloEarlyStoppingHook",
        monitor="apollo/mAP",
        rule="greater",
        patience=5,
        min_delta=1e-4,
        start_epoch=5,
        strict=False,
    ),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", by_epoch=True, window_size=50)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
default_scope = "mmdet3d"
work_dir = "./work_dirs/centerpoint_trt_nuscenes_4task"
