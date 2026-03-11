_base_ = ["./centerpoint_trt_nuscenes_4task_train.py"]


data_root = "data/custom_lidar/"

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="ApolloRangeFilter3D",
        point_cloud_range=[-51.2, -51.2, -3.5, 51.2, 51.2, 3.5],
        enable_rotate_45degree=True,
    ),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

eval_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(
        type="ApolloRangeFilter3D",
        point_cloud_range=[-51.2, -51.2, -3.5, 51.2, 51.2, 3.5],
        enable_rotate_45degree=True,
    ),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CustomLidarDataset",
        data_root=data_root,
        ann_file="converted/infos/custom_infos_train.pkl",
        metainfo=dict(classes=["car", "pedestrian", "bicycle", "traffic_cone"]),
        pipeline=train_pipeline,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CustomLidarDataset",
        data_root=data_root,
        ann_file="converted/infos/custom_infos_val.pkl",
        metainfo=dict(classes=["car", "pedestrian", "bicycle", "traffic_cone"]),
        pipeline=eval_pipeline,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CustomLidarDataset",
        data_root=data_root,
        ann_file="converted/infos/custom_infos_test.pkl",
        metainfo=dict(classes=["car", "pedestrian", "bicycle", "traffic_cone"]),
        pipeline=eval_pipeline,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

train_eval_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CustomLidarDataset",
        data_root=data_root,
        ann_file="converted/infos/custom_infos_train.pkl",
        metainfo=dict(classes=["car", "pedestrian", "bicycle", "traffic_cone"]),
        pipeline=eval_pipeline,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

work_dir = "./work_dirs/centerpoint_trt_custom_4task"
