# This is an *example model snippet* for training an CenterPointTRT
# with MMDetection3D.
#
# You still need to provide:
# - dataset / dataloader / runtime / optimizer settings (project-specific)
# - a proper `point_cloud_range` / `voxel_size` consistent with Apollo config
#
# The key is the model contract:
# - PFE: voxels(9d) -> pillar_feature(48d)
# - Scatter + extra BEV features (16d) -> canvas_feature(64, 512, 512)
# - CenterPoint backbone+neck+head -> (bbox_preds, scores, dir_scores)

custom_imports = dict(
    imports=[
        # Add this folder to PYTHONPATH in your training env, then import:
        "apollo_centerpoint_trt",
    ],
    allow_failed_imports=False,
)

# Apollo-like preprocessing parameters (match C++ config).
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
    intensity_scale=255.0,
    use_cnnseg_features=True,
    height_bin_min_height=-3.0,
    height_bin_max_height=2.0,
    height_bin_voxel_size=0.5,
    pillar_feature_dim=48,
    cnnseg_feature_dim=16,
)

pfe_cfg = dict(in_channels=9, out_channels=48)

trt_binding_cfg = dict(
    input_voxels="voxels",
    pillar_feature_blob="pillar_feature",
    input_canvas_feature="canvas_feature",
    output_box="bbox_preds",
    output_cls="scores",
    output_dir="dir_scores",
)

# For empty-export inspection, avoid building MMDet3D `CenterPoint` detector
# directly (its init signature differs across versions). Instead, build only
# the modules needed by Apollo export: backbone + neck + bbox_head.

pts_backbone = dict(
    # Use a custom backbone to avoid MMDet3D `SECOND` signature drift and match
    # Apollo's shipped ONNX (Conv+ReLU, no BN).
    type="ApolloSecondBackboneLite",
    in_channels=64,
    stem_out_channels=64,
    stage_channels=[64, 128, 256],
    stage_blocks=[3, 5, 5],
)

pts_neck = dict(
    # Apollo neck is not SECONDFPN; it is a light 3-branch align+concat neck.
    type="ApolloNeckLite",
    in_channels=[64, 128, 256],
    out_channels=128,
)

train_cfg = dict(
    pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 7.0],
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0] * 7,
    )
)

test_cfg = dict(
    pts=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 4, 4, 4],
        score_threshold=0.1,
        task_score_thresholds=[0.25, 0.18, 0.40, 0.35],
        pc_range=[-51.2, -51.2],
        out_size_factor=4,
        voxel_size=[0.2, 0.2],
        nms_type="rotate",
        pre_max_size=4096,
        post_max_size=500,
        nms_thr=0.2,
    )
)

pts_bbox_head = dict(
    type="CenterHead",
    in_channels=384,
    tasks=[
        # Some MMDet3D versions use `num_class`, some use `num_classes`.
        dict(num_class=1, num_classes=1, class_names=["Car"]),
        dict(num_class=1, num_classes=1, class_names=["Pedestrian"]),
        dict(num_class=1, num_classes=1, class_names=["Bicycle"]),
        dict(num_class=1, num_classes=1, class_names=["TrafficCone"]),
    ],
    common_heads=dict(
        reg=(2, 2),
        height=(1, 2),
        dim=(3, 2),
        rot=(2, 2),
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
        code_size=7,
    ),
    separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
    # In MMDet3D 1.4.0, these losses live in MMDet's registry scope.
    loss_cls=dict(_scope_="mmdet", type="GaussianFocalLoss", reduction="mean"),
    loss_bbox=dict(
        _scope_="mmdet", type="L1Loss", reduction="mean", loss_weight=0.25
    ),
    norm_bbox=True,
    norm_cfg=None,
    train_cfg=train_cfg.get("pts"),
    test_cfg=test_cfg.get("pts"),
)

from apollo_centerpoint_trt.export_model import CenterPointTRTExportModel  # noqa: E402

model = CenterPointTRTExportModel(
    pts_backbone=pts_backbone,
    pts_neck=pts_neck,
    pts_bbox_head=pts_bbox_head,
    pfe_cfg=pfe_cfg,
)
