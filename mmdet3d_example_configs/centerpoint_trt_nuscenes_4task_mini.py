_base_ = ["./centerpoint_trt_nuscenes_4task_train.py"]

# Smoke / small-scale training config for nuScenes v1.0-mini.
#
# In some MMDetection3D 1.x installs, `tools/create_data.py --version v1.0-mini`
# still writes `nuscenes_infos_train.pkl` / `nuscenes_infos_val.pkl` instead of
# `nuscenes_mini_infos_*.pkl`. This config follows that actual output behavior.

train_dataloader = dict(
    dataset=dict(
        ann_file="nuscenes_infos_train.pkl",
    ),
)

val_dataloader = dict(
    dataset=dict(
        ann_file="nuscenes_infos_val.pkl",
    ),
)

test_dataloader = dict(
    dataset=dict(
        ann_file="nuscenes_infos_val.pkl",
    ),
)

# Keep the same total epochs so the small dataset can overfit if needed; this
# is useful for smoke-testing whether loss decreases and the custom metric moves.
work_dir = "./work_dirs/centerpoint_trt_nuscenes_4task_mini"
