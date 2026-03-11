from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover
  from mmdet3d.datasets import Det3DDataset  # type: ignore
except Exception:  # pragma: no cover
  Det3DDataset = object  # type: ignore


class CustomLidarDataset(Det3DDataset):
  """Minimal dataset for the repo-local custom LiDAR info format.

  Expected ann_file payload:

  {
    "metainfo": {...},
    "data_list": [
      {
        "sample_idx": "...",
        "lidar_points": {"lidar_path": "...", "num_pts_feats": 4},
        "instances": [
          {
            "bbox_3d": [x, y, z, l, w, h, yaw],
            "bbox_label_3d": 0,
            "bbox_label_name": "car",
            "num_lidar_pts": 32
          }
        ]
      }
    ]
  }
  """

  METAINFO = {
      "classes": ("car", "pedestrian", "bicycle", "traffic_cone"),
      "box_type_3d": "LiDAR",
  }

  def parse_data_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
    info = dict(info)
    lidar_points = dict(info.get("lidar_points", {}))
    lidar_path = lidar_points.get("lidar_path", None)
    if lidar_path is None:
      raise KeyError("Each sample must define `lidar_points.lidar_path`.")

    lidar_path = Path(lidar_path)
    if not lidar_path.is_absolute():
      lidar_path = Path(self.data_root) / lidar_path
    lidar_points["lidar_path"] = str(lidar_path)
    info["lidar_points"] = lidar_points
    info["lidar_path"] = str(lidar_path)
    info.setdefault("num_pts_feats", int(lidar_points.get("num_pts_feats", 4)))

    if "instances" not in info:
      info["instances"] = []
    info.setdefault("ann_info", None)
    return super().parse_data_info(info)

  def parse_ann_info(self, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    instances = info.get("instances", [])
    if len(instances) == 0:
      ann_info = dict(
          gt_bboxes_3d=np.zeros((0, 7), dtype=np.float32),
          gt_labels_3d=np.zeros((0,), dtype=np.int64),
          gt_names=np.asarray([], dtype=object),
      )
      return ann_info

    ann_info = super().parse_ann_info(info)
    if ann_info is None:
      return None

    if "gt_labels_3d" in ann_info:
      ann_info["gt_labels_3d"] = ann_info["gt_labels_3d"].astype(np.int64, copy=False)
    if "gt_bboxes_3d" in ann_info:
      ann_info["gt_bboxes_3d"] = ann_info["gt_bboxes_3d"].astype(np.float32, copy=False)
    if "gt_names" not in ann_info:
      ann_info["gt_names"] = np.asarray(
          [instance.get("bbox_label_name", "") for instance in instances],
          dtype=object,
      )
    return ann_info


def register_to_mmdet3d() -> None:
  try:  # pragma: no cover
    from mmdet3d.registry import DATASETS  # type: ignore

    DATASETS.register_module(module=CustomLidarDataset, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
