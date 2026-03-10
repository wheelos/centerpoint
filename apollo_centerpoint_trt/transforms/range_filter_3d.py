from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
  import torch
except Exception:  # pragma: no cover
  torch = None  # type: ignore

try:  # pragma: no cover
  from mmcv.transforms import BaseTransform  # type: ignore
except Exception:  # pragma: no cover
  BaseTransform = object  # type: ignore


def _as_numpy(x: Any) -> np.ndarray:
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
    return x.detach().cpu().numpy()
  if hasattr(x, "tensor"):
    t = getattr(x, "tensor")
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(t):
      return t.detach().cpu().numpy()
  return np.asarray(x)


def _as_bool_mask(ref: Any, mask: np.ndarray) -> Any:
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(ref):
    return torch.as_tensor(mask, dtype=torch.bool, device=ref.device)  # type: ignore[attr-defined]
  return mask.astype(bool, copy=False)


def _filter_by_mask(value: Any, mask: Any) -> Any:
  if value is None:
    return None
  try:
    return value[mask]
  except Exception:
    return value


class ApolloRangeFilter3D(BaseTransform):
  """Filter points and GT with Apollo preprocessing range semantics.

  Compared with stock `PointsRangeFilter/ObjectRangeFilter`, this transform can
  apply the same optional +45° xy rotation used by Apollo preprocessing before
  checking x/y bounds.
  """

  def __init__(
      self,
      point_cloud_range: Sequence[float],
      enable_rotate_45degree: bool = True,
  ) -> None:
    if len(point_cloud_range) != 6:
      raise ValueError("point_cloud_range must be [minx,miny,minz,maxx,maxy,maxz]")
    self.point_cloud_range = [float(x) for x in point_cloud_range]
    self.enable_rotate_45degree = bool(enable_rotate_45degree)

  def _rotate45_np(self, xy: np.ndarray) -> np.ndarray:
    c = 1.0 / math.sqrt(2.0)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack((c * x - c * y, c * x + c * y), axis=-1)

  def _in_range_mask(self, centers_xyz: np.ndarray) -> np.ndarray:
    xyz = centers_xyz[:, :3]
    xy = xyz[:, :2]
    z = xyz[:, 2]
    if self.enable_rotate_45degree:
      xy = self._rotate45_np(xy)
    min_x, min_y, min_z, max_x, max_y, max_z = self.point_cloud_range
    mask = (
        (xy[:, 0] >= min_x) & (xy[:, 0] <= max_x) &
        (xy[:, 1] >= min_y) & (xy[:, 1] <= max_y) &
        (z >= min_z) & (z <= max_z)
    )
    return mask.astype(bool, copy=False)

  def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
    points = results.get("points", None)
    if points is not None:
      points_xyz = _as_numpy(points)
      point_mask_np = self._in_range_mask(points_xyz)
      point_mask = _as_bool_mask(getattr(points, "tensor", points), point_mask_np)
      results["points"] = _filter_by_mask(points, point_mask)

    gt_bboxes_3d = results.get("gt_bboxes_3d", None)
    if gt_bboxes_3d is not None:
      box_tensor = _as_numpy(gt_bboxes_3d)
      box_mask_np = self._in_range_mask(box_tensor)
      box_mask = _as_bool_mask(getattr(gt_bboxes_3d, "tensor", gt_bboxes_3d), box_mask_np)
      results["gt_bboxes_3d"] = _filter_by_mask(gt_bboxes_3d, box_mask)
      if "gt_labels_3d" in results:
        results["gt_labels_3d"] = _filter_by_mask(results["gt_labels_3d"], box_mask)
      if "gt_names" in results:
        results["gt_names"] = [n for keep, n in zip(box_mask_np.tolist(), results["gt_names"]) if keep]
      if "gt_bboxes_3d_mask" in results:
        results["gt_bboxes_3d_mask"] = _filter_by_mask(results["gt_bboxes_3d_mask"], box_mask)

    return results


def register_to_mmdet3d() -> None:
  try:  # pragma: no cover
    from mmdet3d.registry import TRANSFORMS  # type: ignore

    TRANSFORMS.register_module(module=ApolloRangeFilter3D, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
