from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover
  import torch
except Exception:  # pragma: no cover
  torch = None  # type: ignore


def _as_numpy(x: Any) -> np.ndarray:
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
    return x.detach().cpu().numpy()
  return np.asarray(x)


def _as_same_type(ref: Any, arr: np.ndarray) -> Any:
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(ref):
    return torch.as_tensor(arr, dtype=torch.long, device=ref.device)  # type: ignore[attr-defined]
  return arr.astype(np.int64, copy=False)


def _as_bool_mask(ref: Any, mask: np.ndarray) -> Any:
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(ref):
    return torch.as_tensor(mask, dtype=torch.bool, device=ref.device)  # type: ignore[attr-defined]
  return mask.astype(bool, copy=False)


def _filter_by_mask(value: Any, mask: Any) -> Any:
  if value is None:
    return None
  # Most MMDet3D structures support boolean indexing: ndarray, Tensor, BasePoints,
  # BaseInstance3DBoxes, etc.
  try:
    return value[mask]
  except Exception:
    return value


try:  # pragma: no cover
  from mmcv.transforms import BaseTransform  # type: ignore
except Exception:  # pragma: no cover
  BaseTransform = object  # type: ignore


class ApolloMapClasses3D(BaseTransform):
  """Map dataset classes into a small fixed label space (e.g. 4 tasks).

  This is meant to be used in an MMDet3D dataset pipeline after annotations are
  loaded (i.e. `gt_bboxes_3d`, `gt_labels_3d` and/or `gt_names` are present).

  Args:
    mapping: Dict from source class name -> target label id (int).
      Unmapped classes are dropped by default.
    src_classes: Optional list of source class names in the original label
      space (used when `gt_names` is not available).
    keep_unmapped: If True, keep objects not in mapping and keep their original
      label ids (requires `gt_labels_3d` to exist). Default False to drop them.
  """

  def __init__(
      self,
      mapping: Mapping[str, int],
      src_classes: Optional[Sequence[str]] = None,
      keep_unmapped: bool = False,
  ) -> None:
    self.mapping = dict(mapping)
    self.src_classes = list(src_classes) if src_classes is not None else None
    self.keep_unmapped = bool(keep_unmapped)

  def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
    labels = results.get("gt_labels_3d", None)
    names = results.get("gt_names", None)
    if labels is None and names is None:
      return results

    if names is not None:
      src_names = list(names)
    else:
      if self.src_classes is None:
        # Best-effort fallback: some loaders stash class names in results.
        maybe = results.get("classes", None) or results.get("class_names", None)
        if isinstance(maybe, (list, tuple)) and maybe:
          src_classes = list(maybe)
        else:
          raise KeyError(
              "ApolloMapClasses3D needs `gt_names` or `src_classes` to map labels."
          )
      else:
        src_classes = self.src_classes

      labels_np = _as_numpy(labels).reshape(-1)
      src_names = [src_classes[int(i)] for i in labels_np.tolist()]

    keep = []
    mapped = []
    labels_np = _as_numpy(labels).reshape(-1) if labels is not None else None
    for i, n in enumerate(src_names):
      if n in self.mapping:
        keep.append(True)
        mapped.append(int(self.mapping[n]))
      else:
        if self.keep_unmapped:
          if labels_np is None:
            raise KeyError(
                "keep_unmapped=True requires `gt_labels_3d` to keep original ids."
            )
          keep.append(True)
          mapped.append(int(labels_np[i]))
        else:
          keep.append(False)
          mapped.append(-1)

    keep_np = np.asarray(keep, dtype=bool)
    if not np.any(keep_np):
      # Keep empty but well-formed tensors/arrays.
      if labels is not None:
        empty = _as_numpy(labels).reshape(-1)[:0]
        results["gt_labels_3d"] = _as_same_type(labels, empty)
      if "gt_bboxes_3d" in results:
        results["gt_bboxes_3d"] = _filter_by_mask(results["gt_bboxes_3d"], keep_np)
      if "gt_names" in results:
        results["gt_names"] = []
      return results

    mapped_arr = np.asarray([m for (k, m) in zip(keep, mapped) if k], dtype=np.int64)
    if labels is None:
      results["gt_labels_3d"] = mapped_arr
    else:
      results["gt_labels_3d"] = _as_same_type(labels, mapped_arr)

    mask = _as_bool_mask(labels, keep_np) if labels is not None else keep_np
    if "gt_bboxes_3d" in results:
      results["gt_bboxes_3d"] = _filter_by_mask(results["gt_bboxes_3d"], mask)
    if "gt_bboxes_3d_mask" in results:
      results["gt_bboxes_3d_mask"] = _filter_by_mask(results["gt_bboxes_3d_mask"], mask)
    if "gt_names" in results:
      results["gt_names"] = [n for (k, n) in zip(keep, src_names) if k]

    return results


def register_to_mmdet3d() -> None:
  """Register to MMDet3D registry if available."""
  try:  # pragma: no cover
    from mmdet3d.registry import TRANSFORMS  # type: ignore

    TRANSFORMS.register_module(module=ApolloMapClasses3D, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
