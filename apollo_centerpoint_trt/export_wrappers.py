from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn


def _cat_task_outputs(task_outs: List[Dict[str, torch.Tensor]],
                      keys: Tuple[str, ...]) -> torch.Tensor:
  tensors: List[torch.Tensor] = []
  for out in task_outs:
    for k in keys:
      if k in out:
        tensors.append(out[k])
        break
    else:
      raise KeyError(f"Cannot find any of keys={keys} in task output: {out.keys()}")
  return torch.cat(tensors, dim=1)


def _get_task_tensor(task_out: Dict[str, torch.Tensor],
                     keys: Tuple[str, ...]) -> torch.Tensor:
  for k in keys:
    if k in task_out:
      return task_out[k]
  raise KeyError(f"Cannot find any of keys={keys} in task output: {task_out.keys()}")


def _wrap_feats_for_centerhead(feats: Any) -> List[torch.Tensor]:
  """CenterHead in MMDet3D 1.x expects a list of feature maps."""
  if isinstance(feats, (list, tuple)):
    return list(feats)
  if isinstance(feats, torch.Tensor):
    return [feats]
  raise TypeError(f"Unexpected feats type for head: {type(feats)}")


def _normalize_centerhead_outs(outs: Any) -> List[Dict[str, torch.Tensor]]:
  """Normalize MMDet3D CenterHead forward outputs across versions.

  Common patterns:
  - List[Dict[str, Tensor]]: per-task outputs
  - Tuple[List[Dict[str, Tensor]], ...]: aux outputs appended
  - Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]], ...]:
      outputs already split per-task by `multi_apply` (one list per task)
  """
  if isinstance(outs, list):
    task_outs = outs
  elif isinstance(outs, tuple) and len(outs) > 0:
    # Case B (must check first): multi_apply split per-task:
    #   ( [dict_level0], [dict_level0], [dict_level0], [dict_level0] )
    if all(isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict)
           for o in outs):
      task_outs = [o[0] for o in outs]  # type: ignore[list-item]
    # Case A: (task_outs, aux1, aux2, ...)
    elif isinstance(outs[0], list):
      task_outs = outs[0]
      # Some versions return task_outs per feature level: [[dict, dict, ...]]
      if len(task_outs) > 0 and isinstance(task_outs[0], list):
        task_outs = task_outs[0]
    else:
      raise TypeError(
          f"Unexpected head outputs tuple form: {type(outs[0])}")
  else:
    raise TypeError(f"Unexpected head outputs type: {type(outs)}")

  if len(task_outs) == 0:
    raise ValueError("Empty head outputs")
  if not isinstance(task_outs[0], dict):
    raise TypeError(f"Unexpected task output type: {type(task_outs[0])}")
  return task_outs  # type: ignore[return-value]


class BackboneHeadExportWrapper(nn.Module):
  """Export wrapper that produces Apollo C++ expected tensors:
    - bbox_preds
    - scores
    - dir_scores

  This wrapper assumes the model uses a CenterPoint-like layout with:
    - model.pts_backbone
    - model.pts_neck (optional)
    - model.pts_bbox_head (CenterHead-style, multi-task)

  Forward input:
    - canvas_feature: [B, C, H, W]
  """

  def __init__(self, model: nn.Module):
    super().__init__()
    # Some training wrappers store the real CenterPoint module under
    # `model.centerpoint`.
    core = model
    if not hasattr(core, "pts_backbone") and hasattr(core, "centerpoint"):
      core = getattr(core, "centerpoint")

    self.pts_backbone = getattr(core, "pts_backbone", None)
    self.pts_neck = getattr(core, "pts_neck", None)
    self.pts_bbox_head = getattr(core, "pts_bbox_head", None)
    if self.pts_backbone is None or self.pts_bbox_head is None:
      raise ValueError("Model must have pts_backbone and pts_bbox_head (or model.centerpoint.*)")

  def forward(
      self, canvas_feature: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = self.pts_backbone(canvas_feature)
    if self.pts_neck is not None:
      x = self.pts_neck(x)

    feats = _wrap_feats_for_centerhead(x)
    outs: Any = self.pts_bbox_head(feats)
    task_outs = _normalize_centerhead_outs(outs)

    # Each task dict typically includes:
    # - reg: [B, 2, H, W]
    # - height/hei: [B, 1, H, W]
    # - dim: [B, 3, H, W]
    # - rot: [B, 2, H, W]
    # - heatmap/hm: [B, num_cls_task, H, W]
    # Apollo's shipped ONNX packs bbox_preds channels *per task* and uses a
    # single Concat with 12 inputs:
    #   [t0.reg, t0.hei, t0.dim, t1.reg, t1.hei, t1.dim, ...]
    bbox_parts: List[torch.Tensor] = []
    score_parts: List[torch.Tensor] = []
    dir_parts: List[torch.Tensor] = []
    for t in task_outs:
      bbox_parts.append(_get_task_tensor(t, ("reg",)))
      bbox_parts.append(_get_task_tensor(t, ("height", "hei")))
      bbox_parts.append(_get_task_tensor(t, ("dim",)))
      score_parts.append(_get_task_tensor(t, ("heatmap", "hm", "scores")))
      dir_parts.append(_get_task_tensor(t, ("rot",)))

    bbox_preds = torch.cat(bbox_parts, dim=1)
    scores = torch.cat(score_parts, dim=1)
    dir_scores = torch.cat(dir_parts, dim=1)
    # Match Apollo shipped ONNX graph output ordering:
    #   scores, bbox_preds, dir_scores
    return scores, bbox_preds, dir_scores
