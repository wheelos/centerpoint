from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
  import torch
except Exception:  # pragma: no cover
  torch = None  # type: ignore


def _to_numpy(x: Any) -> np.ndarray:
  if x is None:
    return np.zeros((0,), dtype=np.float32)
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
    return x.detach().cpu().numpy()
  if hasattr(x, "tensor"):
    t = getattr(x, "tensor")
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(t):
      return t.detach().cpu().numpy()
  return np.asarray(x)


def _filter_range(boxes: Any, scores: Any, labels: Any,
                  center_range: Sequence[float]) -> Tuple[Any, Any, Any]:
  if boxes is None:
    return boxes, scores, labels
  if center_range is None:
    return boxes, scores, labels
  if len(center_range) != 6:
    raise ValueError("center_range must be [minx,miny,minz,maxx,maxy,maxz]")
  b = boxes.tensor if hasattr(boxes, "tensor") else boxes
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(b):
    cx, cy, cz = b[:, 0], b[:, 1], b[:, 2]
    mask = (
        (cx >= center_range[0]) & (cy >= center_range[1]) & (cz >= center_range[2]) &
        (cx <= center_range[3]) & (cy <= center_range[4]) & (cz <= center_range[5])
    )
  else:
    arr = _to_numpy(b)
    cx, cy, cz = arr[:, 0], arr[:, 1], arr[:, 2]
    mask = (
        (cx >= center_range[0]) & (cy >= center_range[1]) & (cz >= center_range[2]) &
        (cx <= center_range[3]) & (cy <= center_range[4]) & (cz <= center_range[5])
    )
  try:
    boxes = boxes[mask]
  except Exception:
    pass
  if scores is not None:
    try:
      scores = scores[mask]
    except Exception:
      pass
  if labels is not None:
    try:
      labels = labels[mask]
    except Exception:
      pass
  return boxes, scores, labels


def _bev_boxes_xywhr(boxes_3d: Any) -> Any:
  """Convert BaseInstance3DBoxes to BEV rotated boxes (x,y,w,h,angle) tensor."""
  if boxes_3d is None:
    if torch is not None:
      return torch.zeros((0, 5), dtype=torch.float32)
    return np.zeros((0, 5), dtype=np.float32)

  # MMDet3D boxes usually provide `.bev` as (x, y, dx, dy, yaw).
  if hasattr(boxes_3d, "bev"):
    bev = boxes_3d.bev
    return bev

  b = boxes_3d.tensor if hasattr(boxes_3d, "tensor") else boxes_3d
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(b):
    if b.numel() == 0:
      return b.new_zeros((0, 5))
    return b[:, [0, 1, 3, 4, 6]].contiguous()
  arr = _to_numpy(b)
  if arr.size == 0:
    return arr.reshape(0, 5)
  return arr[:, [0, 1, 3, 4, 6]]


def _box_iou_rotated_bev(a_bev: Any, b_bev: Any) -> np.ndarray:
  """BEV IoU using mmcv's rotated box IoU if available (fallback axis-aligned)."""
  if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(a_bev):
    a_t = a_bev
    b_t = b_bev
  else:
    if torch is None:
      raise RuntimeError("ApolloMergedClassMetric3D requires torch in the runtime.")
    a_t = torch.as_tensor(a_bev, dtype=torch.float32)
    b_t = torch.as_tensor(b_bev, dtype=torch.float32)

  if a_t.numel() == 0 or b_t.numel() == 0:
    return np.zeros((int(a_t.shape[0]), int(b_t.shape[0])), dtype=np.float32)

  try:
    from mmcv.ops import box_iou_rotated  # type: ignore

    iou = box_iou_rotated(a_t, b_t, aligned=False)
    return iou.detach().cpu().numpy()
  except Exception:
    # Fallback: axis-aligned IoU using (x,y,w,h) ignoring angle.
    ax, ay, aw, ah = a_t[:, 0], a_t[:, 1], a_t[:, 2], a_t[:, 3]
    bx, by, bw, bh = b_t[:, 0], b_t[:, 1], b_t[:, 2], b_t[:, 3]
    a_x1 = ax - aw / 2
    a_y1 = ay - ah / 2
    a_x2 = ax + aw / 2
    a_y2 = ay + ah / 2
    b_x1 = bx - bw / 2
    b_y1 = by - bh / 2
    b_x2 = bx + bw / 2
    b_y2 = by + bh / 2

    # broadcast
    inter_x1 = torch.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = torch.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = torch.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = torch.minimum(a_y2[:, None], b_y2[None, :])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter = inter_w * inter_h
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / torch.clamp(union, min=1e-6)
    return iou.detach().cpu().numpy()


def _compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
  """VOC-style AP with precision envelope (integral)."""
  if rec.size == 0:
    return 0.0
  mrec = np.concatenate(([0.0], rec, [1.0]))
  mpre = np.concatenate(([0.0], prec, [0.0]))
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = max(mpre[i - 1], mpre[i])
  idx = np.where(mrec[1:] != mrec[:-1])[0]
  ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
  return ap


@dataclass
class _ClassAgg:
  num_gts: int = 0
  scores: List[float] = field(default_factory=list)
  tp: List[int] = field(default_factory=list)
  fp: List[int] = field(default_factory=list)


try:  # pragma: no cover
  from mmengine.evaluator import BaseMetric  # type: ignore
except Exception:  # pragma: no cover
  BaseMetric = object  # type: ignore


class ApolloMergedClassMetric3D(BaseMetric):
  """Simple BEV mAP metric for a merged-class (4-task) CenterPointTRT setup.

  This metric is intended for *project-internal sanity checking* when you
  collapse dataset classes into a smaller label space (e.g. nuScenes 10 -> 4).
  It does NOT implement official nuScenes evaluation.

  Matching:
    - per-sample, per-class greedy matching by descending score
    - BEV IoU threshold per class

  Args:
    class_names: Names for target label space (index = label id).
    iou_thr: IoU threshold per class (float list aligned with class_names).
    max_dets: Max detections per sample per class to consider.
    score_thr: Score threshold for filtering predictions.
    center_range: Optional range filter [minx,miny,minz,maxx,maxy,maxz] applied
      to both GT and predictions before evaluation.
  """

  default_prefix: Optional[str] = "apollo"

  def __init__(
      self,
      class_names: Sequence[str] = ("car", "pedestrian", "bicycle", "traffic_cone"),
      iou_thr: Sequence[float] = (0.5, 0.25, 0.25, 0.25),
      max_dets: int = 500,
      score_thr: float = 0.0,
      center_range: Optional[Sequence[float]] = None,
      collect_device: str = "cpu",
      prefix: Optional[str] = None,
  ) -> None:
    super().__init__(collect_device=collect_device, prefix=prefix)
    self.class_names = list(class_names)
    self.iou_thr = list(iou_thr)
    if len(self.iou_thr) != len(self.class_names):
      raise ValueError("iou_thr length must match class_names length")
    self.max_dets = int(max_dets)
    self.score_thr = float(score_thr)
    self.center_range = list(center_range) if center_range is not None else None

    self._agg = [_ClassAgg() for _ in range(len(self.class_names))]

  def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:  # type: ignore[override]
    for sample in data_samples:
      gt = getattr(sample, "gt_instances_3d", None)
      pred = getattr(sample, "pred_instances_3d", None)
      if gt is None or pred is None:
        continue

      gt_boxes = getattr(gt, "bboxes_3d", None)
      gt_labels = getattr(gt, "labels_3d", None)
      pred_boxes = getattr(pred, "bboxes_3d", None)
      pred_labels = getattr(pred, "labels_3d", None)
      pred_scores = getattr(pred, "scores_3d", None)
      if pred_scores is None:
        pred_scores = getattr(pred, "scores", None)

      gt_boxes, _, gt_labels = _filter_range(gt_boxes, None, gt_labels, self.center_range)
      pred_boxes, pred_scores, pred_labels = _filter_range(
          pred_boxes, pred_scores, pred_labels, self.center_range
      )

      gt_labels_np = _to_numpy(gt_labels).astype(np.int64, copy=False).reshape(-1)
      pred_labels_np = _to_numpy(pred_labels).astype(np.int64, copy=False).reshape(-1)
      pred_scores_np = _to_numpy(pred_scores).astype(np.float32, copy=False).reshape(-1)

      for cls_id in range(len(self.class_names)):
        thr = float(self.iou_thr[cls_id])
        agg = self._agg[cls_id]

        gt_idx = np.where(gt_labels_np == cls_id)[0]
        pred_idx = np.where(pred_labels_np == cls_id)[0]

        agg.num_gts += int(gt_idx.size)
        if pred_idx.size == 0:
          continue

        # score filter + per-class top-k
        scores = pred_scores_np[pred_idx]
        keep = scores >= self.score_thr
        if not np.any(keep):
          continue
        pred_idx = pred_idx[keep]
        scores = scores[keep]

        order = np.argsort(-scores)
        if self.max_dets > 0:
          order = order[: self.max_dets]
        pred_idx = pred_idx[order]
        scores = scores[order]

        # No GT: all are FP
        if gt_idx.size == 0:
          agg.scores.extend(scores.tolist())
          agg.tp.extend([0] * int(scores.size))
          agg.fp.extend([1] * int(scores.size))
          continue

        pred_boxes_c = pred_boxes[pred_idx]
        gt_boxes_c = gt_boxes[gt_idx]

        iou = _box_iou_rotated_bev(_bev_boxes_xywhr(pred_boxes_c), _bev_boxes_xywhr(gt_boxes_c))
        matched = np.zeros((gt_idx.size,), dtype=bool)

        for i in range(int(scores.size)):
          j = int(np.argmax(iou[i])) if iou.shape[1] > 0 else -1
          best = float(iou[i, j]) if j >= 0 else 0.0
          if best >= thr and (j >= 0) and (not matched[j]):
            matched[j] = True
            agg.scores.append(float(scores[i]))
            agg.tp.append(1)
            agg.fp.append(0)
          else:
            agg.scores.append(float(scores[i]))
            agg.tp.append(0)
            agg.fp.append(1)

  def compute_metrics(self, results: List[Any]) -> Dict[str, float]:  # type: ignore[override]
    # `results` unused because we aggregate online in process().
    metrics: Dict[str, float] = {}
    aps: List[float] = []
    for cls_id, name in enumerate(self.class_names):
      agg = self._agg[cls_id]
      if agg.num_gts <= 0:
        metrics[f"AP/{name}"] = float("nan")
        continue

      scores = np.asarray(agg.scores, dtype=np.float32)
      tp = np.asarray(agg.tp, dtype=np.int64)
      fp = np.asarray(agg.fp, dtype=np.int64)
      if scores.size == 0:
        metrics[f"AP/{name}"] = 0.0
        aps.append(0.0)
        continue

      order = np.argsort(-scores)
      tp = tp[order]
      fp = fp[order]
      tp_cum = np.cumsum(tp)
      fp_cum = np.cumsum(fp)
      rec = tp_cum / float(agg.num_gts)
      prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
      ap = _compute_ap(rec.astype(np.float32), prec.astype(np.float32))
      metrics[f"AP/{name}"] = float(ap)
      aps.append(float(ap))

    if aps:
      metrics["mAP"] = float(np.nanmean(np.asarray(aps, dtype=np.float32)))
    return metrics


def register_to_mmdet3d() -> None:
  """Register to MMDet3D registry if available."""
  try:  # pragma: no cover
    from mmdet3d.registry import METRICS  # type: ignore

    METRICS.register_module(module=ApolloMergedClassMetric3D, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass

