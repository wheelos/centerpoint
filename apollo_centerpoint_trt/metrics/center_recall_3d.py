from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .merged_ap_3d import _filter_range, _get_field, _to_numpy

try:  # pragma: no cover
  from mmengine.evaluator import BaseMetric  # type: ignore
except Exception:  # pragma: no cover
  BaseMetric = object  # type: ignore


@dataclass
class _RecallAgg:
  num_gts: int = 0
  num_matched: int = 0


class ApolloCenterRecallMetric3D(BaseMetric):
  """Center-distance recall metric for 4-task detection.

  Matching is done per-sample, per-class using BEV center distance. A prediction
  is a true positive if its center is within `distance_thr` meters of an
  unmatched GT center of the same class.
  """

  default_prefix: Optional[str] = "apollo_center"

  def __init__(
      self,
      class_names: Sequence[str] = ("car", "pedestrian", "bicycle", "traffic_cone"),
      distance_thr: float = 0.5,
      max_dets: int = 500,
      score_thr: float = 0.0,
      center_range: Optional[Sequence[float]] = None,
      collect_device: str = "cpu",
      prefix: Optional[str] = None,
  ) -> None:
    super().__init__(collect_device=collect_device, prefix=prefix)
    self.class_names = list(class_names)
    self.distance_thr = float(distance_thr)
    self.max_dets = int(max_dets)
    self.score_thr = float(score_thr)
    self.center_range = list(center_range) if center_range is not None else None
    self._agg = [_RecallAgg() for _ in range(len(self.class_names))]

  def _reset_agg(self) -> None:
    self._agg = [_RecallAgg() for _ in range(len(self.class_names))]

  def _extract_gt_labels(self, gt: Any) -> Any:
    labels = _get_field(gt, "labels_3d", None)
    if labels is None:
      labels = _get_field(gt, "labels", None)
    return labels

  def _extract_batch_samples(self, data_batch: Any) -> Sequence[Any]:
    if isinstance(data_batch, dict):
      samples = data_batch.get("data_samples", None)
      if isinstance(samples, Sequence):
        return samples
    return ()

  def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:  # type: ignore[override]
    batch_samples = self._extract_batch_samples(data_batch)
    if batch_samples and len(batch_samples) == len(data_samples):
      sample_pairs = zip(batch_samples, data_samples)
    else:
      sample_pairs = zip(data_samples, data_samples)

    for gt_sample, pred_sample in sample_pairs:
      gt = _get_field(gt_sample, "gt_instances_3d", None)
      pred = _get_field(pred_sample, "pred_instances_3d", None)
      if gt is None:
        gt = _get_field(pred_sample, "gt_instances_3d", None)
      self.results.append(1)
      if gt is None or pred is None:
        continue

      gt_boxes = _get_field(gt, "bboxes_3d", None)
      gt_labels = self._extract_gt_labels(gt)
      pred_boxes = _get_field(pred, "bboxes_3d", None)
      pred_labels = _get_field(pred, "labels_3d", None)
      pred_scores = _get_field(pred, "scores_3d", None)
      if pred_scores is None:
        pred_scores = _get_field(pred, "scores", None)

      gt_boxes, _, gt_labels = _filter_range(gt_boxes, None, gt_labels, self.center_range)
      pred_boxes, pred_scores, pred_labels = _filter_range(
          pred_boxes, pred_scores, pred_labels, self.center_range
      )

      gt_boxes_np = _to_numpy(gt_boxes) if gt_boxes is not None else np.zeros((0, 7), dtype=np.float32)
      pred_boxes_np = _to_numpy(pred_boxes) if pred_boxes is not None else np.zeros((0, 7), dtype=np.float32)
      if gt_boxes_np.ndim == 1:
        gt_boxes_np = gt_boxes_np.reshape(0, 7)
      if pred_boxes_np.ndim == 1:
        pred_boxes_np = pred_boxes_np.reshape(0, 7)
      gt_labels_np = _to_numpy(gt_labels).astype(np.int64, copy=False).reshape(-1)
      pred_labels_np = _to_numpy(pred_labels).astype(np.int64, copy=False).reshape(-1)
      pred_scores_np = _to_numpy(pred_scores).astype(np.float32, copy=False).reshape(-1)

      for cls_id in range(len(self.class_names)):
        agg = self._agg[cls_id]
        gt_idx = np.where(gt_labels_np == cls_id)[0]
        pred_idx = np.where(pred_labels_np == cls_id)[0]
        agg.num_gts += int(gt_idx.size)
        if gt_idx.size == 0 or pred_idx.size == 0:
          continue

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

        gt_xy = gt_boxes_np[gt_idx, :2]
        pred_xy = pred_boxes_np[pred_idx, :2]
        matched = np.zeros((gt_xy.shape[0],), dtype=bool)
        for i in range(pred_xy.shape[0]):
          dist = np.linalg.norm(gt_xy - pred_xy[i : i + 1], axis=1)
          if dist.size == 0:
            continue
          best = int(np.argmin(dist))
          if (not matched[best]) and float(dist[best]) <= self.distance_thr:
            matched[best] = True
        agg.num_matched += int(matched.sum())

  def compute_metrics(self, results: List[Any]) -> Dict[str, float]:  # type: ignore[override]
    metrics: Dict[str, float] = {}
    recalls: List[float] = []
    suffix = f"Recall@{self.distance_thr:.1f}m"
    for cls_id, name in enumerate(self.class_names):
      agg = self._agg[cls_id]
      if agg.num_gts <= 0:
        metrics[f"{suffix}/{name}"] = float("nan")
        continue
      recall = float(agg.num_matched) / float(agg.num_gts)
      metrics[f"{suffix}/{name}"] = recall
      recalls.append(recall)
    metrics[f"m{suffix}"] = float(np.mean(recalls)) if recalls else float("nan")
    self._reset_agg()
    return metrics


def register_to_mmdet3d() -> None:
  try:  # pragma: no cover
    from mmdet3d.registry import METRICS  # type: ignore

    METRICS.register_module(module=ApolloCenterRecallMetric3D, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
