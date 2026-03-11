#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
from pathlib import Path
import runpy
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _import_custom_imports(cfg_globals: dict) -> None:
  custom_imports = cfg_globals.get("custom_imports")
  if not isinstance(custom_imports, dict):
    return
  imports = custom_imports.get("imports")
  if not isinstance(imports, (list, tuple)):
    return
  for mod in imports:
    if isinstance(mod, str) and mod:
      __import__(mod)


def _register_all_modules() -> None:
  try:
    from mmdet3d.utils import register_all_modules  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Cannot import mmdet3d.utils.register_all_modules. "
        "Please run this script inside the MMDetection3D environment used for training."
    ) from exc

  kwargs = {}
  try:
    sig = inspect.signature(register_all_modules)
    if "init_default_scope" in sig.parameters:
      kwargs["init_default_scope"] = True
  except Exception:
    pass
  register_all_modules(**kwargs)  # type: ignore[arg-type]


def _load_cfg_globals(cfg_path: str) -> dict:
  try:
    cfg_globals = runpy.run_path(cfg_path, run_name="__onnx_eval_cfg__")
  except Exception as exc:
    raise RuntimeError(f"Failed to execute config: {cfg_path}") from exc
  _import_custom_imports(cfg_globals)
  _register_all_modules()
  return cfg_globals


def _build_model_from_cfg_globals(cfg_globals: dict) -> torch.nn.Module:
  model_cfg = cfg_globals.get("model")
  if model_cfg is None:
    raise KeyError("Config file does not define `model`.")
  if isinstance(model_cfg, torch.nn.Module):
    model = model_cfg
  else:
    if not isinstance(model_cfg, dict):
      raise TypeError(f"`model` must be dict or nn.Module, got {type(model_cfg)}")
    from mmdet3d.registry import MODELS  # type: ignore

    model = MODELS.build(model_cfg)
  model.eval()
  return model


def _build_dataset(dataset_cfg: dict):
  from mmdet3d.registry import DATASETS  # type: ignore

  return DATASETS.build(deepcopy(dataset_cfg))


def _build_dataloader(dataloader_cfg: dict, num_workers_override: int | None = None) -> DataLoader:
  cfg = deepcopy(dataloader_cfg)
  dataset = _build_dataset(cfg.pop("dataset"))
  batch_size = int(cfg.pop("batch_size", 1))
  num_workers = int(cfg.pop("num_workers", 0))
  if num_workers_override is not None:
    num_workers = int(num_workers_override)
  persistent_workers = bool(cfg.pop("persistent_workers", False)) and num_workers > 0
  cfg.pop("sampler", None)
  collate_cfg = cfg.pop("collate_fn", None)
  pin_memory = bool(cfg.pop("pin_memory", False))
  if cfg:
    unknown = ", ".join(sorted(cfg.keys()))
    raise ValueError(f"Unsupported dataloader fields in ONNX eval script: {unknown}")

  try:
    from mmengine.dataset import pseudo_collate  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError("Cannot import mmengine.dataset.pseudo_collate") from exc

  collate_fn = pseudo_collate
  if isinstance(collate_cfg, dict):
    collate_type = collate_cfg.get("type", "pseudo_collate")
    if collate_type != "pseudo_collate":
      raise ValueError(f"Unsupported collate_fn type for ONNX eval: {collate_type}")

  return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      persistent_workers=persistent_workers,
      pin_memory=pin_memory,
      collate_fn=collate_fn,
  )


def _build_evaluator(evaluator_cfg: Any, dataset_meta: Any):
  try:
    from mmengine.evaluator import Evaluator  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError("Cannot import mmengine.evaluator.Evaluator") from exc

  evaluator = Evaluator(deepcopy(evaluator_cfg))
  if dataset_meta is not None and hasattr(evaluator, "dataset_meta"):
    evaluator.dataset_meta = dataset_meta
  return evaluator


def _extract_batch_points(batch: Any) -> List[torch.Tensor]:
  if not isinstance(batch, dict):
    raise TypeError(f"Expected dict batch from dataloader, got {type(batch)}")
  inputs = batch.get("inputs", None)
  if not isinstance(inputs, dict):
    raise KeyError("Batch missing `inputs` dict.")
  points = inputs.get("points", None)
  if points is None:
    raise KeyError("Batch missing `inputs['points']`.")
  if torch.is_tensor(points):
    return [points]
  if isinstance(points, (list, tuple)):
    out: List[torch.Tensor] = []
    for item in points:
      if hasattr(item, "tensor"):
        item = item.tensor
      if not torch.is_tensor(item):
        raise TypeError(f"Unsupported points item type: {type(item)}")
      out.append(item)
    return out
  raise TypeError(f"Unsupported points container type: {type(points)}")


def _extract_data_samples(batch: Any) -> List[Any]:
  if not isinstance(batch, dict):
    return []
  data_samples = batch.get("data_samples", None)
  if data_samples is None:
    return []
  if isinstance(data_samples, list):
    return data_samples
  if isinstance(data_samples, tuple):
    return list(data_samples)
  return [data_samples]


def _select_dataloader_cfg(cfg_globals: dict, split: str) -> Tuple[dict, Any]:
  if split == "train":
    dataloader_cfg = cfg_globals.get("train_eval_dataloader")
    evaluator_cfg = cfg_globals.get("train_evaluator")
  elif split == "val":
    dataloader_cfg = cfg_globals.get("val_dataloader")
    evaluator_cfg = cfg_globals.get("val_evaluator")
  elif split == "test":
    dataloader_cfg = cfg_globals.get("test_dataloader")
    evaluator_cfg = cfg_globals.get("test_evaluator")
  else:
    raise ValueError(f"Unsupported split: {split}")
  if dataloader_cfg is None:
    raise KeyError(f"Config does not define `{split}_dataloader`.")
  if evaluator_cfg is None:
    raise KeyError(f"Config does not define `{split}_evaluator`.")
  return dataloader_cfg, evaluator_cfg


def _resolve_output_map(session: Any, outputs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
  names = [out.name for out in session.get_outputs()]
  return {name: value for name, value in zip(names, outputs)}


def _parse_providers(raw: str) -> List[str]:
  providers = [part.strip() for part in raw.split(",") if part.strip()]
  return providers or ["CPUExecutionProvider"]


def _create_ort_session(path: Path, providers: Sequence[str]):
  try:
    import onnxruntime as ort  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "onnxruntime is required for ONNX evaluation. Install `onnxruntime` "
        "or `onnxruntime-gpu` in the current environment."
    ) from exc

  sess_options = ort.SessionOptions()
  sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  available = set(ort.get_available_providers())
  selected = [provider for provider in providers if provider in available]
  if not selected:
    selected = ["CPUExecutionProvider"]
  try:
    return ort.InferenceSession(str(path), sess_options=sess_options, providers=selected)
  except Exception as exc:
    if "CPUExecutionProvider" in selected or "CPUExecutionProvider" not in available:
      raise
    print(
        f"[test_onnx] Failed to create session with providers={selected} for {path}: {exc}\n"
        "[test_onnx] Falling back to CPUExecutionProvider."
    )
    return ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def _ensure_cpu_float_tensor(x: torch.Tensor) -> torch.Tensor:
  return x.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _empty_canvas_from_model(model: Any, pts_xyzi: torch.Tensor) -> torch.Tensor:
  gx = int(model.bev_gen.grid_x_size)
  gy = int(model.bev_gen.grid_y_size)
  channels = int(model.bev_gen.cfg.pillar_feature_dim + model.bev_gen.cfg.cnnseg_feature_dim)
  return pts_xyzi.new_zeros((1, channels, gy, gx))


def _build_canvas_feature(model: Any, pfe_session: Any, points: torch.Tensor) -> torch.Tensor:
  pts_xyzi = model._points_to_xyzi(_ensure_cpu_float_tensor(points))
  voxels, _, _ = model.bev_gen.build_voxel_features(pts_xyzi)
  if voxels.size(0) == 0:
    return _empty_canvas_from_model(model, pts_xyzi)

  voxels_np = voxels.unsqueeze(1).unsqueeze(-1).cpu().numpy().astype(np.float32, copy=False)
  outputs = pfe_session.run(None, {"voxels": voxels_np})
  output_map = _resolve_output_map(pfe_session, outputs)
  pillar_feature_np = output_map.get("pillar_feature", outputs[0])
  pillar_feature = torch.from_numpy(pillar_feature_np).to(dtype=pts_xyzi.dtype, device="cpu")
  return model.bev_gen(pts_xyzi, pillar_feature)


def _head_requires_vel(head: Any) -> bool:
  common_heads = getattr(head, "common_heads", None)
  if isinstance(common_heads, dict) and "vel" in common_heads:
    return True
  separate_head = getattr(head, "separate_head", None)
  heads = getattr(separate_head, "heads", None)
  return isinstance(heads, dict) and "vel" in heads


def _build_head_outputs(
    head: Any,
    scores_np: np.ndarray,
    bbox_preds_np: np.ndarray,
    dir_scores_np: np.ndarray,
) -> List[Dict[str, torch.Tensor]]:
  scores = torch.from_numpy(scores_np).to(dtype=torch.float32, device="cpu")
  bbox_preds = torch.from_numpy(bbox_preds_np).to(dtype=torch.float32, device="cpu")
  dir_scores = torch.from_numpy(dir_scores_np).to(dtype=torch.float32, device="cpu")

  if scores.ndim != 4 or bbox_preds.ndim != 4 or dir_scores.ndim != 4:
    raise ValueError(
        "Expected backbone ONNX outputs to be 4D tensors: "
        f"scores={scores.shape}, bbox_preds={bbox_preds.shape}, dir_scores={dir_scores.shape}"
    )

  num_tasks = int(scores.shape[1])
  expected_bbox_channels = num_tasks * 6
  expected_rot_channels = num_tasks * 2
  if int(bbox_preds.shape[1]) != expected_bbox_channels:
    raise ValueError(
        f"bbox_preds channels mismatch: got {bbox_preds.shape[1]}, expected {expected_bbox_channels}"
    )
  if int(dir_scores.shape[1]) != expected_rot_channels:
    raise ValueError(
        f"dir_scores channels mismatch: got {dir_scores.shape[1]}, expected {expected_rot_channels}"
    )

  need_vel = _head_requires_vel(head)
  task_outs: List[Dict[str, torch.Tensor]] = []
  bbox_offset = 0
  rot_offset = 0
  for task_idx in range(num_tasks):
    task = dict(
        heatmap=scores[:, task_idx:task_idx + 1, :, :],
        reg=bbox_preds[:, bbox_offset:bbox_offset + 2, :, :],
        height=bbox_preds[:, bbox_offset + 2:bbox_offset + 3, :, :],
        dim=bbox_preds[:, bbox_offset + 3:bbox_offset + 6, :, :],
        rot=dir_scores[:, rot_offset:rot_offset + 2, :, :],
    )
    if need_vel:
      task["vel"] = torch.zeros(
          (scores.shape[0], 2, scores.shape[2], scores.shape[3]),
          dtype=scores.dtype,
          device=scores.device,
    )
    task_outs.append(task)
    bbox_offset += 6
    rot_offset += 2

  return task_outs


def _xywhr_to_corners(box: np.ndarray) -> np.ndarray:
  x, y, w, h, yaw = [float(v) for v in box.tolist()]
  cos_yaw = float(np.cos(yaw))
  sin_yaw = float(np.sin(yaw))
  half_w = w * 0.5
  half_h = h * 0.5
  corners = np.asarray(
      [
          [-half_w, -half_h],
          [half_w, -half_h],
          [half_w, half_h],
          [-half_w, half_h],
      ],
      dtype=np.float32,
  )
  rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
  return corners @ rot.T + np.asarray([x, y], dtype=np.float32)


def _polygon_area(points: np.ndarray) -> float:
  if points.shape[0] < 3:
    return 0.0
  x = points[:, 0]
  y = points[:, 1]
  return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside_edge(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
  return float((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
               (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= -1e-6


def _segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
  s = p2 - p1
  r = q2 - q1
  denom = float(s[0] * r[1] - s[1] * r[0])
  if abs(denom) < 1e-6:
    return p2.astype(np.float32)
  t = float(((q1[0] - p1[0]) * r[1] - (q1[1] - p1[1]) * r[0]) / denom)
  return (p1 + t * s).astype(np.float32)


def _polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
  output = subject.astype(np.float32)
  for i in range(clipper.shape[0]):
    clip_start = clipper[i]
    clip_end = clipper[(i + 1) % clipper.shape[0]]
    if output.shape[0] == 0:
      break
    input_list = output
    output_pts: List[np.ndarray] = []
    prev = input_list[-1]
    for cur in input_list:
      cur_inside = _inside_edge(cur, clip_start, clip_end)
      prev_inside = _inside_edge(prev, clip_start, clip_end)
      if cur_inside:
        if not prev_inside:
          output_pts.append(_segment_intersection(prev, cur, clip_start, clip_end))
        output_pts.append(cur.astype(np.float32))
      elif prev_inside:
        output_pts.append(_segment_intersection(prev, cur, clip_start, clip_end))
      prev = cur
    if output_pts:
      output = np.stack(output_pts, axis=0).astype(np.float32)
    else:
      output = np.zeros((0, 2), dtype=np.float32)
  return output


def _rotated_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
  poly1 = _xywhr_to_corners(box1)
  poly2 = _xywhr_to_corners(box2)
  inter_poly = _polygon_clip(poly1, poly2)
  inter_area = _polygon_area(inter_poly)
  area1 = float(box1[2] * box1[3])
  area2 = float(box2[2] * box2[3])
  union = max(area1 + area2 - inter_area, 1e-6)
  return float(inter_area / union)


def _greedy_nms_rotated(
    boxes_xywhr: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
    post_max_size: int,
) -> np.ndarray:
  if boxes_xywhr.shape[0] == 0:
    return np.zeros((0,), dtype=np.int64)
  order = np.argsort(-scores)
  keep: List[int] = []
  suppressed = np.zeros((boxes_xywhr.shape[0],), dtype=bool)
  for pos, idx in enumerate(order):
    if suppressed[pos]:
      continue
    keep.append(int(idx))
    if 0 < post_max_size <= len(keep):
      break
    for next_pos in range(pos + 1, order.shape[0]):
      if suppressed[next_pos]:
        continue
      jdx = int(order[next_pos])
      if _rotated_iou_single(boxes_xywhr[idx], boxes_xywhr[jdx]) > iou_thr:
        suppressed[next_pos] = True
  return np.asarray(keep, dtype=np.int64)


def _decode_single_sample(model: Any, task_outs: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
  cfg = getattr(model.pts_bbox_head, "test_cfg", None)
  if cfg is None:
    raise RuntimeError("pts_bbox_head.test_cfg is required for ONNX decoding.")

  score_threshold = float(cfg.get("score_threshold", 0.1))
  pre_max_size = int(cfg.get("pre_max_size", 1000))
  post_max_size = int(cfg.get("post_max_size", 83))
  nms_thr = float(cfg.get("nms_thr", 0.2))
  out_size_factor = float(cfg.get("out_size_factor", 4))
  voxel_size = cfg.get("voxel_size", [0.2, 0.2])
  pc_range = cfg.get("pc_range", None)
  point_cloud_range = cfg.get("point_cloud_range", None)
  if pc_range is None and point_cloud_range is not None:
    pc_range = point_cloud_range[:2]
  if pc_range is None:
    raise RuntimeError("test_cfg must provide `pc_range` or `point_cloud_range`.")
  min_x = float(pc_range[0])
  min_y = float(pc_range[1])

  all_boxes: List[torch.Tensor] = []
  all_scores: List[torch.Tensor] = []
  all_labels: List[torch.Tensor] = []

  for task_id, task in enumerate(task_outs):
    heatmap = torch.sigmoid(task["heatmap"][0]).cpu()  # [C,H,W]
    reg = task["reg"][0].cpu()
    height = task["height"][0].cpu()
    dim = task["dim"][0].cpu()
    rot = task["rot"][0].cpu()
    num_classes, feat_h, feat_w = heatmap.shape

    scores_flat = heatmap.reshape(num_classes, -1)
    cls_scores, cls_ids = torch.max(scores_flat, dim=0)
    keep_mask = cls_scores > score_threshold
    if int(keep_mask.sum()) == 0:
      continue

    candidate_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    if candidate_idx.numel() > pre_max_size > 0:
      topk_scores, topk_order = torch.topk(cls_scores[candidate_idx], k=pre_max_size)
      candidate_idx = candidate_idx[topk_order]
      cls_scores = topk_scores
      cls_ids = cls_ids[candidate_idx]
    else:
      cls_scores = cls_scores[candidate_idx]
      cls_ids = cls_ids[candidate_idx]

    xs = candidate_idx % feat_w
    ys = torch.div(candidate_idx, feat_w, rounding_mode="floor")

    x = (reg[0, ys, xs] + xs.to(dtype=reg.dtype)) * out_size_factor * float(voxel_size[0]) + min_x
    y = (reg[1, ys, xs] + ys.to(dtype=reg.dtype)) * out_size_factor * float(voxel_size[1]) + min_y
    z = height[0, ys, xs]
    l = torch.exp(dim[0, ys, xs])
    w = torch.exp(dim[1, ys, xs])
    h = torch.exp(dim[2, ys, xs])
    yaw = torch.atan2(rot[0, ys, xs], rot[1, ys, xs])

    labels = cls_ids.to(dtype=torch.long) + task_id
    boxes = torch.stack((x, y, z, l, w, h, yaw), dim=1)

    if int(boxes.shape[0]) > 0:
      boxes_xywhr = torch.stack((boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]), dim=1)
      keep = _greedy_nms_rotated(
          boxes_xywhr.numpy().astype(np.float32, copy=False),
          cls_scores.numpy().astype(np.float32, copy=False),
          iou_thr=nms_thr,
          post_max_size=post_max_size,
      )
      if keep.size == 0:
        continue
      keep_t = torch.from_numpy(keep).to(dtype=torch.long)
      boxes = boxes[keep_t]
      cls_scores = cls_scores[keep_t]
      labels = labels[keep_t]

    all_boxes.append(boxes)
    all_scores.append(cls_scores)
    all_labels.append(labels)

  if all_boxes:
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    order = torch.argsort(scores, descending=True)
    if post_max_size > 0:
      order = order[:post_max_size]
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]
  else:
    boxes = torch.zeros((0, 7), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    labels = torch.zeros((0,), dtype=torch.long)

  if boxes.shape[0] > 0 and bool(model.bev_gen.cfg.enable_rotate_45degree):
    c = float(1.0 / np.sqrt(2.0))
    orig_x = c * boxes[:, 0] + c * boxes[:, 1]
    orig_y = -c * boxes[:, 0] + c * boxes[:, 1]
    boxes[:, 0] = orig_x
    boxes[:, 1] = orig_y
    boxes[:, 6] = boxes[:, 6] - float(np.pi / 4.0)

  if boxes.shape[0] > 0:
    boxes[:, 6] = ((boxes[:, 6] + float(np.pi)) % float(2 * np.pi)) - float(np.pi)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 5] * 0.5

  return {
      "pred_instances_3d": {
          "bboxes_3d": boxes,
          "scores_3d": scores,
          "labels_3d": labels,
      }
  }


def _run_model_on_batch(model: Any, pfe_session: Any, backbone_session: Any, batch: dict) -> List[Dict[str, Any]]:
  points_list = _extract_batch_points(batch)
  outputs: List[Dict[str, Any]] = []
  for points in points_list:
    canvas = _build_canvas_feature(model, pfe_session, points)
    canvas_np = canvas.cpu().numpy().astype(np.float32, copy=False)
    ort_outputs = backbone_session.run(None, {"canvas_feature": canvas_np})
    output_map = _resolve_output_map(backbone_session, ort_outputs)
    scores_np = output_map.get("scores", ort_outputs[0])
    bbox_preds_np = output_map.get("bbox_preds", ort_outputs[1] if len(ort_outputs) > 1 else None)
    dir_scores_np = output_map.get("dir_scores", ort_outputs[2] if len(ort_outputs) > 2 else None)
    if bbox_preds_np is None or dir_scores_np is None:
      raise KeyError(
          "Backbone ONNX must expose outputs named `scores`, `bbox_preds`, `dir_scores` "
          "or keep that exact positional order."
      )

    task_outs = _build_head_outputs(model.pts_bbox_head, scores_np, bbox_preds_np, dir_scores_np)
    outputs.append(_decode_single_sample(model, task_outs))
  return outputs


def _format_metrics(metrics: Dict[str, Any]) -> str:
  parts = []
  for key, value in metrics.items():
    if isinstance(value, (float, int, np.floating, np.integer)):
      parts.append(f"{key}: {float(value):.4f}")
    else:
      parts.append(f"{key}: {value}")
  return "  ".join(parts)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run quantitative evaluation directly on exported Apollo-style ONNX models.",
  )
  parser.add_argument("--config", required=True, help="Plain python MMDetection3D config used for dataset/model construction.")
  parser.add_argument("--name", default="onnx_model", help="Label used in printed metrics.")
  parser.add_argument("--pfe-onnx", required=True, help="Exported Apollo PFE ONNX path.")
  parser.add_argument("--backbone-onnx", required=True, help="Exported Apollo backbone/head ONNX path.")
  parser.add_argument("--split", choices=["train", "val", "test"], default="test")
  parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader num_workers.")
  parser.add_argument("--max-samples", type=int, default=0, help="Stop after this many samples (0 = full split).")
  parser.add_argument("--log-interval", type=int, default=20)
  parser.add_argument(
      "--providers",
      default="CUDAExecutionProvider,CPUExecutionProvider",
      help="Comma-separated ONNX Runtime providers priority.",
  )
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root))

  cfg_globals = _load_cfg_globals(args.config)
  dataloader_cfg, evaluator_cfg = _select_dataloader_cfg(cfg_globals, args.split)
  dataloader = _build_dataloader(dataloader_cfg, num_workers_override=args.num_workers)
  dataset_meta = getattr(dataloader.dataset, "metainfo", None)
  model = _build_model_from_cfg_globals(cfg_globals)

  providers = _parse_providers(args.providers)
  sessions = (
      _create_ort_session(Path(args.pfe_onnx), providers),
      _create_ort_session(Path(args.backbone_onnx), providers),
  )
  evaluator = _build_evaluator(evaluator_cfg, dataset_meta)

  total_samples = 0
  with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
      batch_size = len(_extract_batch_points(batch))
      outputs = _run_model_on_batch(model, sessions[0], sessions[1], batch)
      evaluator.process(data_batch=batch, data_samples=outputs)

      total_samples += batch_size
      if args.log_interval > 0 and ((batch_idx + 1) % args.log_interval == 0):
        print(f"[test_onnx] processed batches={batch_idx + 1} samples={total_samples}")
      if args.max_samples > 0 and total_samples >= args.max_samples:
        break

  eval_size = min(total_samples, len(dataloader.dataset)) if args.max_samples > 0 else len(dataloader.dataset)
  print(f"[test_onnx] split={args.split} evaluated_samples={eval_size}")
  metrics = evaluator.evaluate(eval_size)
  print(f"[test_onnx] model={args.name}    {_format_metrics(metrics)}")


if __name__ == "__main__":
  main()
