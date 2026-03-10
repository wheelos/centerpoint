# from __future__ import annotations

import inspect
import math
from typing import Any, List, Optional

try:
  from .bev_feature import ApolloBevFeatureConfig, ApolloBevFeatureGenerator
  from .pfe import ApolloPFE
except ImportError:  # pragma: no cover
  # When this file is executed directly (e.g. via runpy), relative imports
  # have no package context. Fall back to absolute imports.
  from apollo_centerpoint_trt.bev_feature import (  # type: ignore
      ApolloBevFeatureConfig,
      ApolloBevFeatureGenerator,
  )
  from apollo_centerpoint_trt.pfe import ApolloPFE  # type: ignore


def _is_lazy_proxy(obj: Any) -> bool:
  mod = getattr(obj.__class__, "__module__", "")
  return isinstance(mod, str) and mod.startswith("mmengine.config.lazy")


def _is_mmengine_lazy_object(obj: Any) -> bool:
  # When MMEngine parses a "lazy config", it may replace objects with
  # LazyObject/LazyAttr proxies. Calling methods on those proxies raises
  # RuntimeError during config parsing.
  try:  # pragma: no cover
    from mmengine.config.lazy import LazyObject  # type: ignore

    return isinstance(obj, LazyObject)
  except Exception:
    return False


def _find_non_finite(obj: Any, prefix: str = "") -> Optional[str]:
  if torch is not None and torch.is_tensor(obj):
    if not torch.isfinite(obj).all():
      return prefix or "<tensor>"
    return None
  if isinstance(obj, dict):
    for key, value in obj.items():
      bad = _find_non_finite(value, f"{prefix}.{key}" if prefix else str(key))
      if bad is not None:
        return bad
    return None
  if isinstance(obj, (list, tuple)):
    for index, value in enumerate(obj):
      bad = _find_non_finite(value, f"{prefix}[{index}]" if prefix else f"[{index}]")
      if bad is not None:
        return bad
  return None


def _rotate_boxes_inplace(boxes: Any, angle_rad: float) -> Any:
  """Rotate 3D boxes around z-axis by `angle_rad` in-place if possible."""
  if not hasattr(boxes, "tensor"):
    if hasattr(boxes, "rotate"):
      try:
        boxes.rotate(angle_rad)
        return boxes
      except TypeError:
        out = boxes.rotate(angle_rad)
        return out[0] if isinstance(out, (tuple, list)) else out
    raise TypeError(f"Unsupported boxes type for rotation: {type(boxes)}")

  box_tensor = getattr(boxes, "tensor")
  if torch is not None and torch.is_tensor(box_tensor):
    angle = box_tensor.new_tensor(float(angle_rad))
    c = torch.cos(angle)
    s = torch.sin(angle)

    x = box_tensor[:, 0].clone()
    y = box_tensor[:, 1].clone()
    box_tensor[:, 0] = c * x - s * y
    box_tensor[:, 1] = s * x + c * y
    if box_tensor.size(1) >= 7:
      box_tensor[:, 6] = box_tensor[:, 6] + angle
    if box_tensor.size(1) >= 9:
      vx = box_tensor[:, 7].clone()
      vy = box_tensor[:, 8].clone()
      box_tensor[:, 7] = c * vx - s * vy
      box_tensor[:, 8] = s * vx + c * vy
    return boxes

  t = boxes.tensor.clone()
  c = math.cos(angle_rad)
  s = math.sin(angle_rad)
  x = t[:, 0].clone()
  y = t[:, 1].clone()
  t[:, 0] = c * x - s * y
  t[:, 1] = s * x + c * y
  if t.size(1) >= 7:
    t[:, 6] = t[:, 6] + angle_rad
  if t.size(1) >= 9:
    vx = t[:, 7].clone()
    vy = t[:, 8].clone()
    t[:, 7] = c * vx - s * vy
    t[:, 8] = s * vx + c * vy
  if hasattr(boxes, "new_box"):
    return boxes.new_box(t)
  return boxes.__class__(t)


try:
  import torch  # type: ignore
except Exception:  # pragma: no cover
  torch = None  # type: ignore


if torch is not None and not _is_lazy_proxy(torch):

  try:
    from mmengine.model import BaseModel  # type: ignore
  except Exception:  # pragma: no cover
    BaseModel = torch.nn.Module  # type: ignore
  _HAS_MMENGINE = BaseModel is not torch.nn.Module

  def _build_centerpoint_backbone_head(centerpoint_cfg: dict) -> torch.nn.Module:
    """Build a minimal module holding pts_backbone/pts_neck/pts_bbox_head.

    MMDet3D's `CenterPoint` detector init signature differs across versions.
    For training/export with Apollo preprocessing, we only need the submodules.
    If the provided dict contains `pts_backbone`/`pts_bbox_head`, prefer
    composing them instead of instantiating the detector.
    """
    from mmdet3d.registry import MODELS  # type: ignore

    if "pts_backbone" in centerpoint_cfg and "pts_bbox_head" in centerpoint_cfg:
      pts_backbone = MODELS.build(centerpoint_cfg["pts_backbone"])
      pts_neck_cfg = centerpoint_cfg.get("pts_neck")
      pts_neck = MODELS.build(pts_neck_cfg) if isinstance(pts_neck_cfg,
                                                         dict) else None
      pts_bbox_head = MODELS.build(centerpoint_cfg["pts_bbox_head"])

      class _BackboneHead(torch.nn.Module):

        def __init__(self, bb, neck, head):
          super().__init__()
          self.pts_backbone = bb
          self.pts_neck = neck
          self.pts_bbox_head = head

      return _BackboneHead(pts_backbone, pts_neck, pts_bbox_head)

    return MODELS.build(centerpoint_cfg)

  def _maybe_build_data_preprocessor(cfg: Any) -> Any:
    if cfg is None or not isinstance(cfg, dict):
      return cfg
    if "type" not in cfg:
      return cfg
    # In MMDet3D 1.x, data preprocessors are registered in mmdet3d.registry.MODELS.
    try:  # pragma: no cover
      from mmdet3d.registry import MODELS  # type: ignore

      return MODELS.build(cfg)
    except Exception:
      return cfg


  class CenterPointTRTDetector(BaseModel):
    """MMDet3D 1.x-trainable model that matches Apollo CenterPointTRT preprocessing.

    This is a plugin intended to be used in MMDetection3D training:
    - Generates Apollo-style voxel/BEV features (same as car-side C++)
    - Uses a tiny point-wise PFE (Linear+BN+ReLU)
    - Reuses MMDetection3D's SECOND/CenterHead backbone+neck+head via composition

    Notes:
    - We intentionally do NOT build the full `CenterPoint` detector wrapper,
      because its init signature (e.g. `pts_voxel_layer`) varies by MMDet3D
      version and is not part of Apollo's TRT contract.
    - Only `pts_backbone/pts_neck/pts_bbox_head` participate in ONNX export.
    """

    def __init__(
        self,
        centerpoint: Any = None,
        bev_feature_cfg: Optional[dict] = None,
        pfe_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
      if _HAS_MMENGINE:
        super().__init__(
            data_preprocessor=_maybe_build_data_preprocessor(data_preprocessor),
            init_cfg=init_cfg,
        )
      else:  # pragma: no cover
        super().__init__()

      if centerpoint is None:
        raise ValueError("`centerpoint` must be provided (dict or nn.Module)")

      core = centerpoint
      if isinstance(core, dict):  # pragma: no cover
        core = _build_centerpoint_backbone_head(core)

      # Keep both: the core container and the standard attribute names.
      self.centerpoint = core
      self.pts_backbone = getattr(core, "pts_backbone", None)
      self.pts_neck = getattr(core, "pts_neck", None)
      self.pts_bbox_head = getattr(core, "pts_bbox_head", None)
      if self.pts_backbone is None or self.pts_bbox_head is None:
        raise AttributeError(
            "centerpoint must expose `pts_backbone` and `pts_bbox_head`")

      self.pfe = ApolloPFE(**(pfe_cfg or {}))
      self.bev_gen = ApolloBevFeatureGenerator(
          ApolloBevFeatureConfig(**(bev_feature_cfg or {})))

    def _points_to_xyzi(self, pts: torch.Tensor) -> torch.Tensor:
      if pts.numel() == 0:
        return pts.new_zeros((0, 4))
      if pts.size(-1) < 4:
        raise ValueError(
            f"points must have at least 4 dims (x,y,z,i), got {pts.shape}")
      return pts[:, :4].contiguous()

    def _build_canvas_batch(self, points: List[torch.Tensor]) -> torch.Tensor:
      canvases = []
      for sample_idx, pts in enumerate(points):
        pts_xyzi = self._points_to_xyzi(pts)
        bad_pts = _find_non_finite(pts_xyzi, f"points[{sample_idx}]")
        if bad_pts is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_pts}")
        voxels, _, _ = self.bev_gen.build_voxel_features(pts_xyzi)
        bad_voxels = _find_non_finite(voxels, f"voxels[{sample_idx}]")
        if bad_voxels is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_voxels}")
        if voxels.size(0) == 0:
          gx = self.bev_gen.grid_x_size
          gy = self.bev_gen.grid_y_size
          channels = (
              self.bev_gen.cfg.pillar_feature_dim +
              self.bev_gen.cfg.cnnseg_feature_dim
          )
          canvas = pts_xyzi.new_zeros((1, channels, gx, gy))
          bad_canvas = _find_non_finite(canvas, f"canvas[{sample_idx}]")
          if bad_canvas is not None:
            raise RuntimeError(f"Non-finite value detected in {bad_canvas}")
          canvases.append(canvas.squeeze(0))
          continue
        pillar_feature = self.pfe(voxels)  # [M, 48]
        bad_pillar = _find_non_finite(pillar_feature, f"pillar_feature[{sample_idx}]")
        if bad_pillar is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_pillar}")
        canvas = self.bev_gen(pts_xyzi, pillar_feature)  # [1, C, gx, gy]
        bad_canvas = _find_non_finite(canvas, f"canvas[{sample_idx}]")
        if bad_canvas is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_canvas}")
        canvases.append(canvas.squeeze(0))
      return torch.stack(canvases, dim=0)

    def extract_feat(self, points: List[torch.Tensor]) -> Any:
      canvas = self._build_canvas_batch(points)
      bad_canvas = _find_non_finite(canvas, "canvas_batch")
      if bad_canvas is not None:
        raise RuntimeError(f"Non-finite value detected in {bad_canvas}")
      x = self.pts_backbone(canvas)
      bad_backbone = _find_non_finite(x, "pts_backbone")
      if bad_backbone is not None:
        raise RuntimeError(f"Non-finite value detected in {bad_backbone}")
      if self.pts_neck is not None:
        x = self.pts_neck(x)
        bad_neck = _find_non_finite(x, "pts_neck")
        if bad_neck is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_neck}")
      return x

    def _rotate_gt_inplace(self, data_samples: list, angle_rad: float) -> None:
      for ds in data_samples:
        gt = getattr(ds, "gt_instances_3d", None)
        if gt is None:
          continue
        b = getattr(gt, "bboxes_3d", None)
        if b is None:
          continue
        rotated = _rotate_boxes_inplace(b, angle_rad)
        if rotated is not b:
          setattr(gt, "bboxes_3d", rotated)

    def _head_loss(self, feats: Any, data_samples: list) -> dict:
      feats_list = feats if isinstance(feats, (list, tuple)) else [feats]
      # Preferred API in MMDet3D 1.x heads: loss(x, batch_data_samples)
      loss_fn = getattr(self.pts_bbox_head, "loss", None)
      if callable(loss_fn):
        try:
          return loss_fn(feats_list, data_samples)
        except TypeError:
          pass

      outs = self.pts_bbox_head(feats_list)
      loss_by_feat = getattr(self.pts_bbox_head, "loss_by_feat", None)
      if callable(loss_by_feat):
        try:
          return loss_by_feat(outs, data_samples)
        except TypeError:
          pass

      # Fallback to older explicit GT lists signature.
      gt_bboxes_3d = []
      gt_labels_3d = []
      img_metas = []
      for ds in data_samples:
        gt = getattr(ds, "gt_instances_3d", None)
        if gt is None:
          raise AttributeError("data_samples missing gt_instances_3d")
        gt_bboxes_3d.append(getattr(gt, "bboxes_3d"))
        gt_labels_3d.append(getattr(gt, "labels_3d"))
        meta = getattr(ds, "metainfo", None)
        img_metas.append(meta() if callable(meta) else (meta or {}))

      if callable(loss_fn):
        try:
          return loss_fn(gt_bboxes_3d, gt_labels_3d, outs, img_metas)
        except TypeError:
          return loss_fn(outs, gt_bboxes_3d, gt_labels_3d, img_metas)
      raise RuntimeError("pts_bbox_head does not provide a usable loss API")

    def _head_predict(self, feats: Any, data_samples: list):
      feats_list = feats if isinstance(feats, (list, tuple)) else [feats]
      pred_fn = getattr(self.pts_bbox_head, "predict", None)
      if callable(pred_fn):
        try:
          return pred_fn(feats_list, data_samples)
        except TypeError:
          pass
      outs = self.pts_bbox_head(feats_list)
      pred_by_feat = getattr(self.pts_bbox_head, "predict_by_feat", None)
      if callable(pred_by_feat):
        return pred_by_feat(outs, data_samples)
      get_bboxes = getattr(self.pts_bbox_head, "get_bboxes", None)
      if callable(get_bboxes):
        img_metas = []
        for ds in data_samples:
          meta = getattr(ds, "metainfo", None)
          img_metas.append(meta() if callable(meta) else (meta or {}))
        return get_bboxes(outs, img_metas)
      return None

    def forward(self, inputs: Any, data_samples: Optional[list] = None, mode: str = "tensor"):
      # MMEngine BaseModel entrypoint used by Runner.
      if mode == "loss":
        if data_samples is None:
          raise ValueError("data_samples is required for mode='loss'")
        return self.loss(inputs, data_samples)
      if mode == "predict":
        return self.predict(inputs, data_samples)
      if mode == "tensor":
        return self._forward(inputs, data_samples)
      raise ValueError(f"Unsupported mode: {mode}")

    def _forward(self, inputs: Any, data_samples: Optional[list] = None):
      points = inputs.get("points") if isinstance(inputs, dict) else inputs
      if not isinstance(points, list):
        raise TypeError("inputs must be a dict with `points` or a list of tensors")
      feats = self.extract_feat(points)
      feats_list = feats if isinstance(feats, (list, tuple)) else [feats]
      return self.pts_bbox_head(feats_list)

    def loss(self, inputs: Any, data_samples: list) -> dict:
      points = inputs.get("points") if isinstance(inputs, dict) else inputs
      if not isinstance(points, list):
        raise TypeError("inputs must be a dict with `points` or a list of tensors")

      angle = math.pi / 4.0 if self.bev_gen.cfg.enable_rotate_45degree else 0.0
      if angle != 0.0:
        self._rotate_gt_inplace(data_samples, angle)
      try:
        feats = self.extract_feat(points)
        bad_feat = _find_non_finite(feats, "feats")
        if bad_feat is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_feat}")
        losses = self._head_loss(feats, data_samples)
        bad_loss = _find_non_finite(losses, "losses")
        if bad_loss is not None:
          raise RuntimeError(f"Non-finite value detected in {bad_loss}")
      finally:
        if angle != 0.0:
          self._rotate_gt_inplace(data_samples, -angle)

      if not isinstance(losses, dict):
        raise TypeError(f"Head loss must return dict, got: {type(losses)}")
      return losses

    def predict(self, inputs: Any, data_samples: Optional[list] = None):
      points = inputs.get("points") if isinstance(inputs, dict) else inputs
      if not isinstance(points, list):
        raise TypeError("inputs must be a dict with `points` or a list of tensors")

      if data_samples is None:
        # Allow inference without data_samples; create placeholders.
        data_samples = [{} for _ in range(len(points))]  # type: ignore

      feats = self.extract_feat(points)
      pred = self._head_predict(feats, data_samples)
      if pred is None:
        return data_samples

      # If preprocessing rotates 45deg, rotate decoded boxes back to lidar frame.
      angle = math.pi / 4.0 if self.bev_gen.cfg.enable_rotate_45degree else 0.0

      # MMDet3D head.predict usually returns List[InstanceData].
      if isinstance(pred, list) and len(pred) == len(data_samples):
        for ds, inst in zip(data_samples, pred):
          if angle != 0.0:
            b = getattr(inst, "bboxes_3d", None)
            if b is not None:
              _rotate_boxes_inplace(b, -angle)
          # Attach prediction back to datasample if possible.
          try:
            setattr(ds, "pred_instances_3d", inst)
          except Exception:
            pass
        return data_samples

      return pred

else:
  # In MMEngine lazy-config parsing, importing `torch` returns a LazyObject.
  # Defining a torch.nn.Module subclass will crash during class creation.
  CenterPointTRTDetector = None  # type: ignore


def register_to_mmdet3d() -> None:
  """Register CenterPointTRTDetector to MMDetection3D registry if possible."""
  if CenterPointTRTDetector is None:
    return

  # MMDetection3D 1.x (mmengine registry)
  try:  # pragma: no cover
    from mmdet3d.registry import MODELS  # type: ignore

    if not _is_mmengine_lazy_object(MODELS):
      MODELS.register_module(module=CenterPointTRTDetector, force=True)
    return
  except Exception:
    pass

  # MMDetection3D 0.x (mmcv registry)
  try:  # pragma: no cover
    from mmdet3d.models.builder import DETECTORS  # type: ignore

    if not _is_mmengine_lazy_object(DETECTORS):
      DETECTORS.register_module(module=CenterPointTRTDetector, force=True)
  except Exception:
    pass


# Best-effort auto registration when imported in a real runtime (non-lazy) env.
try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
