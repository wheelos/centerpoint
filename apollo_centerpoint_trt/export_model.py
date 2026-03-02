from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from apollo_centerpoint_trt.pfe import ApolloPFE


_REGISTERED = False


def _ensure_mmdet3d_modules_registered() -> None:
  """Best-effort registry initialization for MMDet3D/MMDet modules.

  In MMDet3D 1.x, some components used by 3D models (e.g. losses like
  `GaussianFocalLoss`) are registered under the `mmdet` scope. When we build
  modules from plain python globals (empty export path), we may not go through
  the official entrypoints that call `register_all_modules()`.
  """
  global _REGISTERED
  if _REGISTERED:
    return
  _REGISTERED = True

  try:  # pragma: no cover
    from mmdet3d.utils import register_all_modules  # type: ignore
    import inspect

    sig = inspect.signature(register_all_modules)
    kwargs = {}
    # Common signatures across versions:
    # - register_all_modules(init_default_scope=True)
    # - register_all_modules(init_default_scope=False)
    if "init_default_scope" in sig.parameters:
      kwargs["init_default_scope"] = True
    register_all_modules(**kwargs)  # type: ignore
    return
  except Exception:
    pass

  # Fallback: import common packages so they register into their scopes.
  try:  # pragma: no cover
    import mmdet  # noqa: F401
    import mmdet.models  # noqa: F401
  except Exception:
    pass
  try:  # pragma: no cover
    import mmdet3d  # noqa: F401
    import mmdet3d.models  # noqa: F401
  except Exception:
    pass


def _build_mmdet3d_module(cfg: dict) -> nn.Module:
  try:
    _ensure_mmdet3d_modules_registered()
    from mmdet3d.registry import MODELS  # type: ignore

    return MODELS.build(cfg)
  except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to build module from mmdet3d.registry.MODELS. "
        "Please ensure you run this in an MMDetection3D environment and that "
        f"the module type exists. cfg.type={cfg.get('type')}"
    ) from exc


class CenterPointTRTExportModel(nn.Module):
  """A minimal model container for exporting Apollo-style ONNX.

  This avoids depending on MMDet3D `CenterPoint` detector init signatures,
  and only keeps the submodules required by the Apollo inference contract:
    - pfe: ApolloPFE (voxels -> pillar_feature)
    - pts_backbone
    - pts_neck (optional)
    - pts_bbox_head
  """

  def __init__(
      self,
      pts_backbone: dict,
      pts_bbox_head: dict,
      pts_neck: Optional[dict] = None,
      pfe_cfg: Optional[dict] = None,
  ) -> None:
    super().__init__()
    self.pfe = ApolloPFE(**(pfe_cfg or {}))
    self.pts_backbone = _build_mmdet3d_module(pts_backbone)
    self.pts_neck = _build_mmdet3d_module(pts_neck) if pts_neck else None
    self.pts_bbox_head = _build_mmdet3d_module(pts_bbox_head)

  def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover
    raise RuntimeError(
        "CenterPointTRTExportModel is not meant for inference directly. "
        "Use export_wrappers.BackboneHeadExportWrapper for ONNX export."
    )
