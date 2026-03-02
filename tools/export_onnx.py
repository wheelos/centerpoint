#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import runpy
from pathlib import Path
import sys

import torch


def _resolve_device(requested: str) -> str:
  """Return a usable torch device string.

  If user requests CUDA but the current PyTorch build/runtime has no CUDA,
  fall back to CPU so export still works in a CPU-only environment.
  """
  req = (requested or "").strip() or "cpu"
  if not req.startswith("cuda"):
    return req

  # CPU-only PyTorch build: torch.version.cuda is None and initializing CUDA
  # would raise "Torch not compiled with CUDA enabled".
  if getattr(torch.version, "cuda", None) is None:
    print(
        f"[export_onnx] Requested device={req} but this PyTorch has no CUDA. "
        "Falling back to cpu."
    )
    return "cpu"

  try:
    if not torch.cuda.is_available():
      print(
          f"[export_onnx] Requested device={req} but CUDA is not available at runtime. "
          "Falling back to cpu."
      )
      return "cpu"
  except Exception:
    print(
        f"[export_onnx] Requested device={req} but CUDA init failed. Falling back to cpu."
    )
    return "cpu"

  return req


def _try_get_pfe(model) -> torch.nn.Module:
  candidates = [
      "pfe",
      "pts_pfe",
  ]
  for name in candidates:
    if hasattr(model, name):
      return getattr(model, name)

  # try common mmdet3d module locations
  for parent_name in ["pts_voxel_encoder", "voxel_encoder", "pts_middle_encoder"]:
    if hasattr(model, parent_name):
      parent = getattr(model, parent_name)
      if hasattr(parent, "pfe"):
        return getattr(parent, "pfe")

  raise AttributeError(
      "Cannot find PFE module on model. Expected `model.pfe` (recommended) "
      "or `model.pts_voxel_encoder.pfe`."
  )

def _onnx_export(*args, **kwargs) -> None:
  """torch.onnx.export wrapper compatible across torch versions.

  Filters unsupported kwargs (PyTorch minor versions change the signature).
  """
  # Some versions don't accept `training=None`. Only pass it when not None.
  if kwargs.get("training", None) is None:
    kwargs.pop("training", None)
  try:
    sig = inspect.signature(torch.onnx.export)
    supported = sig.parameters.keys()
    filtered = {k: v for k, v in kwargs.items() if k in supported}
  except Exception:
    filtered = kwargs
  torch.onnx.export(*args, **filtered)


def _strip_onnx_identity_nodes(path: Path) -> bool:
  """Remove Identity nodes from an ONNX graph (best-effort).

  This is purely for making the exported graph look closer to Apollo's shipped
  ONNX in Netron. It does not change semantics.
  """
  try:
    import onnx  # type: ignore
  except Exception:
    print(
        f"[export_onnx] onnx python package not found; skip Identity stripping for {path}."
    )
    return False

  model = onnx.load(str(path))
  graph = model.graph

  # Map Identity outputs -> inputs.
  replace: dict[str, str] = {}
  new_nodes = []
  for node in graph.node:
    if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
      replace[node.output[0]] = node.input[0]
    else:
      new_nodes.append(node)

  if not replace:
    return True

  def _resolve(name: str) -> str:
    seen = set()
    while name in replace and name not in seen:
      seen.add(name)
      name = replace[name]
    return name

  # Rewrite node inputs.
  for node in new_nodes:
    for i, inp in enumerate(node.input):
      if inp:
        node.input[i] = _resolve(inp)

  # Rewrite graph outputs if needed.
  for out in graph.output:
    if out.name in replace:
      out.name = _resolve(out.name)

  # Replace nodes.
  del graph.node[:]
  graph.node.extend(new_nodes)

  try:
    onnx.checker.check_model(model)
  except Exception as exc:
    print(f"[export_onnx] Identity stripping produced an invalid ONNX: {exc}.")
    return False

  onnx.save(model, str(path))
  print(f"[export_onnx] Stripped Identity nodes: {path}")
  return True


def _reorder_onnx_outputs(path: Path, names_in_order: list[str]) -> bool:
  """Reorder ONNX graph outputs to the given name order (best-effort)."""
  try:
    import onnx  # type: ignore
  except Exception:
    print(
        f"[export_onnx] onnx python package not found; skip output reordering for {path}."
    )
    return False

  model = onnx.load(str(path))
  graph = model.graph
  existing = {o.name: o for o in graph.output}
  if not existing:
    return True

  missing = [n for n in names_in_order if n not in existing]
  if missing:
    print(
        f"[export_onnx] Cannot reorder outputs for {path}; missing outputs: {missing}."
    )
    return False

  # Keep any other outputs (if present) appended at the end.
  new_outputs = [existing[n] for n in names_in_order] + [
      o for o in graph.output if o.name not in names_in_order
  ]
  del graph.output[:]
  graph.output.extend(new_outputs)

  try:
    onnx.checker.check_model(model)
  except Exception as exc:
    print(f"[export_onnx] Output reordering produced an invalid ONNX: {exc}.")
    return False
  onnx.save(model, str(path))
  print(f"[export_onnx] Reordered outputs: {path} -> {names_in_order}")
  return True


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

def _build_model_from_cfg_globals(cfg_globals: dict, device: str):
  if "model" not in cfg_globals:
    raise KeyError(
        "Config file does not define `model`. "
        "For empty export, please provide a config that includes `model = dict(...)`."
    )

  model_cfg = cfg_globals["model"]
  if isinstance(model_cfg, torch.nn.Module):
    model = model_cfg
  else:
    if not isinstance(model_cfg, dict):
      raise TypeError(f"`model` must be a dict or torch.nn.Module, got: {type(model_cfg)}")

    # MMDetection3D 1.x (mmengine registry)
    try:
      from mmdet3d.registry import MODELS  # type: ignore

      model = MODELS.build(model_cfg)
    except Exception as exc:
      raise RuntimeError(
          "Failed to build model from mmdet3d.registry.MODELS.\n"
          f"model.type={model_cfg.get('type')}\n"
          "If you are using an export-only config, consider setting "
          "`model = <torch.nn.Module instance>` instead of a dict."
      ) from exc

  model.to(device)
  model.eval()
  return model


def load_mmdet3d_model(cfg_path: str, checkpoint: str, device: str):
  # Works for many MMDetection3D versions.
  try:
    from mmdet3d.apis import init_model  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Cannot import mmdet3d.apis.init_model. "
        "Please run this script inside an MMDetection3D environment."
    ) from exc

  # If no checkpoint is provided, prefer building from python globals to avoid
  # MMEngine lazy-config parsing issues.
  if not checkpoint:
    try:
      cfg_globals = runpy.run_path(cfg_path, run_name="__mmdet3d_cfg__")
    except Exception as exc:
      raise RuntimeError(
          "Failed to execute config via python (empty export path).\n"
          f"Config: {cfg_path}\n"
          "Tip: for empty export, use a plain python config that directly defines "
          "`model = dict(...)` (no lazy/base indirection), e.g. the example under "
          "`modules/perception/lidar/tools/center_point_trt_mmdet3d/mmdet3d_example_configs/`."
      ) from exc
    _import_custom_imports(cfg_globals)
    try:
      return _build_model_from_cfg_globals(cfg_globals, device)
    except Exception as exc:
      keys = sorted([k for k in cfg_globals.keys() if not k.startswith("__")])
      raise RuntimeError(
          "Failed to build model from executed config globals.\n"
          f"Config: {cfg_path}\n"
          f"Top-level keys found: {keys}\n"
          "Tip: ensure the config defines `model = dict(...)` and that your "
          "`custom_imports` loads `apollo_centerpoint_trt` so the model type "
          "`CenterPointTRTDetector` is registered."
      ) from exc

  # Otherwise try the official init_model first (loads checkpoint).
  try:
    model = init_model(cfg_path, checkpoint, device=device)
    model.eval()
    return model
  except Exception:
    # Fallback: plain python exec + build + load checkpoint, useful when
    # Config.fromfile() in init_model cannot see `model`.
    cfg_globals = runpy.run_path(cfg_path, run_name="__mmdet3d_cfg__")
    _import_custom_imports(cfg_globals)
    model = _build_model_from_cfg_globals(cfg_globals, device)
    try:
      from mmengine.runner import load_checkpoint  # type: ignore

      load_checkpoint(model, checkpoint, map_location=device)
    except Exception:
      try:
        from mmcv.runner import load_checkpoint  # type: ignore

        load_checkpoint(model, checkpoint, map_location=device)
      except Exception:
        # If loading fails, re-raise the original init_model error for clarity.
        raise
    model.eval()
    return model


def export_pfe(pfe: torch.nn.Module, out_path: Path, opset: int):
  out_path.parent.mkdir(parents=True, exist_ok=True)
  pfe = pfe.eval()
  device = next(pfe.parameters()).device if any(
      True for _ in pfe.parameters()) else torch.device("cpu")
  voxels = torch.zeros((100000, 1, 9, 1),
                       dtype=torch.float32,
                       device=device)
  _onnx_export(
      pfe,
      (voxels,),
      str(out_path),
      opset_version=opset,
      input_names=["voxels"],
      output_names=["pillar_feature"],
      do_constant_folding=True,
      keep_initializers_as_inputs=False,
      training=getattr(torch.onnx, "TrainingMode", None).EVAL
      if hasattr(torch.onnx, "TrainingMode") else None,
      dynamic_axes={
          "voxels": {0: "point_size"},
          "pillar_feature": {0: "point_size"},
      },
  )


def export_backbone(model: torch.nn.Module, out_path: Path, opset: int):
  from apollo_centerpoint_trt.export_wrappers import BackboneHeadExportWrapper

  out_path.parent.mkdir(parents=True, exist_ok=True)
  wrapper = BackboneHeadExportWrapper(model).eval()
  device = next(wrapper.parameters()).device if any(
      True for _ in wrapper.parameters()) else torch.device("cpu")
  canvas_feature = torch.zeros((1, 64, 512, 512),
                               dtype=torch.float32,
                               device=device)
  _onnx_export(
      wrapper,
      (canvas_feature,),
      str(out_path),
      opset_version=opset,
      input_names=["canvas_feature"],
      output_names=["scores", "bbox_preds", "dir_scores"],
      do_constant_folding=True,
      keep_initializers_as_inputs=False,
      training=getattr(torch.onnx, "TrainingMode", None).EVAL
      if hasattr(torch.onnx, "TrainingMode") else None,
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, help="MMDetection3D config .py")
  parser.add_argument(
      "--checkpoint",
      default="",
      help="Checkpoint .pth. If omitted, exports an untrained (random) model for structure inspection.",
  )
  parser.add_argument("--out-dir", required=True, help="Output directory")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--opset", type=int, default=11)
  parser.add_argument("--pfe-name", default="cpdet_pfe.onnx")
  parser.add_argument("--backbone-name", default="cpdet_backbone.onnx")
  parser.add_argument(
      "--dynamic-batch",
      action="store_true",
      help="Export backbone with dynamic batch axis (Apollo shipped ONNX uses batch=1).",
  )
  parser.add_argument(
      "--strip-identity",
      action="store_true",
      help="Post-process ONNX to remove Identity nodes (requires `onnx` package).",
  )
  parser.add_argument(
      "--backbone-output-order",
      default="",
      help="Comma-separated output names order for backbone ONNX (e.g. scores,bbox_preds,dir_scores).",
  )
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # Ensure plugin import path is visible.
  plugin_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(plugin_root))

  device = _resolve_device(args.device)
  model = load_mmdet3d_model(args.config, args.checkpoint, device)

  # Helpful sanity checks for Apollo contract.
  try:
    core = model
    if not hasattr(core, "pts_bbox_head") and hasattr(core, "centerpoint"):
      core = getattr(core, "centerpoint")
    head = getattr(core, "pts_bbox_head", None)
    if head is not None and hasattr(head, "task_heads"):
      print(f"[export_onnx] pts_bbox_head.task_heads: {len(head.task_heads)}")
  except Exception:
    pass
  pfe = _try_get_pfe(model)

  export_pfe(pfe, out_dir / args.pfe_name, args.opset)
  # Optionally re-export backbone with dynamic batch.
  if args.dynamic_batch:
    def _export_backbone_dynamic():
      from apollo_centerpoint_trt.export_wrappers import BackboneHeadExportWrapper
      wrapper = BackboneHeadExportWrapper(model).eval()
      device = next(wrapper.parameters()).device if any(True for _ in wrapper.parameters()) else torch.device("cpu")
      canvas_feature = torch.zeros((1, 64, 512, 512), dtype=torch.float32, device=device)
      _onnx_export(
          wrapper,
          (canvas_feature,),
          str(out_dir / args.backbone_name),
          opset_version=args.opset,
          input_names=["canvas_feature"],
          output_names=["scores", "bbox_preds", "dir_scores"],
          do_constant_folding=True,
          keep_initializers_as_inputs=False,
          training=getattr(torch.onnx, "TrainingMode", None).EVAL if hasattr(torch.onnx, "TrainingMode") else None,
          dynamic_axes={
              "canvas_feature": {0: "batch"},
              "scores": {0: "batch"},
              "bbox_preds": {0: "batch"},
              "dir_scores": {0: "batch"},
          },
      )
    _export_backbone_dynamic()
  else:
    export_backbone(model, out_dir / args.backbone_name, args.opset)

  if args.strip_identity:
    _strip_onnx_identity_nodes(out_dir / args.pfe_name)
    _strip_onnx_identity_nodes(out_dir / args.backbone_name)

  if args.backbone_output_order:
    order = [x.strip() for x in args.backbone_output_order.split(",") if x.strip()]
    if order:
      _reorder_onnx_outputs(out_dir / args.backbone_name, order)

  print(f"Exported:\n  {out_dir / args.pfe_name}\n  {out_dir / args.backbone_name}")


if __name__ == "__main__":
  main()
