#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import sys
from typing import Optional

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Train CenterPointTRT with MMEngine/MMDetection3D")
  parser.add_argument("config", help="Path to a python config file")
  parser.add_argument(
      "--work-dir",
      default="",
      help="Override `work_dir` from config. Defaults to ./work_dirs/<config_name>.",
  )
  parser.add_argument(
      "--resume",
      nargs="?",
      const="auto",
      default=None,
      help="Resume training. Use `--resume` to auto-resume latest checkpoint, "
      "or `--resume /path/to/ckpt.pth` to resume from a specific checkpoint.",
  )
  parser.add_argument(
      "--load-from",
      default="",
      help="Load weights from checkpoint without marking the run as resumed.",
  )
  parser.add_argument(
      "--amp",
      action="store_true",
      help="Switch OptimWrapper to AmpOptimWrapper when possible.",
  )
  parser.add_argument(
      "--cfg-options",
      nargs="+",
      action=DictAction,
      help="Override config values, e.g. key=value key2=\"[a,b]\"",
  )
  parser.add_argument(
      "--launcher",
      choices=["none", "pytorch", "slurm", "mpi"],
      default="none",
      help="Job launcher for distributed training.",
  )
  parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
  return parser.parse_args()


def _register_mmdet3d_modules() -> None:
  try:
    from mmdet3d.utils import register_all_modules  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Cannot import `mmdet3d.utils.register_all_modules`. "
        "Please run this script inside an MMDetection3D 1.x environment."
    ) from exc

  kwargs = {}
  try:
    sig = inspect.signature(register_all_modules)
    if "init_default_scope" in sig.parameters:
      kwargs["init_default_scope"] = False
  except Exception:
    pass
  register_all_modules(**kwargs)  # type: ignore[arg-type]


def _apply_custom_imports(cfg: Config) -> None:
  custom_imports = cfg.get("custom_imports")
  if not custom_imports:
    return
  imports = custom_imports.get("imports", [])
  allow_failed_imports = custom_imports.get("allow_failed_imports", False)
  import_modules_from_strings(imports, allow_failed_imports)


def _derive_work_dir(cfg: Config, config_path: str, cli_work_dir: str) -> str:
  if cli_work_dir:
    return cli_work_dir
  if cfg.get("work_dir", None):
    return cfg.work_dir
  config_name = Path(config_path).stem
  return str(Path("./work_dirs") / config_name)


def _apply_resume_settings(cfg: Config, resume_arg: Optional[str],
                           load_from_arg: str) -> None:
  if load_from_arg:
    cfg.load_from = load_from_arg

  if resume_arg is None:
    cfg.resume = bool(cfg.get("resume", False))
    return

  cfg.resume = True
  if resume_arg != "auto":
    cfg.load_from = resume_arg


def _enable_amp(cfg: Config) -> None:
  optim_wrapper = cfg.get("optim_wrapper", None)
  if optim_wrapper is None:
    raise KeyError("Config missing `optim_wrapper`; cannot enable AMP.")

  wrapper_type = optim_wrapper.get("type", "OptimWrapper")
  if wrapper_type == "AmpOptimWrapper":
    optim_wrapper.setdefault("loss_scale", "dynamic")
    return
  if wrapper_type != "OptimWrapper":
    raise ValueError(
        f"--amp only supports OptimWrapper -> AmpOptimWrapper, got {wrapper_type}"
    )

  optim_wrapper["type"] = "AmpOptimWrapper"
  optim_wrapper.setdefault("loss_scale", "dynamic")


def _validate_training_cfg(cfg: Config) -> None:
  required = [
      "model",
      "train_dataloader",
      "train_cfg",
      "optim_wrapper",
  ]
  missing = [key for key in required if key not in cfg]
  if missing:
    raise KeyError(
        "Training config is incomplete. Missing keys: " + ", ".join(missing))


def main() -> None:
  args = parse_args()
  os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

  plugin_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(plugin_root))

  cfg = Config.fromfile(args.config)
  if args.cfg_options:
    cfg.merge_from_dict(args.cfg_options)

  _register_mmdet3d_modules()
  _apply_custom_imports(cfg)

  cfg.launcher = args.launcher
  cfg.work_dir = _derive_work_dir(cfg, args.config, args.work_dir)
  cfg.setdefault("default_scope", "mmdet3d")
  _apply_resume_settings(cfg, args.resume, args.load_from)

  if args.amp:
    _enable_amp(cfg)

  _validate_training_cfg(cfg)

  runner = Runner.from_cfg(cfg)
  runner.train()


if __name__ == "__main__":
  main()
