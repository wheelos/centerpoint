from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

try:  # pragma: no cover
  import torch
except Exception:  # pragma: no cover
  torch = None  # type: ignore

try:  # pragma: no cover
  from mmengine.hooks import Hook  # type: ignore
except Exception:  # pragma: no cover
  Hook = object  # type: ignore


class ApolloTrainEvalHook(Hook):
  """Run an extra evaluation pass on the training split.

  This is intended for debugging training correctness rather than model
  selection, so it only logs metrics and does not interact with checkpointing or
  early stopping.
  """

  priority = "LOW"

  def __init__(
      self,
      dataloader: dict,
      evaluator: dict,
      interval: int = 1,
      start_epoch: int = 1,
  ) -> None:
    if interval < 1:
      raise ValueError("interval must be >= 1")
    self.dataloader_cfg = deepcopy(dataloader)
    self.evaluator_cfg = deepcopy(evaluator)
    self.interval = int(interval)
    self.start_epoch = int(start_epoch)
    self._dataloader: Optional[Any] = None
    self._evaluator: Optional[Any] = None

  def _build_once(self, runner: Any) -> None:
    if self._dataloader is None:
      self._dataloader = runner.build_dataloader(deepcopy(self.dataloader_cfg))
    if self._evaluator is None:
      self._evaluator = runner.build_evaluator(deepcopy(self.evaluator_cfg))
      dataset_meta = getattr(self._dataloader.dataset, "metainfo", None)
      if dataset_meta is not None and hasattr(self._evaluator, "dataset_meta"):
        self._evaluator.dataset_meta = dataset_meta

  def _should_run(self, runner: Any) -> bool:
    current_epoch = int(runner.epoch) + 1
    if current_epoch < self.start_epoch:
      return False
    return (current_epoch - self.start_epoch) % self.interval == 0

  def after_train_epoch(self, runner: Any) -> None:
    if torch is None or not self._should_run(runner):
      return
    self._build_once(runner)

    model = runner.model
    was_training = model.training
    model.eval()
    try:
      for data_batch in self._dataloader:
        with torch.no_grad():
          outputs = model.val_step(data_batch)
        self._evaluator.process(data_batch=data_batch, data_samples=outputs)

      metrics = self._evaluator.evaluate(len(self._dataloader.dataset))
      if hasattr(runner, "logger"):
        pretty = "  ".join(
            f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}"
            for key, value in metrics.items()
        )
        runner.logger.info(f"Epoch(train_eval) [{int(runner.epoch) + 1}]    {pretty}")
    finally:
      if was_training:
        model.train()


def register_to_mmdet3d() -> None:
  try:  # pragma: no cover
    from mmdet3d.registry import HOOKS  # type: ignore

    HOOKS.register_module(module=ApolloTrainEvalHook, force=True)
    return
  except Exception:
    pass

  try:  # pragma: no cover
    from mmengine.registry import HOOKS  # type: ignore

    HOOKS.register_module(module=ApolloTrainEvalHook, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
