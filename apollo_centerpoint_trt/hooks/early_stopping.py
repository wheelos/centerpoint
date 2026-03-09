from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover
  from mmengine.hooks import Hook  # type: ignore
except Exception:  # pragma: no cover
  Hook = object  # type: ignore


class ApolloEarlyStoppingHook(Hook):
  """Stop training when a monitored validation metric stops improving.

  This hook is intentionally lightweight so the project does not depend on a
  specific MMEngine version shipping a built-in early stopping hook.
  """

  priority = "LOW"

  def __init__(
      self,
      monitor: str = "apollo/mAP",
      rule: str = "greater",
      patience: int = 5,
      min_delta: float = 1e-4,
      start_epoch: int = 1,
      strict: bool = False,
  ) -> None:
    if rule not in ("greater", "less"):
      raise ValueError(f"Unsupported rule: {rule}")
    if patience < 1:
      raise ValueError("patience must be >= 1")
    self.monitor = monitor
    self.rule = rule
    self.patience = int(patience)
    self.min_delta = float(min_delta)
    self.start_epoch = int(start_epoch)
    self.strict = bool(strict)

    self.best_score: Optional[float] = None
    self.num_bad_epochs = 0

  def _is_improved(self, current: float) -> bool:
    if self.best_score is None:
      return True
    if self.rule == "greater":
      return current > (self.best_score + self.min_delta)
    return current < (self.best_score - self.min_delta)

  def _request_stop(self, runner: Any, reason: str) -> None:
    if hasattr(runner, "logger"):
      runner.logger.info(reason)
    if hasattr(runner, "train_loop"):
      setattr(runner.train_loop, "stop_training", True)
    if hasattr(runner, "_train_loop"):
      setattr(runner._train_loop, "stop_training", True)
    setattr(runner, "should_stop", True)

  def after_val_epoch(self, runner: Any, metrics: Optional[dict] = None) -> None:
    if not metrics:
      return
    if runner.epoch + 1 < self.start_epoch:
      return

    if self.monitor not in metrics:
      message = (
          f"ApolloEarlyStoppingHook cannot find metric `{self.monitor}` in "
          f"validation metrics: {sorted(metrics.keys())}"
      )
      if self.strict:
        raise KeyError(message)
      if hasattr(runner, "logger"):
        runner.logger.warning(message)
      return

    current = float(metrics[self.monitor])
    if self._is_improved(current):
      self.best_score = current
      self.num_bad_epochs = 0
      if hasattr(runner, "logger"):
        runner.logger.info(
            f"ApolloEarlyStoppingHook observed improved {self.monitor}={current:.6f}"
        )
      return

    self.num_bad_epochs += 1
    if hasattr(runner, "logger"):
      runner.logger.info(
          "ApolloEarlyStoppingHook no improvement on "
          f"{self.monitor}: current={current:.6f}, best={self.best_score:.6f}, "
          f"bad_epochs={self.num_bad_epochs}/{self.patience}"
      )
    if self.num_bad_epochs >= self.patience:
      self._request_stop(
          runner,
          "ApolloEarlyStoppingHook triggered early stop on "
          f"{self.monitor} after {self.num_bad_epochs} bad epochs.",
      )


def register_to_mmdet3d() -> None:
  try:  # pragma: no cover
    from mmdet3d.registry import HOOKS  # type: ignore

    HOOKS.register_module(module=ApolloEarlyStoppingHook, force=True)
    return
  except Exception:
    pass

  try:  # pragma: no cover
    from mmengine.registry import HOOKS  # type: ignore

    HOOKS.register_module(module=ApolloEarlyStoppingHook, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass

