from __future__ import annotations

from typing import Any, List, Optional

import torch
from torch import nn


class ApolloNeckLite(nn.Module):
  """Apollo-style neck that matches shipped `cpdet_backbone.onnx`.

  Input: list of 3 feature maps from SECOND backbone:
    - x0: [B, 64, 256, 256]
    - x1: [B, 128, 128, 128]
    - x2: [B, 256, 64, 64]

  Process:
    - x0 -> Conv2d(k=2,s=2) -> ReLU          => [B,128,128,128]
    - x1 -> Conv2d(k=1,s=1) -> ReLU          => [B,128,128,128]
    - x2 -> ConvT2d(k=2,s=2) -> BN -> ReLU   => [B,128,128,128]
    - concat along channel                   => [B,384,128,128]

  Note: only the deconv branch uses BatchNorm in the shipped ONNX.
  """

  def __init__(
      self,
      in_channels: Optional[List[int]] = None,
      out_channels: int = 128,
      bn_eps: float = 1e-3,
      bn_momentum: float = 0.01,
  ) -> None:
    super().__init__()
    in_channels = in_channels or [64, 128, 256]
    if len(in_channels) != 3:
      raise ValueError("ApolloNeckLite expects 3 input feature levels")
    c0, c1, c2 = in_channels

    self.downsample0 = nn.Conv2d(c0, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
    self.lateral1 = nn.Conv2d(c1, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    # Apollo shipped ONNX has no bias on this ConvTranspose (W only).
    self.upsample2 = nn.ConvTranspose2d(c2, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x: Any) -> torch.Tensor:
    if not isinstance(x, (list, tuple)) or len(x) < 3:
      raise TypeError("ApolloNeckLite expects a list/tuple of 3 feature maps")
    x0, x1, x2 = x[0], x[1], x[2]
    y0 = self.relu(self.downsample0(x0))
    y1 = self.relu(self.lateral1(x1))
    y2 = self.relu(self.bn2(self.upsample2(x2)))
    return torch.cat([y0, y1, y2], dim=1)


def register_to_mmdet3d() -> None:
  try:  # pragma: no cover
    from mmdet3d.registry import MODELS  # type: ignore

    MODELS.register_module(module=ApolloNeckLite, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
