from __future__ import annotations

import torch
from torch import nn


class ApolloPFE(nn.Module):
  """Apollo-style point-wise PFE.

  Matches the shipped `cpdet_pfe.onnx` structure in this repo:
  - Squeeze (remove singleton dims)
  - MatMul/Linear (no bias)
  - BatchNorm
  - ReLU

  Input:
    - voxels: [N, 9] or [N, 1, 9, 1]
  Output:
    - pillar_feature: [N, 48]
  """

  def __init__(
      self,
      in_channels: int = 9,
      out_channels: int = 48,
      bn_eps: float = 1e-3,
      bn_momentum: float = 0.01,
  ) -> None:
    super().__init__()
    self.fc = nn.Linear(in_channels, out_channels, bias=False)
    self.bn = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_momentum)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, voxels: torch.Tensor) -> torch.Tensor:
    x = voxels
    # Accept [N, 1, C, 1] (Apollo C++ shape) or [N, C] (training-friendly).
    if x.dim() == 4:
      x = x.squeeze(-1).squeeze(1)
    elif x.dim() == 3 and x.size(1) == 1:
      x = x.squeeze(1)
    if x.dim() != 2:
      raise ValueError(f"Unexpected voxels shape: {tuple(voxels.shape)}")
    x = self.fc(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

