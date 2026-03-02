from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import nn


class ApolloSecondBackboneLite(nn.Module):
  """Apollo-style SECOND backbone (Conv+ReLU, no BN) matching shipped ONNX.

  Input:
    - x: [B, 64, 512, 512] (canvas_feature)
  Output:
    - [x0, x1, x2] where:
      - x0: [B,  64, 256, 256]
      - x1: [B, 128, 128, 128]
      - x2: [B, 256,  64,  64]

  Structure (after a 1x1 stem):
    - stage0: Conv3x3 s2 + 3x Conv3x3 s1 (64ch)
    - stage1: Conv3x3 s2 + 5x Conv3x3 s1 (128ch)
    - stage2: Conv3x3 s2 + 5x Conv3x3 s1 (256ch)

  This avoids MMDet3D version drift in `SECOND` init signature and produces an
  ONNX graph that is much closer to Apollo's shipped `cpdet_backbone.onnx`.
  """

  def __init__(
      self,
      in_channels: int = 64,
      stem_out_channels: int = 64,
      stage_channels: Sequence[int] = (64, 128, 256),
      stage_blocks: Sequence[int] = (3, 5, 5),
  ) -> None:
    super().__init__()
    if len(stage_channels) != 3 or len(stage_blocks) != 3:
      raise ValueError("ApolloSecondBackboneLite expects 3 stages")

    self.stem = nn.Sequential(
        nn.Conv2d(
            in_channels,
            stem_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
        nn.ReLU(inplace=True),
    )

    prev_c = stem_out_channels
    stages: List[nn.Sequential] = []
    for out_c, num_extra in zip(stage_channels, stage_blocks):
      layers: List[nn.Module] = []
      # first conv in stage: stride-2 downsample
      layers.append(
          nn.Conv2d(prev_c, out_c, kernel_size=3, stride=2, padding=1, bias=True))
      layers.append(nn.ReLU(inplace=True))
      # extra convs: stride-1
      for _ in range(int(num_extra)):
        layers.append(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
      stages.append(nn.Sequential(*layers))
      prev_c = out_c

    self.stage0 = stages[0]
    self.stage1 = stages[1]
    self.stage2 = stages[2]

  def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
    x = self.stem(x)
    x0 = self.stage0(x)
    x1 = self.stage1(x0)
    x2 = self.stage2(x1)
    return [x0, x1, x2]


class ApolloBackboneWithStem(nn.Module):
  """Wrap an MMDet3D backbone with a Conv1x1+ReLU stem.

  Apollo's shipped `cpdet_backbone.onnx` begins with a 1x1 Conv (64->64) before
  the stride-2 3x3 conv of the FIRST SECOND block. MMDet3D's `SECOND` backbone
  typically starts with the stride-2 conv directly. This wrapper adds the stem
  so the exported ONNX structure matches Apollo.
  """

  def __init__(
      self,
      backbone: Any,
      stem_in_channels: int = 64,
      stem_out_channels: int = 64,
      stem_kernel_size: int = 1,
      stem_stride: int = 1,
      stem_padding: int = 0,
      stem_bias: bool = True,
      stem_relu: bool = True,
  ) -> None:
    super().__init__()
    self.stem = nn.Conv2d(
        stem_in_channels,
        stem_out_channels,
        kernel_size=stem_kernel_size,
        stride=stem_stride,
        padding=stem_padding,
        bias=stem_bias,
    )
    self.relu = nn.ReLU(inplace=True) if stem_relu else nn.Identity()

    if isinstance(backbone, dict):
      from mmdet3d.registry import MODELS  # type: ignore

      self.backbone = MODELS.build(backbone)
    else:
      self.backbone = backbone

  def forward(self, x: torch.Tensor):
    x = self.relu(self.stem(x))
    return self.backbone(x)


def register_to_mmdet3d() -> None:
  """Register to MMDet3D registry if available."""
  try:  # pragma: no cover
    from mmdet3d.registry import MODELS  # type: ignore

    MODELS.register_module(module=ApolloSecondBackboneLite, force=True)
    MODELS.register_module(module=ApolloBackboneWithStem, force=True)
  except Exception:
    pass


try:  # pragma: no cover
  register_to_mmdet3d()
except Exception:
  pass
