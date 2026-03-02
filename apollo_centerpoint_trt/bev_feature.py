from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
  """Scatter-add on dim=0. src: [N, C], index: [N]."""
  out = torch.zeros((dim_size, src.size(-1)), dtype=src.dtype, device=src.device)
  out.index_add_(0, index, src)
  return out


def _scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
  """Scatter-max on dim=0. src: [N, C], index: [N]."""
  if hasattr(torch.Tensor, "scatter_reduce_"):
    # PyTorch >= 2.0
    out = torch.full(
        (dim_size, src.size(-1)),
        -torch.finfo(src.dtype).max,
        dtype=src.dtype,
        device=src.device,
    )
    idx = index.view(-1, 1).expand(-1, src.size(-1))
    out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    return out
  try:
    # Optional dependency for PyTorch 1.x
    from torch_scatter import scatter_max  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scatter_max requires PyTorch>=2.0 (scatter_reduce_) or torch-scatter"
    ) from exc
  out, _ = scatter_max(src, index, dim=0, dim_size=dim_size)
  return out


@dataclass
class ApolloBevFeatureConfig:
  # point cloud range
  min_x_range: float = -51.2
  max_x_range: float = 51.2
  min_y_range: float = -51.2
  max_y_range: float = 51.2
  min_z_range: float = -3.5
  max_z_range: float = 3.5

  # voxel size (pillar-style: z is a single bin in Apollo config)
  voxel_x_size: float = 0.2
  voxel_y_size: float = 0.2
  voxel_z_size: float = 7.0

  enable_rotate_45degree: bool = True
  use_input_norm: bool = True

  # intensity scale to match Apollo C++ (PointXYZIT intensity is typically 0..255)
  intensity_scale: float = 255.0

  # cnnseg extra features
  use_cnnseg_features: bool = True
  height_bin_min_height: float = -3.0
  height_bin_max_height: float = 2.0
  height_bin_voxel_size: float = 0.5

  # dims (must match training model/backbone input)
  pillar_feature_dim: int = 48
  cnnseg_feature_dim: int = 16  # 6 + height_bin_dim

  def grid_size(self) -> Tuple[int, int]:
    gx = int((self.max_x_range - self.min_x_range) / self.voxel_x_size)
    gy = int((self.max_y_range - self.min_y_range) / self.voxel_y_size)
    return gx, gy

  def map_size(self) -> int:
    gx, gy = self.grid_size()
    return gx * gy

  def height_bin_dim(self) -> int:
    return int((self.height_bin_max_height - self.height_bin_min_height) /
               self.height_bin_voxel_size)


class ApolloBevFeatureGenerator(torch.nn.Module):
  """Generate Apollo-style `voxels` features and dense `canvas_feature`.

  This implements the same feature math as:
    - GeneratePfnFeature*()
    - GenerateBackboneFeature*()
  in `modules/perception/lidar/lib/detector/center_point_trt/center_point_trt.cc`.
  """

  def __init__(self, cfg: ApolloBevFeatureConfig) -> None:
    super().__init__()
    self.cfg = cfg
    gx, gy = cfg.grid_size()
    self.grid_x_size = gx
    self.grid_y_size = gy
    self.map_size = gx * gy
    self.x_offset = cfg.voxel_x_size / 2.0 + cfg.min_x_range
    self.y_offset = cfg.voxel_y_size / 2.0 + cfg.min_y_range

    if cfg.use_cnnseg_features:
      hdim = cfg.height_bin_dim()
      expected = 6 + hdim
      if cfg.cnnseg_feature_dim != expected:
        raise ValueError(
            f"cnnseg_feature_dim mismatch: got {cfg.cnnseg_feature_dim}, "
            f"expected {expected} (6 + height_bin_dim={hdim})"
        )

  def _rotate45(self, xy: torch.Tensor) -> torch.Tensor:
    # Apollo uses:
    # px = 0.707*x - 0.707*y
    # py = 0.707*x + 0.707*y
    c = 1.0 / math.sqrt(2.0)
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack((c * x - c * y, c * x + c * y), dim=-1)

  def _pc_to_pixel(
      self,
      pc_axis: torch.Tensor,
      voxel_size: float,
      start_range: float,
  ) -> torch.Tensor:
    fpixel = (pc_axis - start_range) / voxel_size
    return torch.floor(fpixel).to(torch.int64)

  @torch.no_grad()
  def build_voxel_features(
      self, points_xyzi: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-point voxel features + grid indices.

    Args:
      points_xyzi: [N, 4] (x,y,z,intensity) in lidar frame.

    Returns:
      voxels: [M, 9] (filtered valid points only)
      grid_idx: [M] linear grid index = y * grid_x + x
      grid_xy: [M, 2] integer (x, y) coords
    """
    if points_xyzi.numel() == 0:
      empty = points_xyzi.new_zeros((0, 9))
      return empty, points_xyzi.new_zeros((0,), dtype=torch.long), points_xyzi.new_zeros((0, 2), dtype=torch.long)

    cfg = self.cfg
    pts = points_xyzi
    xy = pts[:, 0:2]
    z = pts[:, 2:3]
    intensity = pts[:, 3:4]

    if cfg.enable_rotate_45degree:
      xy_rot = self._rotate45(xy)
    else:
      xy_rot = xy

    pos_x = self._pc_to_pixel(xy_rot[:, 0], cfg.voxel_x_size, cfg.min_x_range)
    pos_y = self._pc_to_pixel(xy_rot[:, 1], cfg.voxel_y_size, cfg.min_y_range)

    valid = (
        (pos_x >= 0) & (pos_x < self.grid_x_size) &
        (pos_y >= 0) & (pos_y < self.grid_y_size)
    )
    if not torch.any(valid):
      empty = pts.new_zeros((0, 9))
      return empty, pts.new_zeros((0,), dtype=torch.long), pts.new_zeros((0, 2), dtype=torch.long)

    pos_x = pos_x[valid]
    pos_y = pos_y[valid]
    xy_rot = xy_rot[valid]
    z = z[valid]
    intensity = intensity[valid]

    grid_idx = pos_y * self.grid_x_size + pos_x
    grid_xy = torch.stack((pos_x, pos_y), dim=-1)

    # cluster mean in each grid cell (Apollo computes it on rotated px/py)
    ones = torch.ones((grid_idx.size(0), 1), dtype=xy_rot.dtype, device=xy_rot.device)
    cnt = _scatter_add(ones, grid_idx, self.map_size).clamp_min_(1.0)
    sum_xyz = _scatter_add(torch.cat((xy_rot, z), dim=-1), grid_idx, self.map_size)
    mean_xyz = sum_xyz / cnt
    mean_xyz_pts = mean_xyz[grid_idx]

    dx_mean = xy_rot[:, 0:1] - mean_xyz_pts[:, 0:1]
    dy_mean = xy_rot[:, 1:2] - mean_xyz_pts[:, 1:2]
    dz_mean = z - mean_xyz_pts[:, 2:3]

    dx_center = xy_rot[:, 0:1] - (pos_x.to(xy_rot.dtype).unsqueeze(-1) * cfg.voxel_x_size + self.x_offset)
    dy_center = xy_rot[:, 1:2] - (pos_y.to(xy_rot.dtype).unsqueeze(-1) * cfg.voxel_y_size + self.y_offset)

    if cfg.use_input_norm:
      x0 = xy_rot[:, 0:1] / cfg.max_x_range
      y0 = xy_rot[:, 1:2] / cfg.max_y_range
      z0 = z / cfg.max_z_range
      i0 = intensity / cfg.intensity_scale
    else:
      x0 = xy[:, 0:1][valid]
      y0 = xy[:, 1:2][valid]
      z0 = z
      i0 = intensity

    voxels = torch.cat(
        (x0, y0, z0, i0, dx_mean, dy_mean, dz_mean, dx_center, dy_center),
        dim=-1,
    )
    return voxels, grid_idx.to(torch.long), grid_xy.to(torch.long)

  @torch.no_grad()
  def build_cnnseg_features(
      self,
      grid_idx: torch.Tensor,
      pz: torch.Tensor,
      intensity: torch.Tensor,
  ) -> torch.Tensor:
    """Build per-grid extra features (Apollo `use_cnnseg_features`)."""
    cfg = self.cfg
    device = grid_idx.device
    dtype = pz.dtype

    max_h = _scatter_max(pz, grid_idx, self.map_size)  # [map, 1]
    sum_h = _scatter_add(pz, grid_idx, self.map_size)
    sum_i = _scatter_add(intensity, grid_idx, self.map_size)
    ones = torch.ones((grid_idx.size(0), 1), dtype=dtype, device=device)
    cnt = _scatter_add(ones, grid_idx, self.map_size)
    mean_h = sum_h / cnt.clamp_min_(1.0)
    mean_i = sum_i / cnt.clamp_min_(1.0)

    # top_intensity: intensity of (one of) the highest points in the cell.
    # We compute it as max intensity among points that reach max height.
    max_h_pts = max_h[grid_idx]
    same_as_max = (pz >= (max_h_pts - 1e-6))
    masked_i = torch.where(same_as_max, intensity, intensity.new_full(intensity.shape, -torch.finfo(dtype).max))
    top_i = _scatter_max(masked_i, grid_idx, self.map_size)

    nonempty = (cnt > 0).to(dtype)
    count_feat = torch.log1p(cnt).to(dtype)

    # height bin occupancy
    hdim = cfg.height_bin_dim()
    if hdim <= 0:
      raise ValueError("Invalid height_bin_dim")
    bin_idx = torch.floor((pz.squeeze(-1) - cfg.height_bin_min_height) / cfg.height_bin_voxel_size).to(torch.long)
    bin_idx = torch.clamp(bin_idx, 0, hdim - 1)
    height_bin = torch.zeros((hdim, self.map_size), dtype=dtype, device=device)
    height_bin[bin_idx, grid_idx] = 1.0

    # pack to [map, 6 + hdim]
    feats = torch.cat(
        (
            max_h,          # 1
            mean_h,         # 1
            top_i,          # 1
            mean_i,         # 1
            count_feat,     # 1
            nonempty,       # 1
            height_bin.t(),  # hdim
        ),
        dim=-1,
    )
    return feats  # [map, cnnseg_feature_dim]

  def forward(
      self,
      points_xyzi: torch.Tensor,
      pillar_feature: torch.Tensor,
  ) -> torch.Tensor:
    """Build dense canvas_feature from points and pillar_feature.

    Args:
      points_xyzi: [N, 4] points (x,y,z,i) in lidar frame.
      pillar_feature: [M, pillar_feature_dim] from PFE, must align with valid points.

    Returns:
      canvas_feature: [1, (pillar_feature_dim + cnnseg_feature_dim), grid_x, grid_y]
    """
    cfg = self.cfg
    voxels, grid_idx, _ = self.build_voxel_features(points_xyzi)
    if voxels.size(0) != pillar_feature.size(0):
      raise ValueError(
          f"pillar_feature rows must match valid points. "
          f"got voxels={voxels.size(0)}, pillar_feature={pillar_feature.size(0)}"
      )

    # scatter max-pool pillar_feature to BEV grid (must keep grad for training)
    bev_pf = _scatter_max(pillar_feature, grid_idx, self.map_size)  # [map, 48]

    channels = [bev_pf]
    if cfg.use_cnnseg_features:
      # Use rotated pz/intensity domain to match C++ statistics.
      # build_voxel_features already filtered points; reuse those points.
      # Recompute pz/intensity for the valid subset to avoid extra masking state.
      # Note: intensity is normalized like C++ when use_input_norm is true.
      valid_pts = points_xyzi  # original
      # Recreate the valid mask implicitly by using voxels' normalized x channel:
      # build_voxel_features already performed filtering; use its selection order.
      # We cannot recover the valid mask here without redoing filtering; do it once:
      pts = points_xyzi
      xy = pts[:, 0:2]
      z = pts[:, 2:3]
      intensity = pts[:, 3:4]
      if cfg.enable_rotate_45degree:
        xy_rot = self._rotate45(xy)
      else:
        xy_rot = xy
      pos_x = self._pc_to_pixel(xy_rot[:, 0], cfg.voxel_x_size, cfg.min_x_range)
      pos_y = self._pc_to_pixel(xy_rot[:, 1], cfg.voxel_y_size, cfg.min_y_range)
      valid = (
          (pos_x >= 0) & (pos_x < self.grid_x_size) &
          (pos_y >= 0) & (pos_y < self.grid_y_size)
      )
      z = z[valid]
      intensity = intensity[valid]
      if cfg.use_input_norm:
        intensity = intensity / cfg.intensity_scale
      # cnnseg features are derived from raw points only; they do not need grad.
      cnn = self.build_cnnseg_features(grid_idx, z, intensity).detach()
      channels.append(cnn)

    bev = torch.cat(channels, dim=-1)  # [map, C]
    C = bev.size(-1)
    # Layout: match Apollo config (grid_x, grid_y). Note the square default.
    canvas = bev.t().contiguous().view(1, C, self.grid_x_size, self.grid_y_size)
    return canvas
