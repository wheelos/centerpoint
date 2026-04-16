#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
from typing import List

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from custom_lidar_tool.pcd import load_pcd_xyzi


def parse_args() -> argparse.Namespace:
  default_pcd_path = REPO_ROOT / "reference" / "test.pcd"
  parser = argparse.ArgumentParser(
      description="Run a smoke test against centerpoint inference service and render a BEV SVG."
  )
  parser.add_argument(
      "pcd_path",
      nargs="?",
      type=Path,
      default=default_pcd_path,
      help=f"Path to the local .pcd file. Defaults to {default_pcd_path}",
  )
  parser.add_argument(
      "--endpoint",
      default="http://127.0.0.1:8010",
      help="Inference service base URL",
  )
  parser.add_argument(
      "--api-mode",
      choices=("upload", "batch"),
      default="upload",
      help="Use /infer(upload) or /v1/inference/batch(path-based)",
  )
  parser.add_argument(
      "--sample-uri",
      default=None,
      help="Optional sample URI for batch mode. Defaults to file://<absolute pcd_path>.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=None,
      help="Output SVG path. Defaults to <pcd_stem>_detections.svg",
  )
  parser.add_argument(
      "--response-json",
      type=Path,
      default=None,
      help="Optional path to save raw API response JSON",
  )
  parser.add_argument(
      "--score-thr",
      type=float,
      default=0.3,
      help="Filter boxes below this score in visualization",
  )
  parser.add_argument(
      "--max-points",
      type=int,
      default=40000,
      help="Maximum number of points rendered in SVG",
  )
  parser.add_argument(
      "--canvas-size",
      type=int,
      default=1400,
      help="Square SVG canvas size in pixels",
  )
  return parser.parse_args()


def _call_upload_api(endpoint: str, points_xyzi: np.ndarray) -> dict:
  payload = io.BytesIO()
  np.save(payload, points_xyzi.astype(np.float32, copy=False))
  payload.seek(0)
  response = requests.post(
      f"{endpoint.rstrip('/')}/infer",
      files={"file": ("pointcloud.npy", payload.getvalue(), "application/octet-stream")},
      timeout=300,
  )
  response.raise_for_status()
  body = response.json()
  predictions = body.get("predictions") or []
  if not predictions:
    return {"boxes": [], "scores": [], "labels": []}, body
  return predictions[0], body


def _call_batch_api(endpoint: str, pcd_path: Path, sample_uri: str | None) -> dict:
  sample_uri = sample_uri or f"file://{pcd_path.resolve()}"
  payload = {
      "request_id": pcd_path.stem,
      "items": [{"sample_id": pcd_path.stem, "sample_uri": sample_uri}],
  }
  response = requests.post(
      f"{endpoint.rstrip('/')}/v1/inference/batch",
      json=payload,
      timeout=300,
  )
  response.raise_for_status()
  body = response.json()
  results = body.get("results") or []
  if not results:
    errors = body.get("errors") or []
    if errors:
      raise RuntimeError(f"batch inference failed: {errors[0]}")
    return {"boxes": [], "scores": [], "labels": []}, body

  detections = results[0].get("detections_3d") or []
  decoded = {
      "boxes": [],
      "scores": [],
      "labels": [],
      "labels_text": [],
  }
  for det in detections:
    center = det.get("center_xyz") or [0.0, 0.0, 0.0]
    size = det.get("size_lwh") or [0.0, 0.0, 0.0]
    decoded["boxes"].append(
        [
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(size[0]),
            float(size[1]),
            float(size[2]),
            float(det.get("yaw") or 0.0),
        ]
    )
    decoded["scores"].append(float(det.get("score") or 0.0))
    decoded["labels"].append(int(det.get("label_id") or 0))
    decoded["labels_text"].append(str(det.get("label") or "unknown"))
  return decoded, body


def _subsample_points(points_xyzi: np.ndarray, max_points: int) -> np.ndarray:
  if points_xyzi.shape[0] <= max_points:
    return points_xyzi
  idx = np.linspace(0, points_xyzi.shape[0] - 1, num=max_points, dtype=np.int64)
  return points_xyzi[idx]


def _rotation_corners(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
  dx = length * 0.5
  dy = width * 0.5
  corners = np.asarray(
      [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]],
      dtype=np.float32,
  )
  c = math.cos(yaw)
  s = math.sin(yaw)
  rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
  return corners @ rot.T + np.asarray([cx, cy], dtype=np.float32)


def _label_text(decoded: dict, idx: int) -> str:
  labels_text = decoded.get("labels_text")
  if labels_text and idx < len(labels_text):
    return str(labels_text[idx])
  labels = decoded.get("labels") or []
  if idx < len(labels):
    return f"class_{int(labels[idx])}"
  return "unknown"


def _world_bounds(points_xyzi: np.ndarray, decoded: dict) -> tuple[float, float, float, float]:
  xs = points_xyzi[:, 0]
  ys = points_xyzi[:, 1]
  min_x = float(np.percentile(xs, 1))
  max_x = float(np.percentile(xs, 99))
  min_y = float(np.percentile(ys, 1))
  max_y = float(np.percentile(ys, 99))

  for box in decoded.get("boxes") or []:
    if len(box) < 7:
      continue
    corners = _rotation_corners(
        float(box[0]),
        float(box[1]),
        float(box[3]),
        float(box[4]),
        float(box[6]),
    )
    min_x = min(min_x, float(np.min(corners[:, 0])))
    max_x = max(max_x, float(np.max(corners[:, 0])))
    min_y = min(min_y, float(np.min(corners[:, 1])))
    max_y = max(max_y, float(np.max(corners[:, 1])))

  pad_x = max(3.0, (max_x - min_x) * 0.08)
  pad_y = max(3.0, (max_y - min_y) * 0.08)
  return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _project_xy(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    canvas_size: int,
    margin: int,
) -> tuple[float, float]:
  usable = float(canvas_size - margin * 2)
  world_w = max(max_x - min_x, 1e-6)
  world_h = max(max_y - min_y, 1e-6)
  scale = min(usable / world_w, usable / world_h)
  offset_x = margin + (usable - world_w * scale) * 0.5
  offset_y = margin + (usable - world_h * scale) * 0.5
  px = offset_x + (x - min_x) * scale
  py = canvas_size - (offset_y + (y - min_y) * scale)
  return px, py


def render_svg(
    points_xyzi: np.ndarray,
    decoded: dict,
    output_path: Path,
    canvas_size: int,
    score_thr: float,
    max_points: int,
) -> int:
  margin = 60
  min_x, max_x, min_y, max_y = _world_bounds(points_xyzi, decoded)
  sampled = _subsample_points(points_xyzi, max_points=max_points)

  lines: List[str] = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">',
      '<rect width="100%" height="100%" fill="#081018" />',
      '<g font-family="monospace">',
  ]

  for px, py in (
      _project_xy(float(pt[0]), float(pt[1]), min_x, max_x, min_y, max_y, canvas_size, margin)
      for pt in sampled
  ):
    lines.append(
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="0.7" fill="#8aa4b8" fill-opacity="0.55" />'
    )

  rendered = 0
  palette = ["#ff6b6b", "#ffd166", "#06d6a0", "#4cc9f0", "#f72585", "#f77f00"]
  boxes = decoded.get("boxes") or []
  scores = decoded.get("scores") or []
  for idx, box in enumerate(boxes):
    if len(box) < 7:
      continue
    score = float(scores[idx]) if idx < len(scores) else 0.0
    if score < score_thr:
      continue

    rendered += 1
    color = palette[idx % len(palette)]
    corners = _rotation_corners(
        float(box[0]),
        float(box[1]),
        float(box[3]),
        float(box[4]),
        float(box[6]),
    )
    polygon = " ".join(
        f"{px:.2f},{py:.2f}"
        for px, py in (
            _project_xy(float(x), float(y), min_x, max_x, min_y, max_y, canvas_size, margin)
            for x, y in corners
        )
    )
    front_mid = np.mean(corners[:2], axis=0)
    center_px, center_py = _project_xy(
        float(box[0]), float(box[1]), min_x, max_x, min_y, max_y, canvas_size, margin
    )
    front_px, front_py = _project_xy(
        float(front_mid[0]), float(front_mid[1]), min_x, max_x, min_y, max_y, canvas_size, margin
    )
    label = _label_text(decoded, idx)

    lines.append(
        f'<polygon points="{polygon}" fill="none" stroke="{color}" stroke-width="2.2" />'
    )
    lines.append(
        f'<line x1="{center_px:.2f}" y1="{center_py:.2f}" x2="{front_px:.2f}" y2="{front_py:.2f}" stroke="{color}" stroke-width="2.2" />'
    )
    lines.append(
        f'<text x="{center_px + 6:.2f}" y="{center_py - 6:.2f}" fill="{color}" font-size="16">{label} {score:.2f}</text>'
    )

  lines.extend(
      [
          '</g>',
          f'<text x="24" y="36" fill="#e8f1f7" font-family="monospace" font-size="22">points={points_xyzi.shape[0]} rendered={sampled.shape[0]} boxes={rendered}</text>',
          '</svg>',
      ]
  )
  output_path.write_text("\n".join(lines), encoding="utf-8")
  return rendered


def main() -> int:
  args = parse_args()
  pcd_path = args.pcd_path.resolve()
  if not pcd_path.exists():
    raise FileNotFoundError(pcd_path)

  output_path = args.output or pcd_path.with_name(f"{pcd_path.stem}_detections.svg")
  response_json_path = args.response_json or output_path.with_suffix(".json")

  points_xyzi = load_pcd_xyzi(pcd_path)
  if points_xyzi.ndim != 2 or points_xyzi.shape[1] != 4:
    raise ValueError(f"expected Nx4 XYZI points, got shape={points_xyzi.shape}")

  if args.api_mode == "upload":
    decoded, response_body = _call_upload_api(args.endpoint, points_xyzi)
  else:
    decoded, response_body = _call_batch_api(args.endpoint, pcd_path, args.sample_uri)

  response_json_path.write_text(json.dumps(response_body, indent=2, ensure_ascii=False), encoding="utf-8")
  rendered_boxes = render_svg(
      points_xyzi=points_xyzi,
      decoded=decoded,
      output_path=output_path,
      canvas_size=args.canvas_size,
      score_thr=args.score_thr,
      max_points=args.max_points,
  )

  print(f"pcd={pcd_path}")
  print(f"api_mode={args.api_mode}")
  print(f"endpoint={args.endpoint}")
  if args.api_mode == "batch":
    print(f"sample_uri={args.sample_uri or f'file://{pcd_path}'}")
  print(f"points={points_xyzi.shape[0]}")
  print(f"rendered_boxes={rendered_boxes}")
  print(f"response_json={response_json_path}")
  print(f"visualization={output_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
