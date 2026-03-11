from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .pcd import load_pcd_xyzi


DEFAULT_SCENE = "default"
DEFAULT_CLASSES = ["car", "pedestrian", "bicycle", "traffic_cone"]
DEFAULT_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}


@dataclass
class SceneMeta:
  scene_id: str
  sensor_name: str
  coord_system: str
  has_sweeps: bool
  frame_count: int
  source_dataset: str
  source_coord: Optional[str]
  target_coord: str
  transform_matrix: Optional[List[List[float]]]

  def to_dict(self) -> Dict[str, Any]:
    return {
        "scene_id": self.scene_id,
        "sensor_name": self.sensor_name,
        "coord_system": self.coord_system,
        "has_sweeps": self.has_sweeps,
        "frame_count": self.frame_count,
        "source_dataset": self.source_dataset,
        "source_coord": self.source_coord,
        "target_coord": self.target_coord,
        "transform_matrix": self.transform_matrix,
    }


def _dataset_dirs(root: Path) -> Dict[str, Path]:
  return {
      "root": root,
      "raw": root / "raw",
      "converted": root / "converted",
      "infos": root / "converted" / "infos",
      "splits": root / "converted" / "splits",
  }


def _scene_dirs(root: Path, scene: str) -> Dict[str, Path]:
  dirs = _dataset_dirs(root)
  return {
      **dirs,
      "raw_scene": dirs["raw"] / scene,
      "raw_lidar": dirs["raw"] / scene / "lidar",
      "raw_labels": dirs["raw"] / scene / "labels",
      "scene_meta": dirs["raw"] / scene / "scene_meta.json",
      "converted_points": dirs["converted"] / "points" / scene,
      "converted_labels": dirs["converted"] / "labels" / scene,
  }


def _ensure_dirs(paths: Sequence[Path]) -> None:
  for path in paths:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _dump_json(path: Path, data: Dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


def _load_transform(args: argparse.Namespace) -> Optional[np.ndarray]:
  if args.transform_values:
    values = [float(v) for v in args.transform_values.split(",") if v.strip()]
    if len(values) != 16:
      raise ValueError("--transform-values must contain exactly 16 comma-separated floats.")
    return np.asarray(values, dtype=np.float32).reshape(4, 4)
  if args.transform_file:
    path = Path(args.transform_file)
    payload = _load_json(path)
    matrix = payload.get("transform_matrix", payload)
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (4, 4):
      raise ValueError(f"Transform matrix must be 4x4, got {arr.shape} from {path}")
    return arr
  return None


def _apply_transform(points_xyzi: np.ndarray, transform: Optional[np.ndarray]) -> np.ndarray:
  if transform is None or points_xyzi.size == 0:
    return points_xyzi
  xyz1 = np.concatenate(
      (points_xyzi[:, :3], np.ones((points_xyzi.shape[0], 1), dtype=np.float32)),
      axis=1,
  )
  xyz_t = xyz1 @ transform.T
  out = points_xyzi.copy()
  out[:, :3] = xyz_t[:, :3]
  return out


def _normalize_intensity(points_xyzi: np.ndarray) -> np.ndarray:
  out = points_xyzi.copy()
  intensity = out[:, 3]
  finite = np.isfinite(intensity)
  if np.any(finite):
    max_value = float(np.max(intensity[finite]))
    if max_value <= 1.0 + 1e-6:
      intensity = intensity * 255.0
    intensity = np.clip(intensity, 0.0, 255.0)
  out[:, 3] = intensity.astype(np.float32, copy=False)
  return out.astype(np.float32, copy=False)


def _iter_scene_pcds(scene_dir: Path) -> List[Path]:
  if scene_dir.is_file():
    return [scene_dir]
  files = sorted(scene_dir.rglob("*.pcd"))
  if not files:
    raise FileNotFoundError(f"No `.pcd` files found under {scene_dir}")
  return files


def _safe_frame_stem(scene_input: Path, file_path: Path) -> str:
  rel = file_path.relative_to(scene_input) if scene_input.is_dir() else file_path.name
  if isinstance(rel, Path):
    rel_str = str(rel.with_suffix(""))
  else:
    rel_str = str(rel)
  return rel_str.replace("/", "__").replace("\\", "__")


def _default_label_json(frame_id: str, scene: str, point_cloud_path: str) -> Dict[str, Any]:
  return {
      "frame_id": frame_id,
      "scene_id": scene,
      "timestamp_us": None,
      "point_cloud_path": point_cloud_path,
      "coord_system": "apollo_lidar",
      "objects": [],
  }


def cmd_init(args: argparse.Namespace) -> None:
  root = Path(args.root).resolve()
  scene = args.scene or DEFAULT_SCENE
  scene_paths = _scene_dirs(root, scene)
  _ensure_dirs([
      scene_paths["raw_lidar"],
      scene_paths["raw_labels"],
      scene_paths["converted_points"],
      scene_paths["converted_labels"],
      scene_paths["infos"],
      scene_paths["splits"],
  ])

  transform = _load_transform(args)
  meta = SceneMeta(
      scene_id=scene,
      sensor_name=args.sensor_name,
      coord_system=args.target_coord,
      has_sweeps=False,
      frame_count=0,
      source_dataset=args.source_dataset,
      source_coord=args.source_coord,
      target_coord=args.target_coord,
      transform_matrix=transform.tolist() if transform is not None else None,
  )
  _dump_json(scene_paths["scene_meta"], meta.to_dict())
  print(f"[custom_lidar_dataset] initialized scene={scene} under {root}")


def cmd_encode_label(args: argparse.Namespace) -> None:
  root = Path(args.root).resolve()
  scene = args.scene or DEFAULT_SCENE
  scene_paths = _scene_dirs(root, scene)
  _ensure_dirs([scene_paths["converted_labels"]])
  stub_path = scene_paths["converted_labels"] / "_pending_label_adapter.json"
  payload = {
      "status": "pending",
      "scene_id": scene,
      "input_path": str(Path(args.input).resolve()),
      "label_format": args.label_format,
      "target_schema": "CUSTOM_DATASET.md",
      "note": "Label conversion adapter not implemented yet. Keep this file as the integration stub.",
  }
  _dump_json(stub_path, payload)
  print(f"[custom_lidar_dataset] label adapter stub created: {stub_path}")


def cmd_encode_pcd(args: argparse.Namespace) -> None:
  root = Path(args.root).resolve()
  scene = args.scene or DEFAULT_SCENE
  scene_paths = _scene_dirs(root, scene)
  scene_input = Path(args.input).resolve()
  pcd_files = _iter_scene_pcds(scene_input)

  meta = _load_json(scene_paths["scene_meta"]) if scene_paths["scene_meta"].exists() else {}
  transform = None
  if meta.get("transform_matrix") is not None:
    transform = np.asarray(meta["transform_matrix"], dtype=np.float32)

  converted_count = 0
  for pcd_path in pcd_files:
    frame_id = _safe_frame_stem(scene_input, pcd_path)
    points_xyzi = load_pcd_xyzi(pcd_path)
    points_xyzi = _apply_transform(points_xyzi, transform)
    points_xyzi = _normalize_intensity(points_xyzi)

    out_bin = scene_paths["converted_points"] / f"{frame_id}.bin"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    points_xyzi.astype(np.float32, copy=False).tofile(out_bin)

    out_label = scene_paths["converted_labels"] / f"{frame_id}.json"
    if not out_label.exists():
      rel_bin = out_bin.relative_to(root).as_posix()
      _dump_json(out_label, _default_label_json(frame_id, scene, rel_bin))
    converted_count += 1

  meta["frame_count"] = converted_count
  if meta:
    _dump_json(scene_paths["scene_meta"], meta)
  print(f"[custom_lidar_dataset] encoded {converted_count} PCD files for scene={scene}")


def _collect_scene_samples(root: Path, scene: str) -> List[Dict[str, Any]]:
  scene_paths = _scene_dirs(root, scene)
  label_files = sorted(scene_paths["converted_labels"].glob("*.json"))
  samples: List[Dict[str, Any]] = []
  for label_path in label_files:
    payload = _load_json(label_path)
    point_cloud_path = payload.get("point_cloud_path", None)
    if not point_cloud_path:
      point_cloud_path = str((scene_paths["converted_points"] / f"{label_path.stem}.bin").relative_to(root).as_posix())

    objects = payload.get("objects", [])
    instances = []
    for obj in objects:
      class_name = obj.get("class_name")
      if class_name not in DEFAULT_CLASS_TO_LABEL:
        continue
      bbox = obj.get("bbox_3d", {})
      center = bbox.get("center", [0.0, 0.0, 0.0])
      size = bbox.get("size", [0.0, 0.0, 0.0])
      yaw = float(bbox.get("yaw", 0.0))
      instances.append({
          "bbox_3d": [
              float(center[0]),
              float(center[1]),
              float(center[2]),
              float(size[0]),
              float(size[1]),
              float(size[2]),
              yaw,
          ],
          "bbox_label_3d": int(DEFAULT_CLASS_TO_LABEL[class_name]),
          "bbox_label_name": class_name,
          "num_lidar_pts": int(obj.get("num_lidar_pts", -1)),
      })

    samples.append({
        "sample_idx": f"{scene}/{label_path.stem}",
        "scene_id": scene,
        "frame_id": payload.get("frame_id", label_path.stem),
        "timestamp": payload.get("timestamp_us", None),
        "lidar_points": {
            "lidar_path": point_cloud_path,
            "num_pts_feats": 4,
        },
        "instances": instances,
        "sweeps": payload.get("sweeps", []),
        "metainfo": {
            "source_dataset": _load_json(scene_paths["scene_meta"]).get("source_dataset", "custom_lidar"),
            "coord_system": payload.get("coord_system", "apollo_lidar"),
        },
    })
  return samples


def _split_scenes(scenes: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
  if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
    raise ValueError("Require 0 < train_ratio, 0 <= val_ratio and train_ratio + val_ratio < 1.")
  test_ratio = 1.0 - train_ratio - val_ratio
  rng = random.Random(seed)
  shuffled = list(scenes)
  rng.shuffle(shuffled)

  if len(shuffled) <= 1:
    return {"train": shuffled, "val": [], "test": []}

  n = len(shuffled)
  n_train = max(1, int(round(n * train_ratio)))
  n_val = int(round(n * val_ratio))
  if n_train + n_val >= n:
    n_val = max(0, n - n_train - 1)
  n_test = n - n_train - n_val
  if test_ratio > 0 and n_test <= 0 and n >= 3:
    n_test = 1
    if n_val > 0:
      n_val -= 1
    else:
      n_train -= 1

  return {
      "train": shuffled[:n_train],
      "val": shuffled[n_train:n_train + n_val],
      "test": shuffled[n_train + n_val:],
  }


def _write_split_file(path: Path, scenes: Sequence[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for scene in scenes:
      f.write(f"{scene}\n")


def _dump_pkl(path: Path, payload: Dict[str, Any]) -> None:
  import pickle

  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as f:
    pickle.dump(payload, f)


def cmd_convert(args: argparse.Namespace) -> None:
  root = Path(args.root).resolve()
  dirs = _dataset_dirs(root)
  raw_root = dirs["raw"]
  if not raw_root.exists():
    raise FileNotFoundError(f"Dataset root has no raw directory: {raw_root}")

  all_scenes = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
  if not all_scenes:
    raise RuntimeError(f"No scenes found under {raw_root}")

  split_map = _split_scenes(all_scenes, args.train_ratio, args.val_ratio, args.seed)
  for split_name, scenes in split_map.items():
    _write_split_file(dirs["splits"] / f"{split_name}.txt", scenes)

  samples_by_scene: Dict[str, List[Dict[str, Any]]] = {}
  for scene in all_scenes:
    samples_by_scene[scene] = _collect_scene_samples(root, scene)

  metainfo = {
      "dataset_type": "CustomLidarDataset",
      "classes": DEFAULT_CLASSES,
      "box_type_3d": "LiDAR",
      "coord_system": "apollo_lidar",
      "version": "v1",
  }
  split_payloads = defaultdict(list)
  for split_name, scenes in split_map.items():
    for scene in scenes:
      split_payloads[split_name].extend(samples_by_scene.get(scene, []))

  for split_name, data_list in split_payloads.items():
    payload = {
        "metainfo": metainfo,
        "data_list": data_list,
    }
    _dump_pkl(dirs["infos"] / f"custom_infos_{split_name}.pkl", payload)

  print(
      "[custom_lidar_dataset] converted dataset: "
      f"train={len(split_payloads['train'])}, "
      f"val={len(split_payloads['val'])}, "
      f"test={len(split_payloads['test'])}"
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="custom_lidar_dataset",
      description="Standalone utility for preparing the repo-local custom LiDAR dataset format.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  init_parser = subparsers.add_parser("init", help="Create dataset directory structure and scene meta.")
  init_parser.add_argument("--root", required=True, help="Dataset root, e.g. data/custom_lidar")
  init_parser.add_argument("--scene", default=DEFAULT_SCENE)
  init_parser.add_argument("--source-dataset", default="business_recording_v1")
  init_parser.add_argument("--sensor-name", default="lidar_top")
  init_parser.add_argument("--source-coord", default=None)
  init_parser.add_argument("--target-coord", default="apollo_lidar")
  init_parser.add_argument("--transform-file", default=None, help="JSON file containing a 4x4 transform_matrix.")
  init_parser.add_argument("--transform-values", default=None, help="16 comma-separated floats for a 4x4 transform.")
  init_parser.set_defaults(func=cmd_init)

  label_parser = subparsers.add_parser(
      "encode-label",
      help="Placeholder interface for converting labeling-tool output into the internal JSON schema.",
  )
  label_parser.add_argument("--root", required=True)
  label_parser.add_argument("--scene", default=DEFAULT_SCENE)
  label_parser.add_argument("--input", required=True, help="Path to the external label export.")
  label_parser.add_argument("--label-format", default="unknown", help="External tool format identifier.")
  label_parser.set_defaults(func=cmd_encode_label)

  pcd_parser = subparsers.add_parser("encode-pcd", help="Convert raw .pcd files into 4-float .bin point clouds.")
  pcd_parser.add_argument("--root", required=True)
  pcd_parser.add_argument("--scene", default=DEFAULT_SCENE)
  pcd_parser.add_argument("--input", required=True, help="A .pcd file or a directory containing .pcd files.")
  pcd_parser.set_defaults(func=cmd_encode_pcd)

  convert_parser = subparsers.add_parser(
      "convert",
      help="Build custom_infos_{train,val,test}.pkl and scene-level split files.",
  )
  convert_parser.add_argument("--root", required=True)
  convert_parser.add_argument("--train-ratio", type=float, default=0.8)
  convert_parser.add_argument("--val-ratio", type=float, default=0.1)
  convert_parser.add_argument("--seed", type=int, default=0)
  convert_parser.set_defaults(func=cmd_convert)
  return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
  parser = build_parser()
  args = parser.parse_args(argv)
  args.func(args)
