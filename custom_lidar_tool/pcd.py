from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PcdHeader:
  version: str
  fields: List[str]
  size: List[int]
  type: List[str]
  count: List[int]
  width: int
  height: int
  points: int
  data: str


def _parse_header(path: Path) -> Tuple[PcdHeader, int]:
  header_lines: List[str] = []
  with path.open("rb") as f:
    while True:
      line = f.readline()
      if not line:
        raise ValueError(f"Unexpected EOF while reading PCD header: {path}")
      decoded = line.decode("utf-8", errors="strict").strip()
      header_lines.append(decoded)
      if decoded.upper().startswith("DATA "):
        offset = f.tell()
        break

  kv: Dict[str, List[str]] = {}
  for line in header_lines:
    if not line or line.startswith("#"):
      continue
    parts = line.split()
    key = parts[0].upper()
    kv[key] = parts[1:]

  def _require(name: str) -> List[str]:
    if name not in kv:
      raise KeyError(f"PCD header missing `{name}`: {path}")
    return kv[name]

  fields = _require("FIELDS")
  size = [int(v) for v in _require("SIZE")]
  pcd_type = _require("TYPE")
  count = [int(v) for v in kv.get("COUNT", ["1"] * len(fields))]
  width = int(_require("WIDTH")[0])
  height = int(_require("HEIGHT")[0])
  points = int(kv.get("POINTS", [str(width * height)])[0])
  data = _require("DATA")[0].lower()

  header = PcdHeader(
      version=" ".join(kv.get("VERSION", ["0.7"])),
      fields=fields,
      size=size,
      type=pcd_type,
      count=count,
      width=width,
      height=height,
      points=points,
      data=data,
  )
  return header, offset


def _dtype_from_header(header: PcdHeader) -> np.dtype:
  fields = []
  for name, size, dtype_name, count in zip(
      header.fields, header.size, header.type, header.count
  ):
    if dtype_name == "F":
      scalar = {4: np.float32, 8: np.float64}.get(size)
    elif dtype_name == "U":
      scalar = {1: np.uint8, 2: np.uint16, 4: np.uint32}.get(size)
    elif dtype_name == "I":
      scalar = {1: np.int8, 2: np.int16, 4: np.int32}.get(size)
    else:
      scalar = None
    if scalar is None:
      raise ValueError(f"Unsupported PCD field type/size: type={dtype_name} size={size}")
    if count == 1:
      fields.append((name, scalar))
    else:
      fields.append((name, scalar, (count,)))
  return np.dtype(fields)


def _field_aliases(name: str) -> List[str]:
  lowered = name.lower()
  if lowered == "intensity":
    return ["intensity", "intensities", "i"]
  return [lowered]


def _find_field(header: PcdHeader, target: str) -> str:
  existing = {field.lower(): field for field in header.fields}
  for alias in _field_aliases(target):
    if alias in existing:
      return existing[alias]
  raise KeyError(f"PCD missing required field `{target}`. Found fields={header.fields}")


def _extract_xyzi(header: PcdHeader, data: np.ndarray) -> np.ndarray:
  x_name = _find_field(header, "x")
  y_name = _find_field(header, "y")
  z_name = _find_field(header, "z")
  i_name = _find_field(header, "intensity")
  x = np.asarray(data[x_name], dtype=np.float32).reshape(-1)
  y = np.asarray(data[y_name], dtype=np.float32).reshape(-1)
  z = np.asarray(data[z_name], dtype=np.float32).reshape(-1)
  intensity = np.asarray(data[i_name], dtype=np.float32).reshape(-1)
  return np.stack((x, y, z, intensity), axis=1).astype(np.float32, copy=False)


def load_pcd_xyzi(path: Path) -> np.ndarray:
  header, offset = _parse_header(path)
  if header.data not in {"ascii", "binary"}:
    raise ValueError(
        f"Only ascii/binary PCD are supported for now; got DATA {header.data} in {path}"
    )

  if header.data == "ascii":
    with path.open("rb") as f:
      f.seek(offset)
      text = f.read().decode("utf-8", errors="strict")
    rows = np.loadtxt(StringIO(text), dtype=np.float32)
    if rows.ndim == 1:
      rows = rows.reshape(1, -1)
    name_to_index = {name: idx for idx, name in enumerate(header.fields)}
    x = rows[:, name_to_index[_find_field(header, "x")]]
    y = rows[:, name_to_index[_find_field(header, "y")]]
    z = rows[:, name_to_index[_find_field(header, "z")]]
    intensity = rows[:, name_to_index[_find_field(header, "intensity")]]
    return np.stack((x, y, z, intensity), axis=1).astype(np.float32, copy=False)

  dtype = _dtype_from_header(header)
  with path.open("rb") as f:
    f.seek(offset)
    raw = f.read()
  data = np.frombuffer(raw, dtype=dtype, count=header.points)
  return _extract_xyzi(header, data)
