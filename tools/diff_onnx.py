#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _require_onnx() -> Any:
  try:
    import onnx  # type: ignore
    from onnx import numpy_helper  # type: ignore

    return onnx, numpy_helper
  except Exception as exc:
    raise SystemExit(
        "Missing dependency: `onnx`.\n"
        "Install it in the same python env you're running this script with:\n"
        "  python3 -m pip install onnx\n"
    ) from exc


@dataclass(frozen=True)
class ConvInfo:
  idx: int
  name: str
  weight: str
  wshape: Tuple[int, ...]
  strides: Tuple[int, ...]
  pads: Tuple[int, ...]
  dilations: Tuple[int, ...]
  groups: int


def _get_attr_ints(node: Any, key: str, default: Sequence[int]) -> Tuple[int, ...]:
  for a in node.attribute:
    if a.name == key:
      if a.ints:
        return tuple(int(x) for x in a.ints)
      if a.i:
        return (int(a.i),)
  return tuple(int(x) for x in default)


def _get_attr_int(node: Any, key: str, default: int) -> int:
  for a in node.attribute:
    if a.name == key:
      return int(a.i)
  return int(default)


def _tensor_shape(t) -> Tuple[int, ...]:
  dims = []
  for d in t.dims:
    dims.append(int(d))
  return tuple(dims)


def _build_initializer_map(graph: Any) -> Dict[str, Any]:
  return {init.name: init for init in graph.initializer}


def _collect_conv_infos(model: Any) -> List[ConvInfo]:
  graph = model.graph
  inits = _build_initializer_map(graph)

  convs: List[ConvInfo] = []
  for idx, node in enumerate(graph.node):
    if node.op_type != "Conv":
      continue
    wname = node.input[1] if len(node.input) > 1 else ""
    winit = inits.get(wname)
    wshape = _tensor_shape(winit) if winit is not None else ()
    convs.append(
        ConvInfo(
            idx=idx,
            name=node.name or f"Conv_{idx}",
            weight=wname,
            wshape=wshape,
            strides=_get_attr_ints(node, "strides", default=(1, 1)),
            pads=_get_attr_ints(node, "pads", default=(0, 0, 0, 0)),
            dilations=_get_attr_ints(node, "dilations", default=(1, 1)),
            groups=_get_attr_int(node, "group", default=1),
        ))
  return convs


def _io_shapes(model: Any) -> Tuple[List[Tuple[str, Tuple[int, ...]]], List[Tuple[str, Tuple[int, ...]]]]:
  def _shape(v) -> Tuple[int, ...]:
    try:
      tt = v.type.tensor_type
      return tuple(int(d.dim_value) for d in tt.shape.dim)
    except Exception:
      return ()

  ins = [(v.name, _shape(v)) for v in model.graph.input]
  outs = [(v.name, _shape(v)) for v in model.graph.output]
  return ins, outs


def _node_op_counts(model: Any) -> Counter:
  return Counter(n.op_type for n in model.graph.node)


def _print_summary(tag: str, model: Any) -> None:
  ins, outs = _io_shapes(model)
  print(f"\n== {tag} ==")
  print(f"inputs:  {[f'{n}:{s}' for n, s in ins]}")
  print(f"outputs: {[f'{n}:{s}' for n, s in outs]}")
  print(f"nodes: {len(model.graph.node)}  initializers: {len(model.graph.initializer)}")
  ops = _node_op_counts(model)
  top_ops = ", ".join(f"{k}={v}" for k, v in ops.most_common(12))
  print(f"top ops: {top_ops}")

  convs = _collect_conv_infos(model)
  if convs:
    c0 = convs[0]
    print(
        f"first Conv: idx={c0.idx} w={c0.wshape} strides={c0.strides} pads={c0.pads} groups={c0.groups}"
    )
  else:
    print("no Conv nodes found")

  init_names = [i.name for i in model.graph.initializer]
  task_ids = sorted({
      n.split("pts_bbox_head.task_heads.")[1].split(".")[0]
      for n in init_names
      if "pts_bbox_head.task_heads." in n
  })
  if task_ids:
    print(f"task_heads: {len(task_ids)} ids={task_ids}")


def _diff_conv_prefix(a: List[ConvInfo], b: List[ConvInfo], limit: int) -> None:
  n = min(len(a), len(b), limit)
  for i in range(n):
    if a[i].wshape != b[i].wshape or a[i].strides != b[i].strides or a[i].pads != b[i].pads or a[i].groups != b[i].groups:
      print(f"\nfirst conv diff at #{i}:")
      print(f"  A: idx={a[i].idx} w={a[i].wshape} strides={a[i].strides} pads={a[i].pads} groups={a[i].groups}")
      print(f"  B: idx={b[i].idx} w={b[i].wshape} strides={b[i].strides} pads={b[i].pads} groups={b[i].groups}")
      return
  print(f"\nconv prefix match for first {n} Conv nodes")


def main() -> None:
  onnx, _ = _require_onnx()

  parser = argparse.ArgumentParser(description="Compare two ONNX graphs (Apollo vs exported).")
  parser.add_argument("--a", required=True, help="ONNX path A (e.g. Apollo shipped cpdet_backbone.onnx)")
  parser.add_argument("--b", required=True, help="ONNX path B (e.g. your exported cpdet_backbone.onnx)")
  parser.add_argument("--conv-limit", type=int, default=20, help="How many Conv nodes to compare/print")
  args = parser.parse_args()

  a_path = Path(args.a)
  b_path = Path(args.b)
  a = onnx.load(str(a_path))
  b = onnx.load(str(b_path))

  _print_summary(f"A {a_path}", a)
  _print_summary(f"B {b_path}", b)

  a_convs = _collect_conv_infos(a)
  b_convs = _collect_conv_infos(b)

  print(f"\nconv count: A={len(a_convs)} B={len(b_convs)}")
  if a_convs and b_convs:
    _diff_conv_prefix(a_convs, b_convs, limit=args.conv_limit)
    print(f"\nFirst {min(args.conv_limit, len(a_convs))} Convs in A:")
    for c in a_convs[: args.conv_limit]:
      print(f"  #{a_convs.index(c)} idx={c.idx} w={c.wshape} strides={c.strides} pads={c.pads} groups={c.groups}")
    print(f"\nFirst {min(args.conv_limit, len(b_convs))} Convs in B:")
    for c in b_convs[: args.conv_limit]:
      print(f"  #{b_convs.index(c)} idx={c.idx} w={c.wshape} strides={c.strides} pads={c.pads} groups={c.groups}")

  # Op diff (counts only)
  a_ops = _node_op_counts(a)
  b_ops = _node_op_counts(b)
  all_ops = sorted(set(a_ops.keys()) | set(b_ops.keys()))
  print("\n== op count diff (A - B) ==")
  for op in all_ops:
    da = a_ops.get(op, 0)
    db = b_ops.get(op, 0)
    if da != db:
      print(f"  {op}: {da} vs {db}")


if __name__ == "__main__":
  main()
