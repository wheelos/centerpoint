#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CALLER_CWD="$(pwd)"

to_abs_path() {
  "$PYTHON_BIN" - "$1" "$CALLER_CWD" <<'PY'
from pathlib import Path
import sys

raw = Path(sys.argv[1])
base = Path(sys.argv[2])
if raw.is_absolute():
  print(raw)
else:
  print((base / raw).resolve())
PY
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/prepare_nuscenes_data.sh --root-path <nuscenes_dir> --out-dir <out_dir> [extra args]

Description:
  Locate the installed MMDetection3D package from the current Python
  environment and invoke its official `tools/create_data.py` for nuScenes.

Examples:
  bash tools/prepare_nuscenes_data.sh \
    --root-path data/nuscenes \
    --out-dir data/nuscenes \
    --extra-tag nuscenes

  PYTHON_BIN=.venv/bin/python bash tools/prepare_nuscenes_data.sh \
    --root-path data/nuscenes \
    --out-dir data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ARGS=()
ROOT_PATH_ABS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-path|--out-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      abs_value="$(to_abs_path "$2")"
      if [[ "$1" == "--root-path" ]]; then
        ROOT_PATH_ABS="$abs_value"
      fi
      ARGS+=("$1" "$abs_value")
      shift 2
      ;;
    --root-path=*|--out-dir=*)
      key="${1%%=*}"
      value="${1#*=}"
      abs_value="$(to_abs_path "$value")"
      if [[ "$key" == "--root-path" ]]; then
        ROOT_PATH_ABS="$abs_value"
      fi
      ARGS+=("${key}=$abs_value")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$ROOT_PATH_ABS" ]]; then
  echo "Missing required argument: --root-path" >&2
  exit 2
fi

mapfile -t FOUND_PATHS < <("$PYTHON_BIN" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("mmdet3d")
if spec is None or spec.origin is None:
  raise SystemExit(
      "Cannot import `mmdet3d` from the current Python environment. "
      "Activate your venv and install MMDetection3D first."
  )

pkg_dir = Path(spec.origin).resolve().parent
repo_root = pkg_dir.parent
candidates = [
    (pkg_dir / ".mim" / "tools" / "create_data.py", pkg_dir / ".mim"),
    (repo_root / "tools" / "create_data.py", repo_root),
    (repo_root / ".mim" / "tools" / "create_data.py", repo_root / ".mim"),
]

for script_path, launch_root in candidates:
  if script_path.exists():
    print(script_path)
    print(launch_root)
    raise SystemExit(0)

raise SystemExit(
    "Found `mmdet3d`, but could not locate its `create_data.py`. "
    f"Tried: {[str(p[0]) for p in candidates]}"
)
PY
)

CREATE_DATA_SCRIPT="${FOUND_PATHS[0]}"
CREATE_DATA_ROOT="${FOUND_PATHS[1]}"

export PYTHONPATH="$CREATE_DATA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LAUNCH_DIR="$(mktemp -d)"
trap 'rm -rf "$LAUNCH_DIR"' EXIT
mkdir -p "$LAUNCH_DIR/data"
ln -s "$ROOT_PATH_ABS" "$LAUNCH_DIR/data/nuscenes"
cd "$LAUNCH_DIR"

exec "$PYTHON_BIN" "$CREATE_DATA_SCRIPT" nuscenes "${ARGS[@]}"
