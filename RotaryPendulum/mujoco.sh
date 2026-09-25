#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != -p ]]; then
  echo "usage: ./mujoco.sh -p script.py [args...]" >&2
  exit 2
fi
shift
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT/source/rotarypendulum${PYTHONPATH:+:$PYTHONPATH}"
exec /home/cuong/miniconda3/envs/mujoco314/bin/python "$@"
