#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
df -h /home/jovyan /workspace-SR006.nfs3
python scripts/cloud/preflight_figure6a_decay.py
