#!/usr/bin/env bash
set -euo pipefail

readonly experiment_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_4gpu
readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud
readonly latest_dir="${experiment_dir}/ckpts/latest"

echo "CHECKPOINT_FILES"
find "${experiment_dir}/ckpts" -maxdepth 2 -type f \
    -printf '%T@ %s %p\n' 2>/dev/null | sort -n || true

if [[ -f "${latest_dir}/main.pt" ]]; then
    CHECKPOINT_PATH="${latest_dir}/main.pt" \
        /home/jovyan/hmoe-cloud/torch251-cu121/bin/python - <<'PY'
import os
from pathlib import Path

import torch

path = Path(os.environ["CHECKPOINT_PATH"])
checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
print(f"LATEST_CHECKPOINT_ITER={checkpoint.get('itr')}")
print(f"LATEST_CHECKPOINT_BYTES={path.stat().st_size}")
PY
else
    echo "LATEST_CHECKPOINT=missing"
fi

latest_log=$(find "${log_dir}" -maxdepth 1 -type f -name '*_full_*.log' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
echo "LATEST_LOG=${latest_log:-missing}"
if [[ -n "${latest_log}" ]]; then
    tr '\r' '\n' < "${latest_log}" | tail -n 250
fi
