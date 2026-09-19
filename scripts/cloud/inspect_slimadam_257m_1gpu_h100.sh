#!/usr/bin/env bash
set -euo pipefail

readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud
readonly results_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100

log_file=$(find "${log_dir}" -maxdepth 1 -type f \
    -name '257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100_full_*.log' \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
if [[ -z "${log_file}" ]]; then
    echo "FULL_LOG_MISSING"
    exit 0
fi

echo "LOG=${log_file}"
echo "LOG_SIZE=$(stat -c %s "${log_file}")"
grep -aE 'GPU_STATUS|NVIDIA H100|PARITY_CONFIG|Resuming Training|FINEWEB_VALIDATION|>Eval:|Train: Iter=|Traceback|Error|TRAIN_EXIT|RELAY_' \
    "${log_file}" | tail -n 40 || true

if [[ -d "${results_dir}/ckpts/latest" ]]; then
    find "${results_dir}/ckpts/latest" -maxdepth 1 -type f \
        -printf 'CHECKPOINT_FILE %f %s\n' | sort
    python - "${results_dir}/ckpts/latest" <<'PY'
import sys
from pathlib import Path

import torch

checkpoint_dir = Path(sys.argv[1])
main = torch.load(checkpoint_dir / "main.pt", map_location="cpu", mmap=True, weights_only=False)
worker = torch.load(checkpoint_dir / "worker_0.pt", map_location="cpu", weights_only=False)
initial_worker = torch.load(
    "/workspace-SR006.nfs3/dimativator/checkpoints/slimadam_257m_fp8_states_8xC/160000/worker_0.pt",
    map_location="cpu", weights_only=False,
)
iteration = int(main["itr"])
reader = worker["train_reader_state"]
initial_reader = initial_worker["train_reader_state"]
step = int(reader["step"])
initial_step = int(initial_reader["step"])
print(f"CHECKPOINT_ITERATION={iteration}")
print(f"WORKER_READER_STEP={step}")
print(f"INITIAL_WORKER_READER_STEP={initial_step}")
print(f"WORKER_READER_TYPE={reader['reader_type']}")
expected_step = initial_step + (iteration - 160000) * 4
print(f"EXPECTED_WORKER_READER_STEP={expected_step}")
if step != expected_step:
    raise RuntimeError(f"reader step {step} does not match expected {expected_step}")
PY
fi
