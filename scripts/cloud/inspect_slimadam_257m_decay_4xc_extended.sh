#!/usr/bin/env bash
set -euo pipefail

readonly experiment=257m_slim_adam_fp8_states_4xC_decay_cloud_4gpu_extended
readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud
readonly checkpoint_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/${experiment}/ckpts/latest

df -h /workspace-SR006.nfs3
for rank in 0 1 2 3; do
    log_file="${log_dir}/${experiment}_rank${rank}.log"
    [[ -f "${log_file}" ]] || { echo "MISSING_LOG rank=${rank}"; exit 1; }
    echo "LOG rank=${rank} bytes=$(stat -c %s "${log_file}")"
    tr '\r' '\n' < "${log_file}" | grep -aE '>Eval:|Train: Iter=157000|DECAY_EXIT|DECAY_4XC_COMPLETE|Traceback|Error' | tail -n 12 || true
    grep -aFq "DECAY_4XC_COMPLETE rank=${rank}" "${log_file}"
    grep -aFq 'DECAY_EXIT scale=4 status=0' "${log_file}"
done

python - "${checkpoint_dir}" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
main_path = root / "main.pt"
main = torch.load(main_path, map_location="cpu", mmap=True, weights_only=False)
iteration = int(main["itr"])
print(f"FINAL_CHECKPOINT_ITERATION={iteration}")
print(f"FINAL_CHECKPOINT_BYTES={main_path.stat().st_size}")
if iteration != 157000:
    raise RuntimeError(f"expected iteration 157000, got {iteration}")
for rank in range(4):
    worker_path = root / f"worker_{rank}.pt"
    worker = torch.load(worker_path, map_location="cpu", weights_only=False)
    reader = worker["train_reader_state"]
    print(f"FINAL_WORKER rank={rank} bytes={worker_path.stat().st_size} reader_step={reader['step']}")
print("SLIMADAM_4XC_EXTENDED_VERIFIED")
PY
