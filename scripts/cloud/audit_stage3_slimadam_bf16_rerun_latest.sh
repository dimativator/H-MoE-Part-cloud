#!/usr/bin/env bash
set -euo pipefail

readonly root=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly experiment=257m_slim_adam_bf16_native_states_8xC_cloud_4gpu_rerun_20260923
readonly latest="${root}/8xChinchilla_257M_bf16_native_states/${experiment}/ckpts/latest"
readonly log="${root}/logs/${experiment}_full_rank0.log"

echo "AUDIT_TIME=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs3
find "${latest}" -maxdepth 1 -type f -printf '%f %s bytes\n' 2>/dev/null | sort || true

python - "${latest}" <<'PY'
from pathlib import Path
import sys

import torch

root = Path(sys.argv[1])
main_path = root / "main.pt"
try:
    main = torch.load(main_path, map_location="cpu", mmap=True, weights_only=False)
    itr = int(main["itr"])
    del main
    print(f"LATEST_MAIN_OK itr={itr} bytes={main_path.stat().st_size}", flush=True)
except Exception as exc:
    print(f"LATEST_MAIN_ERROR type={type(exc).__name__} detail={exc}", flush=True)
    raise SystemExit(2)

steps = []
for rank in range(4):
    path = root / f"worker_{rank}.pt"
    try:
        worker = torch.load(path, map_location="cpu", weights_only=False)
        state = worker.get("train_reader_state", {})
        step = state.get("step", state.get("global_step"))
        steps.append(step)
        print(f"LATEST_WORKER_OK rank={rank} step={step} bytes={path.stat().st_size}", flush=True)
    except Exception as exc:
        print(f"LATEST_WORKER_ERROR rank={rank} type={type(exc).__name__} detail={exc}", flush=True)
        raise SystemExit(3)

consistent = len(set(steps)) == 1 and steps[0] == itr * 4
print(f"LATEST_READER_CONSISTENT={str(consistent).lower()} expected={itr * 4}", flush=True)
if not consistent:
    raise SystemExit(4)
PY

if [[ -f "${log}" ]]; then
    echo '=== FULL LOG TAIL ==='
    tail -c 12000 "${log}" | tr '\r' '\n' | tail -n 90
fi
