#!/usr/bin/env bash
set -euo pipefail

readonly results_dir=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly group=8xChinchilla_257M_bf16_native_states
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - "${results_dir}/${group}" <<'PY'
from pathlib import Path
import sys

import torch

group_root = Path(sys.argv[1])
for label in ("frugal", "slim_adam"):
    experiment = f"257m_{label}_bf16_native_states_8xC_cloud_4gpu"
    checkpoint_root = group_root / experiment / "ckpts"
    for name in ("35325", "70650", "141300", "latest"):
        root = checkpoint_root / name
        main_path = root / "main.pt"
        if not main_path.exists():
            print(f"CHECKPOINT_MISSING optimizer={label} name={name} file=main.pt", flush=True)
            continue
        try:
            main = torch.load(main_path, map_location="cpu", weights_only=False)
            itr = int(main["itr"])
            del main
            workers = []
            for rank in range(4):
                worker_path = root / f"worker_{rank}.pt"
                worker = torch.load(worker_path, map_location="cpu", weights_only=False)
                reader_state = worker.get("train_reader_state", {})
                workers.append(reader_state.get("step", reader_state.get("global_step")))
                del worker
            print(
                f"CHECKPOINT_STATE optimizer={label} name={name} "
                f"itr={itr} worker_reader_steps={workers} "
                f"reader_consistent={len(set(workers)) == 1 and workers[0] == itr * 4}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"CHECKPOINT_ERROR optimizer={label} name={name} "
                f"type={type(exc).__name__} detail={exc}",
                flush=True,
            )
PY

df -h /home/jovyan /workspace-SR006.nfs3
