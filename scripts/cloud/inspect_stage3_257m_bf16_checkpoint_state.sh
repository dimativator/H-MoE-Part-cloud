#!/usr/bin/env bash
set -euo pipefail

readonly results_dir=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly group=8xChinchilla_257M_bf16_native_states

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
            continue
        main = torch.load(main_path, map_location="cpu", weights_only=False)
        workers = []
        for rank in range(4):
            worker_path = root / f"worker_{rank}.pt"
            worker = torch.load(worker_path, map_location="cpu", weights_only=False)
            reader_state = worker.get("train_reader_state", {})
            workers.append(reader_state.get("step", reader_state.get("global_step")))
        print(
            f"CHECKPOINT_STATE optimizer={label} name={name} "
            f"itr={int(main['itr'])} worker_reader_steps={workers}"
        )
PY

df -h /home/jovyan /workspace-SR006.nfs3
