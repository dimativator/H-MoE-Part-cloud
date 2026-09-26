#!/usr/bin/env bash
set -euo pipefail

readonly checkpoint_root=/home/jovyan/dimativator/stage3_257m_scale_20260925/4xChinchilla_257M_bf16_scale/257m_scale_bf16_4xC_cloud_4gpu/ckpts

python - "${checkpoint_root}" <<'PY'
from pathlib import Path
import sys

import torch

checkpoint_root = Path(sys.argv[1])
for iteration in (35325, 70650):
    root = checkpoint_root / str(iteration)
    try:
        main_path = root / "main.pt"
        main = torch.load(main_path, map_location="cpu", weights_only=False)
        main_iteration = int(main["itr"])
        del main
        reader_steps = []
        worker_sizes = []
        for rank in range(4):
            worker_path = root / f"worker_{rank}.pt"
            worker = torch.load(worker_path, map_location="cpu", weights_only=False)
            reader = worker["train_reader_state"]
            reader_steps.append(int(reader.get("step", reader.get("global_step"))))
            worker_sizes.append(worker_path.stat().st_size)
            del worker
        consistent = main_iteration == iteration and reader_steps == [iteration * 4] * 4
        print(
            f"SCALE_MILESTONE_AUDIT iteration={iteration} main_iteration={main_iteration} "
            f"main_bytes={main_path.stat().st_size} worker_bytes={worker_sizes} "
            f"reader_steps={reader_steps} consistent={consistent}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"SCALE_MILESTONE_AUDIT_ERROR iteration={iteration} "
            f"type={type(exc).__name__} detail={exc}",
            flush=True,
        )
PY

df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
