#!/usr/bin/env python3
"""Sync only the Stage3 BF16/native offline runs to the intended W&B project."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/home/jovyan/dimativator/stage3_257m_bf16_native_20260921/wandb_offline")
BASE_URL = "https://wandb-radfan.ru"
ENTITY = "andrey"
PROJECT = "fp8-pretrain"


def main() -> None:
    if not ROOT.is_dir():
        raise RuntimeError(f"Missing offline directory: {ROOT}")
    run_dirs = sorted({path.parent for path in ROOT.rglob("run-*.wandb")})
    if not run_dirs:
        raise RuntimeError(f"No offline runs found under {ROOT}")
    for run_dir in run_dirs:
        if ROOT not in run_dir.parents or not run_dir.name.startswith("offline-run-"):
            raise RuntimeError(f"Unexpected run directory: {run_dir}")
    print(f"SYNC_TARGET={BASE_URL}/{ENTITY}/{PROJECT} RUN_COUNT={len(run_dirs)}", flush=True)
    env = os.environ.copy()
    env.update({
        "WANDB_BASE_URL": BASE_URL,
        "WANDB_ENTITY": ENTITY,
        "WANDB_PROJECT": PROJECT,
    })
    subprocess.run(
        [
            "wandb", "sync", "--include-offline", "--no-include-online",
            "--no-sync-tensorboard", "--entity", ENTITY, "--project", PROJECT,
            *(str(run_dir) for run_dir in run_dirs),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print("STAGE3_BF16_WANDB_SYNC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
