#!/usr/bin/env python3
"""Remove only the four user-approved Stage3 BF16 pre-decay checkpoints."""

import os
import shutil
from pathlib import Path


ROOT = Path("/home/jovyan/dimativator/stage3_257m_bf16_native_20260921")
GROUP = "8xChinchilla_257M_bf16_native_states"
TARGETS = (
    (
        "257m_slim_adam_bf16_native_states_8xC_cloud_4gpu_rerun_20260923",
        35325,
        "257m_slim_adam_bf16_native_states_1xC_decay_cloud_1gpu.log",
        "DECAY_COMPLETE optimizer=slim_adam scale=1 end=39250",
    ),
    (
        "257m_slim_adam_bf16_native_states_8xC_cloud_4gpu_rerun_20260923",
        70650,
        "257m_slim_adam_bf16_native_states_2xC_decay_cloud_1gpu.log",
        "DECAY_COMPLETE optimizer=slim_adam scale=2 end=78500",
    ),
    (
        "257m_slim_adam_bf16_native_states_8xC_cloud_4gpu_rerun_20260923",
        141300,
        "257m_slim_adam_bf16_native_states_4xC_decay_cloud_1gpu.log",
        "DECAY_COMPLETE optimizer=slim_adam scale=4 end=157000",
    ),
    (
        "257m_frugal_bf16_native_states_8xC_cloud_4gpu",
        141300,
        "257m_frugal_bf16_native_states_4xC_decay_cloud_1gpu_retry_20260923.log",
        "DECAY_COMPLETE optimizer=frugal scale=4 end=157000",
    ),
)


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    delete = os.environ.get("DELETE_CONFIRMED") == "YES_REMOVE_FOUR_DECAY_SOURCES"
    checked = []
    for experiment, iteration, log_name, marker in TARGETS:
        path = ROOT / GROUP / experiment / "ckpts" / str(iteration)
        log_path = ROOT / "logs" / log_name
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"Missing or unsafe checkpoint directory: {path}")
        if path.parent.name != "ckpts" or path.parent.parent.name != experiment:
            raise RuntimeError(f"Unexpected checkpoint path: {path}")
        if not path.resolve().is_relative_to((ROOT / GROUP).resolve()):
            raise RuntimeError(f"Checkpoint escapes expected root: {path}")
        if (path / "main.pt").stat().st_size < 1_000_000_000:
            raise RuntimeError(f"Unexpected main checkpoint size: {path}")
        for rank in range(4):
            if not (path / f"worker_{rank}.pt").is_file():
                raise RuntimeError(f"Missing worker checkpoint: {path} rank={rank}")
        if not log_path.is_file() or marker not in log_path.read_text(errors="replace"):
            raise RuntimeError(f"Missing decay completion marker: {log_path}")
        checked.append((path, size_bytes(path)))

    for path, size in checked:
        print(f"VERIFIED bytes={size} path={path}", flush=True)
    print(f"TOTAL_BYTES={sum(size for _, size in checked)}", flush=True)
    if not delete:
        print("DRY_RUN_COMPLETE", flush=True)
        return

    for path, size in checked:
        shutil.rmtree(path)
        if path.exists():
            raise RuntimeError(f"Checkpoint still exists after deletion: {path}")
        print(f"REMOVED bytes={size} path={path}", flush=True)
    print("FOUR_DECAY_SOURCES_REMOVED", flush=True)


if __name__ == "__main__":
    main()
