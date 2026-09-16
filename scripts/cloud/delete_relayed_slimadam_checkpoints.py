#!/usr/bin/env python3
"""Delete only SlimAdam checkpoints already verified on H200 and HF."""

import argparse
import shutil
from pathlib import Path

import torch


CHECKPOINT_ROOT = Path(
    "/workspace-SR006.nfs3/dimativator/exps/"
    "8xChinchilla_257M_fp8_states_cloud/"
    "8xChinchilla_257M_fp8_states/"
    "257m_slim_adam_fp8_states_8xC_cloud_4gpu/ckpts"
)
TARGETS = {
    "35325": 35325,
    "70650": 70650,
    "141300": 141300,
    "latest": 150000,
}
EXPECTED_FILES = ("main.pt", "worker_0.pt", "worker_1.pt", "worker_2.pt", "worker_3.pt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("Refusing to delete without --execute")

    root = CHECKPOINT_ROOT.resolve(strict=True)
    validated: list[Path] = []
    for name, expected_iteration in TARGETS.items():
        target = (root / name).resolve(strict=True)
        if target.parent != root:
            raise RuntimeError(f"Unsafe checkpoint target: {target}")
        missing = [file_name for file_name in EXPECTED_FILES if not (target / file_name).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete checkpoint {target}: missing {missing}")
        checkpoint = torch.load(
            str(target / "main.pt"),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        actual_iteration = int(checkpoint["itr"])
        if actual_iteration != expected_iteration:
            raise RuntimeError(
                f"Iteration mismatch for {target}: "
                f"{actual_iteration} != {expected_iteration}"
            )
        total_bytes = sum(
            path.stat().st_size for path in target.iterdir() if path.is_file()
        )
        print(
            f"DELETE_PREFLIGHT name={name} iteration={actual_iteration} "
            f"files={len(tuple(target.iterdir()))} bytes={total_bytes}",
            flush=True,
        )
        validated.append(target)

    for target in validated:
        shutil.rmtree(target)
        print(f"CHECKPOINT_DELETED path={target}", flush=True)

    remaining = [str(path) for path in validated if path.exists()]
    if remaining:
        raise RuntimeError(f"Checkpoint deletion incomplete: {remaining}")
    print(f"DELETE_COMPLETE root={root} count={len(validated)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
