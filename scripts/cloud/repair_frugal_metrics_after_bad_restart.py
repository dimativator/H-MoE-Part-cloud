#!/usr/bin/env python3
"""Remove only the verified low-iteration tail from the failed 2026-09-16 restart."""

from __future__ import annotations

import json
import os
from pathlib import Path


RESULTS_ROOT = Path(
    "/workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-20260912"
)
GROUP = "2xChinchilla_500M_frugal_muon_2gpu_cloud"
EXPERIMENTS = (
    "llama500M_frugal_muon_adamw_bf16_2xC_2gpu",
    "llama500M_frugal_muon_adamw_fp8_full_2xC_2gpu",
)
BAD_RESTART_TIMESTAMP = 1789553000.0
MAX_BAD_RESTART_ITERATION = 1000


def repair(path: Path) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    split_at: int | None = None
    for index, line in enumerate(lines):
        event = json.loads(line)
        if float(event.get("timestamp", 0.0)) >= BAD_RESTART_TIMESTAMP:
            split_at = index
            break
    if split_at is None:
        print(f"NO_BAD_TAIL={path}", flush=True)
        return

    removed = lines[split_at:]
    for line in removed:
        event = json.loads(line)
        timestamp = event.get("timestamp")
        if timestamp is not None and float(timestamp) < BAD_RESTART_TIMESTAMP:
            raise RuntimeError(f"tail contains an older event: {event}")
        iteration = event.get("iter")
        if iteration is not None and int(iteration) > MAX_BAD_RESTART_ITERATION:
            raise RuntimeError(f"refusing to remove high-iteration event: {event}")

    backup = path.with_name("metrics.bad-restart-20260916.jsonl")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    backup.write_bytes(b"".join(removed))
    temporary = path.with_name(f".{path.name}.repair-incoming")
    with temporary.open("wb") as handle:
        handle.write(b"".join(lines[:split_at]))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    print(
        f"REPAIRED={path} kept_lines={split_at} removed_lines={len(removed)} "
        f"backup={backup}",
        flush=True,
    )


def main() -> int:
    for experiment in EXPERIMENTS:
        repair(RESULTS_ROOT / GROUP / experiment / "metrics.jsonl")
    print("FRUGAL_METRICS_REPAIR=ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
