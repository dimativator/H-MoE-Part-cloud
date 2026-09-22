#!/usr/bin/env python3
"""Decide whether a FineWeb checkpoint reader state can be restored."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

LIVE_READER_TYPE = "fineweb_live_sharded_train_reader_v1"


def contains_live_reader_state(value: object) -> bool:
    if isinstance(value, Mapping):
        if value.get("reader_type") == LIVE_READER_TYPE:
            return True
        return any(contains_live_reader_state(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_live_reader_state(item) for item in value)
    return False


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("worker_checkpoint", type=Path)
    args = parser.parse_args()

    worker_checkpoint = args.worker_checkpoint.resolve(strict=True)
    state = torch.load(worker_checkpoint, map_location="cpu", weights_only=False)
    reader_state = state.get("train_reader_state")
    mode = "load" if contains_live_reader_state(reader_state) else "skip"
    print(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
