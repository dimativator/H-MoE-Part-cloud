#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.cloud.prepare_fineweb_h200_packed import (  # noqa: E402
    BATCH_SIZE,
    BLOCK_TOKENS,
    BLOCKS_PER_RANK,
    CHECKPOINT_EVERY_BLOCKS,
    EXPECTED_MANIFEST_FINGERPRINT,
    EXPECTED_SPLIT_PLAN_FINGERPRINT,
    ITERATIONS,
    MICROSTEPS_PER_ITERATION,
    PREVIEW_BLOCKS,
    WORLD_SIZE,
    build_snapshot,
    load_manifest,
    make_stream,
    write_json_atomic,
)

EXPECTED_CONTINUATION_PREVIEW_SHA256 = {
    0: "00ea50b3e5bff4225d8f502fe81bc5f3668ef2062a19e1f3bc3344207c7bfd59",
    1: "dc0b936666849907e1bef994b98e9f20b021dcd766775e910bc4ab63cfb8d545",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_state(path: Path, rank: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        state = json.load(source)
    if state.get("reader_type") != "fineweb_train_reader_v1":
        raise ValueError("Unexpected H200 reader state type")
    if int(state["batch_size"]) != BATCH_SIZE:
        raise ValueError("Unexpected H200 source batch size")
    if int(state["sequence_length"]) != BLOCK_TOKENS - 1:
        raise ValueError("Unexpected H200 sequence length")
    if int(state["step"]) != ITERATIONS * MICROSTEPS_PER_ITERATION:
        raise ValueError("H200 reader state is not at the 1xC boundary")
    stream_state = state["stream_state"]
    if int(stream_state["rank"]) != rank or int(stream_state["world_size"]) != WORLD_SIZE:
        raise ValueError("H200 stream rank/world_size mismatch")
    if stream_state["manifest_fingerprint"] != EXPECTED_MANIFEST_FINGERPRINT:
        raise ValueError("H200 stream manifest mismatch")
    if stream_state["split_plan_fingerprint"] != EXPECTED_SPLIT_PLAN_FINGERPRINT:
        raise ValueError("H200 stream split-plan mismatch")
    return stream_state


def maybe_finalize(destination: Path) -> None:
    continuation_paths = [
        destination / f"train_rank{rank}.continuation.json" for rank in range(WORLD_SIZE)
    ]
    if not all(path.is_file() for path in continuation_paths):
        return
    metadata_path = destination / "packed_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") == "packed_fineweb_h200_v2":
        print("PACKED_EXTENSION=already_complete", flush=True)
        return
    if metadata.get("format") != "packed_fineweb_h200_v1":
        raise ValueError("The base packed snapshot is not v1")
    if int(metadata["iterations"]) != ITERATIONS:
        raise ValueError("The base packed snapshot is not exactly 1xC")

    continuations = [json.loads(path.read_text()) for path in continuation_paths]
    for rank, rank_metadata in enumerate(metadata["ranks"]):
        continuation = continuations[rank]
        if int(rank_metadata["rank"]) != rank or int(continuation["rank"]) != rank:
            raise ValueError("Rank metadata is out of order")
        original = {
            key: rank_metadata[key]
            for key in ("file", "blocks", "bytes", "sha256")
        }
        rank_metadata["segments"] = [original, continuation]
        rank_metadata["blocks"] = int(original["blocks"]) + int(continuation["blocks"])
        rank_metadata["bytes"] = int(original["bytes"]) + int(continuation["bytes"])
        rank_metadata.pop("file", None)
        rank_metadata.pop("sha256", None)
    metadata["format"] = "packed_fineweb_h200_v2"
    metadata["iterations"] = ITERATIONS * 2
    metadata["blocks_per_rank"] = BLOCKS_PER_RANK * 2
    metadata["continuation_source"] = "H200 fineweb_replay checkpoint at iteration 75457"
    write_json_atomic(metadata_path, metadata)
    print("PACKED_EXTENSION=complete", flush=True)


def extend_rank(
    destination: Path,
    manifest,
    split_plan,
    *,
    rank: int,
    initial_state: dict[str, Any],
) -> None:
    final_path = destination / f"train_rank{rank}.continuation.uint16"
    metadata_path = destination / f"train_rank{rank}.continuation.json"
    if final_path.is_file() and metadata_path.is_file():
        print(f"RANK={rank} continuation already complete", flush=True)
        maybe_finalize(destination)
        return

    part_path = final_path.with_suffix(final_path.suffix + ".part")
    state_path = destination / f"train_rank{rank}.continuation.state.json"
    stream = make_stream(manifest, split_plan, rank)
    blocks_written = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        blocks_written = int(state["blocks_written"])
        stream.load_state_dict(state["stream_state"])
        expected_bytes = blocks_written * BLOCK_TOKENS * 2
        if not part_path.is_file() or part_path.stat().st_size < expected_bytes:
            raise RuntimeError("Continuation partial data is inconsistent with resume state")
        with part_path.open("r+b") as output:
            output.truncate(expected_bytes)
        print(f"RANK={rank} resuming at block {blocks_written}", flush=True)
    else:
        stream.load_state_dict(initial_state)
        part_path.write_bytes(b"")

    try:
        with part_path.open("ab") as output:
            while blocks_written < BLOCKS_PER_RANK:
                count = min(CHECKPOINT_EVERY_BLOCKS, BLOCKS_PER_RANK - blocks_written)
                blocks = [next(stream) for _ in range(count)]
                if max(max(block) for block in blocks) >= 65536:
                    raise ValueError("GPT-2 token id does not fit uint16")
                payload = np.asarray(blocks, dtype="<u2").tobytes()
                if blocks_written == 0:
                    preview_bytes = PREVIEW_BLOCKS * BLOCK_TOKENS * 2
                    preview_sha256 = hashlib.sha256(payload[:preview_bytes]).hexdigest()
                    if preview_sha256 != EXPECTED_CONTINUATION_PREVIEW_SHA256[rank]:
                        raise ValueError(
                            f"Rank {rank} continuation preview does not match H200"
                        )
                output.write(payload)
                output.flush()
                blocks_written += count
                write_json_atomic(
                    state_path,
                    {
                        "blocks_written": blocks_written,
                        "stream_state": stream.state_dict(),
                    },
                )
                print(f"RANK={rank} blocks={blocks_written}/{BLOCKS_PER_RANK}", flush=True)
    finally:
        stream.close()

    expected_bytes = BLOCKS_PER_RANK * BLOCK_TOKENS * 2
    if part_path.stat().st_size != expected_bytes:
        raise RuntimeError("Unexpected packed continuation size")
    os.replace(part_path, final_path)
    continuation = {
        "rank": rank,
        "file": final_path.name,
        "blocks": BLOCKS_PER_RANK,
        "bytes": expected_bytes,
        "sha256": sha256_file(final_path),
        "preview_blocks": PREVIEW_BLOCKS,
        "preview_sha256": EXPECTED_CONTINUATION_PREVIEW_SHA256[rank],
    }
    write_json_atomic(metadata_path, continuation)
    state_path.unlink(missing_ok=True)
    print(f"RANK={rank} sha256={continuation['sha256']}", flush=True)
    maybe_finalize(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(WORLD_SIZE), required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, dataset_root=args.dataset_root)
    snapshot = build_snapshot(manifest)
    initial_state = load_initial_state(args.initial_state, args.rank)
    extend_rank(
        args.destination,
        manifest,
        snapshot.plan,
        rank=args.rank,
        initial_state=initial_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
