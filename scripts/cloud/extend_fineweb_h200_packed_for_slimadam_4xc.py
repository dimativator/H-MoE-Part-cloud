#!/usr/bin/env python3
"""Append deterministic FineWeb blocks without changing the existing packed prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.cloud.extend_fineweb_h200_packed import (  # noqa: E402
    load_initial_state,
    sha256_file,
)
from scripts.cloud.prepare_fineweb_h200_packed import (  # noqa: E402
    BATCH_SIZE,
    BLOCK_TOKENS,
    BLOCKS_PER_RANK,
    CHECKPOINT_EVERY_BLOCKS,
    EXPECTED_MANIFEST_FINGERPRINT,
    EXPECTED_SPLIT_PLAN_FINGERPRINT,
    ITERATIONS,
    MICROSTEPS_PER_ITERATION,
    WORLD_SIZE,
    build_snapshot,
    load_manifest,
    make_stream,
    write_json_atomic,
)

TARGET_ITERATIONS = 157_100
TARGET_BLOCKS = TARGET_ITERATIONS * MICROSTEPS_PER_ITERATION * BATCH_SIZE
EXTRA_BLOCKS = TARGET_BLOCKS - 2 * BLOCKS_PER_RANK
BLOCK_BYTES = BLOCK_TOKENS * 2


def load_v2_metadata(destination: Path) -> dict[str, Any]:
    metadata = json.loads((destination / "packed_metadata.json").read_text())
    if metadata["format"] != "packed_fineweb_h200_v2":
        raise ValueError("Expected unmodified v2 packed FineWeb metadata")
    if metadata["manifest_fingerprint"] != EXPECTED_MANIFEST_FINGERPRINT:
        raise ValueError("Packed FineWeb manifest fingerprint changed")
    if metadata["split_plan_fingerprint"] != EXPECTED_SPLIT_PLAN_FINGERPRINT:
        raise ValueError("Packed FineWeb split fingerprint changed")
    if int(metadata["world_size"]) != WORLD_SIZE:
        raise ValueError("Packed FineWeb source world size changed")
    if int(metadata["batch_size"]) != BATCH_SIZE:
        raise ValueError("Packed FineWeb source batch size changed")
    if int(metadata["blocks_per_rank"]) != 2 * BLOCKS_PER_RANK:
        raise ValueError("Packed FineWeb v2 block count changed")
    return metadata


def continuation_segment(destination: Path, rank_metadata: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    segments = rank_metadata.get("segments", [])
    if len(segments) != 2:
        raise ValueError("Expected exactly two existing packed segments")
    for segment in segments:
        path = destination / segment["file"]
        if path.stat().st_size != int(segment["bytes"]):
            raise ValueError(f"Packed segment size changed: {path}")
    second = segments[1]
    if int(second["blocks"]) != BLOCKS_PER_RANK:
        raise ValueError("Existing continuation length changed")
    path = destination / second["file"]
    if sha256_file(path) != second["sha256"]:
        raise ValueError(f"Existing continuation SHA-256 changed: {path}")
    return path, second


def blocks_payload(stream, count: int) -> bytes:
    blocks = [next(stream) for _ in range(count)]
    if max(max(block) for block in blocks) >= 65536:
        raise ValueError("GPT-2 token id does not fit uint16")
    return np.asarray(blocks, dtype="<u2").tobytes()


def restore_end_of_v2(
    destination: Path,
    rank: int,
    stream,
    continuation_path: Path,
    continuation_sha256: str,
) -> None:
    seed_path = destination / f"train_rank{rank}.third.seed.json"
    if seed_path.is_file():
        seed = json.loads(seed_path.read_text())
        if seed["continuation_sha256"] != continuation_sha256:
            raise ValueError("Saved third-segment seed has a different source hash")
        if int(seed["blocks_verified"]) != BLOCKS_PER_RANK:
            raise ValueError("Saved third-segment seed has wrong length")
        stream.load_state_dict(seed["stream_state"])
        print(f"RANK={rank} FASTFORWARD=seed_verified", flush=True)
        return

    state_path = destination / f"train_rank{rank}.third.fastforward.state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state["continuation_sha256"] != continuation_sha256:
            raise ValueError("Fast-forward state has a different source hash")
        verified = int(state["blocks_verified"])
        if verified < 0 or verified > BLOCKS_PER_RANK:
            raise ValueError("Fast-forward progress out of range")
        stream.load_state_dict(state["stream_state"])
    else:
        verified = 0

    with continuation_path.open("rb") as packed:
        packed.seek(verified * BLOCK_BYTES)
        while verified < BLOCKS_PER_RANK:
            count = min(CHECKPOINT_EVERY_BLOCKS, BLOCKS_PER_RANK - verified)
            generated = blocks_payload(stream, count)
            existing = packed.read(len(generated))
            if generated != existing:
                raise RuntimeError(f"FineWeb continuation mismatch at rank {rank} block {verified}")
            verified += count
            write_json_atomic(
                state_path,
                {
                    "blocks_verified": verified,
                    "continuation_sha256": continuation_sha256,
                    "stream_state": stream.state_dict(),
                },
            )
            print(f"RANK={rank} FASTFORWARD={verified}/{BLOCKS_PER_RANK}", flush=True)

    write_json_atomic(
        seed_path,
        {
            "blocks_verified": BLOCKS_PER_RANK,
            "continuation_sha256": continuation_sha256,
            "stream_state": stream.state_dict(),
        },
    )
    print(f"RANK={rank} FASTFORWARD_COMPLETE", flush=True)


def extend_rank(destination: Path, manifest_path: Path, rank: int) -> None:
    metadata = load_v2_metadata(destination)
    rank_metadata = metadata["ranks"][rank]
    if int(rank_metadata["rank"]) != rank:
        raise ValueError("Packed FineWeb ranks are out of order")
    continuation_path, continuation = continuation_segment(destination, rank_metadata)
    final_path = destination / f"train_rank{rank}.third.uint16"
    metadata_path = destination / f"train_rank{rank}.third.json"
    if final_path.is_file() and metadata_path.is_file():
        third = json.loads(metadata_path.read_text())
        if final_path.stat().st_size != EXTRA_BLOCKS * BLOCK_BYTES:
            raise ValueError("Existing third segment has wrong size")
        if sha256_file(final_path) != third["sha256"]:
            raise ValueError("Existing third segment has wrong SHA-256")
        print(f"RANK={rank} THIRD_SEGMENT=already_complete", flush=True)
        return
    if final_path.exists() or metadata_path.exists():
        raise RuntimeError("Incomplete third-segment finalization requires inspection")

    manifest = load_manifest(manifest_path)
    snapshot = build_snapshot(manifest)
    initial_state = load_initial_state(
        REPOSITORY_ROOT / f"scripts/cloud/fineweb_h200_rank{rank}_after_1xc.json.gz",
        rank,
    )
    stream = make_stream(manifest, snapshot.plan, rank)
    try:
        stream.load_state_dict(initial_state)
        restore_end_of_v2(destination, rank, stream, continuation_path, continuation["sha256"])

        part_path = destination / f"train_rank{rank}.third.uint16.part"
        state_path = destination / f"train_rank{rank}.third.state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            written = int(state["blocks_written"])
            if written < 0 or written > EXTRA_BLOCKS:
                raise ValueError("Third-segment progress out of range")
            if not part_path.is_file() or part_path.stat().st_size < written * BLOCK_BYTES:
                raise RuntimeError("Third-segment partial file is inconsistent")
            with part_path.open("r+b") as output:
                output.truncate(written * BLOCK_BYTES)
            stream.load_state_dict(state["stream_state"])
        else:
            if part_path.exists():
                raise RuntimeError("Third-segment partial file lacks a resume state")
            part_path.touch(exist_ok=False)
            written = 0

        with part_path.open("ab") as output:
            while written < EXTRA_BLOCKS:
                count = min(CHECKPOINT_EVERY_BLOCKS, EXTRA_BLOCKS - written)
                output.write(blocks_payload(stream, count))
                output.flush()
                os.fsync(output.fileno())
                written += count
                write_json_atomic(
                    state_path,
                    {"blocks_written": written, "stream_state": stream.state_dict()},
                )
                print(f"RANK={rank} THIRD_BLOCKS={written}/{EXTRA_BLOCKS}", flush=True)

        expected_bytes = EXTRA_BLOCKS * BLOCK_BYTES
        if part_path.stat().st_size != expected_bytes:
            raise RuntimeError("Third-segment byte count is wrong")
        digest = sha256_file(part_path)
        os.replace(part_path, final_path)
        write_json_atomic(
            metadata_path,
            {
                "rank": rank,
                "file": final_path.name,
                "blocks": EXTRA_BLOCKS,
                "bytes": expected_bytes,
                "sha256": digest,
            },
        )
        print(f"RANK={rank} THIRD_SEGMENT_COMPLETE sha256={digest}", flush=True)
    finally:
        stream.close()


def finalize(destination: Path) -> None:
    metadata_path = destination / "packed_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["format"] == "packed_fineweb_h200_v3":
        if int(metadata["blocks_per_rank"]) != TARGET_BLOCKS:
            raise ValueError("Existing v3 metadata has unexpected length")
        print("PACKED_THIRD_EXTENSION=already_complete", flush=True)
        return
    metadata = load_v2_metadata(destination)
    for rank, rank_metadata in enumerate(metadata["ranks"]):
        continuation_segment(destination, rank_metadata)
        third_path = destination / f"train_rank{rank}.third.json"
        third = json.loads(third_path.read_text())
        data_path = destination / third["file"]
        if int(third["rank"]) != rank or int(third["blocks"]) != EXTRA_BLOCKS:
            raise ValueError("Third-segment metadata mismatch")
        if data_path.stat().st_size != EXTRA_BLOCKS * BLOCK_BYTES:
            raise ValueError("Third-segment file size mismatch")
        if sha256_file(data_path) != third["sha256"]:
            raise ValueError("Third-segment file SHA-256 mismatch")
        rank_metadata["segments"].append(third)
        rank_metadata["blocks"] += EXTRA_BLOCKS
        rank_metadata["bytes"] += EXTRA_BLOCKS * BLOCK_BYTES
        if int(rank_metadata["blocks"]) != TARGET_BLOCKS:
            raise ValueError("Final packed block count mismatch")

    backup_path = destination / "packed_metadata.v2.json"
    original_bytes = metadata_path.read_bytes()
    if backup_path.exists():
        if backup_path.read_bytes() != original_bytes:
            raise ValueError("Existing v2 metadata backup differs from current metadata")
    else:
        with backup_path.open("xb") as backup:
            backup.write(original_bytes)
            backup.flush()
            os.fsync(backup.fileno())

    metadata["format"] = "packed_fineweb_h200_v3"
    metadata["iterations"] = TARGET_ITERATIONS
    metadata["blocks_per_rank"] = TARGET_BLOCKS
    metadata["third_extension_source"] = "verified replay from H200 1xC stream state"
    write_json_atomic(metadata_path, metadata)
    print("PACKED_THIRD_EXTENSION=complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rank", type=int, choices=range(WORLD_SIZE))
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize(args.destination)
    else:
        if args.manifest is None:
            parser.error("--manifest is required with --rank")
        extend_rank(args.destination, args.manifest, args.rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
