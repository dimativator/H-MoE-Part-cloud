from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .fineweb import FineWebValReader
from .fineweb_replay import (
    FineWebReplayTrainReader,
    FineWebSerialReplayTrainReader,
    blocks_sha256,
)


EXPECTED_FORMATS = {"packed_fineweb_h200_v1", "packed_fineweb_h200_v2"}
EXPECTED_MANIFEST_FINGERPRINT = (
    "7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b"
)
EXPECTED_SPLIT_PLAN_FINGERPRINT = (
    "550b33876a3810f4bee389f6b897584017bf9f7a6ac4456aa0de9cf043c09455"
)
EXPECTED_VAL_BLOCKS_SHA256 = (
    "d9b18bcef1a4ef61a493dbcf2ebb2afadd8fa2a207111dd004d34761e406448e"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_rank_segments(
    root: Path,
    rank_metadata: dict[str, Any],
    *,
    block_tokens: int,
    rank: int,
) -> tuple[list[tuple[int, int, np.memmap]], int]:
    segment_metadata = rank_metadata.get("segments", [rank_metadata])
    segments: list[tuple[int, int, np.memmap]] = []
    offset = 0
    total_bytes = 0
    for segment in segment_metadata:
        path = root / str(segment["file"])
        blocks = int(segment["blocks"])
        expected_bytes = int(segment["bytes"])
        if expected_bytes != blocks * block_tokens * 2:
            raise ValueError(f"Packed FineWeb byte count mismatch for rank {rank}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Packed FineWeb size mismatch for rank {rank}")
        if _sha256_file(path) != str(segment["sha256"]):
            raise ValueError(f"Packed FineWeb SHA-256 mismatch for rank {rank}")
        tokens = np.memmap(
            path,
            mode="r",
            dtype="<u2",
            shape=(blocks, block_tokens),
        )
        segments.append((offset, offset + blocks, tokens))
        offset += blocks
        total_bytes += expected_bytes
    if offset != int(rank_metadata["blocks"]):
        raise ValueError(f"Packed FineWeb block count mismatch for rank {rank}")
    if total_bytes != int(rank_metadata["bytes"]):
        raise ValueError(f"Packed FineWeb total byte count mismatch for rank {rank}")
    return segments, offset


def _read_rank_blocks(
    segments: list[tuple[int, int, np.memmap]], start: int, end: int
) -> np.ndarray:
    chunks = [
        tokens[max(start, segment_start) - segment_start : min(end, segment_end) - segment_start]
        for segment_start, segment_end, tokens in segments
        if start < segment_end and end > segment_start
    ]
    if not chunks or sum(len(chunk) for chunk in chunks) != end - start:
        raise RuntimeError("Packed FineWeb segment coverage is incomplete")
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


class PackedFineWebTrainReader:
    requires_checkpoint_state = False

    def __init__(
        self,
        root: Path,
        metadata: dict[str, Any],
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        sequence_length: int,
    ):
        if world_size != int(metadata["world_size"]):
            raise ValueError("Packed FineWeb world_size does not match the run")
        if batch_size != int(metadata["batch_size"]):
            raise ValueError("Packed FineWeb batch_size does not match the run")
        if sequence_length != int(metadata["sequence_length"]):
            raise ValueError("Packed FineWeb sequence_length does not match the run")

        rank_metadata = metadata["ranks"][rank]
        if int(rank_metadata["rank"]) != rank:
            raise ValueError("Packed FineWeb rank metadata is out of order")
        self.rank = rank
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.block_tokens = sequence_length + 1
        self._segments, self.blocks = _open_rank_segments(
            root,
            rank_metadata,
            block_tokens=self.block_tokens,
            rank=rank,
        )
        self._num_steps = self.blocks // batch_size
        self.step = 0

    def set_step(self, step: int) -> None:
        if step < 0 or step > self._num_steps:
            raise ValueError("Packed FineWeb step is out of range")
        self.step = step

    def sample_batch(self):
        if self.step >= self._num_steps:
            raise RuntimeError("Packed FineWeb train reader exhausted")
        start = self.step * self.batch_size
        end = start + self.batch_size
        batch = torch.from_numpy(
            np.array(
                _read_rank_blocks(self._segments, start, end),
                dtype=np.int64,
                copy=True,
            )
        )
        self.step += 1
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        return {
            "reader_type": "packed_fineweb_train_reader_v1",
            "rank": int(self.rank),
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("reader_type") != "packed_fineweb_train_reader_v1":
            raise RuntimeError("Unsupported packed FineWeb checkpoint format")
        if int(state["rank"]) != self.rank:
            raise ValueError("Checkpoint rank does not match packed FineWeb reader")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Checkpoint batch size does not match packed FineWeb reader")
        if int(state["sequence_length"]) != self.sequence_length:
            raise ValueError("Checkpoint sequence length does not match packed FineWeb reader")
        self.set_step(int(state["step"]))


class PackedFineWebShardedTrainReader:
    """Split each packed source-rank batch across several physical ranks."""

    requires_checkpoint_state = False

    def __init__(
        self,
        root: Path,
        metadata: dict[str, Any],
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        sequence_length: int,
    ):
        source_world_size = int(metadata["world_size"])
        source_batch_size = int(metadata["batch_size"])
        if world_size <= source_world_size or world_size % source_world_size != 0:
            raise ValueError(
                "Packed FineWeb physical world_size must be a multiple of the "
                "source world_size"
            )
        if rank < 0 or rank >= world_size:
            raise ValueError("Packed FineWeb physical rank is out of range")

        shards_per_source = world_size // source_world_size
        if source_batch_size % shards_per_source != 0:
            raise ValueError(
                "Packed FineWeb source batch cannot be divided across physical ranks"
            )
        expected_batch_size = source_batch_size // shards_per_source
        if batch_size != expected_batch_size:
            raise ValueError(
                f"Packed FineWeb per-rank batch_size must be {expected_batch_size}; "
                f"got {batch_size}"
            )
        if sequence_length != int(metadata["sequence_length"]):
            raise ValueError("Packed FineWeb sequence_length does not match the run")

        source_rank, shard_rank = divmod(rank, shards_per_source)
        rank_metadata = metadata["ranks"][source_rank]
        if int(rank_metadata["rank"]) != source_rank:
            raise ValueError("Packed FineWeb rank metadata is out of order")
        self.rank = rank
        self.source_rank = source_rank
        self.shard_rank = shard_rank
        self.shards_per_source = shards_per_source
        self.source_batch_size = source_batch_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.block_tokens = sequence_length + 1
        self._segments, self.blocks = _open_rank_segments(
            root,
            rank_metadata,
            block_tokens=self.block_tokens,
            rank=source_rank,
        )
        self._num_steps = self.blocks // source_batch_size
        self.step = 0

    def set_step(self, step: int) -> None:
        if step < 0 or step > self._num_steps:
            raise ValueError("Packed FineWeb step is out of range")
        self.step = step

    def sample_batch(self):
        if self.step >= self._num_steps:
            raise RuntimeError("Packed FineWeb train reader exhausted")
        start = (
            self.step * self.source_batch_size
            + self.shard_rank * self.batch_size
        )
        end = start + self.batch_size
        batch = torch.from_numpy(
            np.array(
                _read_rank_blocks(self._segments, start, end),
                dtype=np.int64,
                copy=True,
            )
        )
        self.step += 1
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        return {
            "reader_type": "packed_fineweb_sharded_train_reader_v1",
            "rank": self.rank,
            "source_rank": self.source_rank,
            "shard_rank": self.shard_rank,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("reader_type") != "packed_fineweb_sharded_train_reader_v1":
            raise RuntimeError("Unsupported packed FineWeb checkpoint format")
        expected = {
            "rank": self.rank,
            "source_rank": self.source_rank,
            "shard_rank": self.shard_rank,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
        }
        for key, value in expected.items():
            if int(state[key]) != value:
                raise ValueError(f"Checkpoint {key} does not match packed FineWeb reader")
        self.set_step(int(state["step"]))


def build_packed_fineweb_readers(args, *, rank: int, world_size: int):
    root = Path(args.datasets_dir).expanduser()
    metadata = json.loads((root / "packed_metadata.json").read_text())
    if metadata.get("format") not in EXPECTED_FORMATS:
        raise ValueError("Unsupported packed FineWeb format")
    if metadata.get("manifest_fingerprint") != EXPECTED_MANIFEST_FINGERPRINT:
        raise ValueError("Packed FineWeb manifest fingerprint does not match H200")
    if metadata.get("split_plan_fingerprint") != EXPECTED_SPLIT_PLAN_FINGERPRINT:
        raise ValueError("Packed FineWeb split fingerprint does not match H200")
    if metadata.get("validation_blocks_sha256") != EXPECTED_VAL_BLOCKS_SHA256:
        raise ValueError("Packed FineWeb validation token hash does not match H200")

    replay_world_size = int(args.fineweb_replay_world_size)
    replay_layout = str(args.fineweb_replay_layout)
    if replay_world_size > 1:
        if world_size != 1 or rank != 0:
            raise ValueError("Packed FineWeb replay requires one training process")
        if replay_world_size != int(metadata["world_size"]):
            raise ValueError("Replay world size does not match packed FineWeb metadata")
        if replay_layout == "concat" and args.batch_size % replay_world_size != 0:
            raise ValueError("Replay world size must divide --batch-size")
        source_batch_size = (
            args.batch_size // replay_world_size
            if replay_layout == "concat"
            else args.batch_size
        )
        source_readers = [
            PackedFineWebTrainReader(
                root,
                metadata,
                rank=source_rank,
                world_size=replay_world_size,
                batch_size=source_batch_size,
                sequence_length=args.sequence_length,
            )
            for source_rank in range(replay_world_size)
        ]
        replay_reader = (
            FineWebReplayTrainReader
            if replay_layout == "concat"
            else FineWebSerialReplayTrainReader
        )
        train_reader = replay_reader(
            source_readers,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
    elif world_size == int(metadata["world_size"]):
        train_reader = PackedFineWebTrainReader(
            root,
            metadata,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
    else:
        train_reader = PackedFineWebShardedTrainReader(
            root,
            metadata,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
    validation_metadata = metadata["validation"]
    validation_path = root / str(validation_metadata["file"])
    if validation_path.stat().st_size != int(validation_metadata["bytes"]):
        raise ValueError("Packed FineWeb validation size mismatch")
    if _sha256_file(validation_path) != str(validation_metadata["sha256"]):
        raise ValueError("Packed FineWeb validation SHA-256 mismatch")
    validation_blocks = int(validation_metadata["blocks"])
    validation_tokens = np.memmap(
        validation_path,
        mode="r",
        dtype="<u2",
        shape=(validation_blocks, args.sequence_length + 1),
    )
    val_reader = FineWebValReader(
        torch.from_numpy(np.array(validation_tokens, dtype=np.int64, copy=True)),
        batch_size=args.eval_batch_size,
        sequence_length=args.sequence_length,
    )
    if int(os.environ.get("FINEWEB_LOG_DATA_HASHES", "0")):
        print(
            f"FINEWEB_VALIDATION_SHA256={blocks_sha256(val_reader.blocks, dtype='<u4')}",
            flush=True,
        )
    return {"train": train_reader, "val": val_reader}
