from __future__ import annotations

import hashlib
import os
from typing import Any, Sequence

import numpy as np
import torch


def _blocks_from_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("FineWeb replay batches must be matching 2-D tensors")
    return torch.cat((x, y[:, -1:]), dim=1)


def blocks_sha256(blocks: torch.Tensor, *, dtype: str = "<u2") -> str:
    array = blocks.detach().to(device="cpu", dtype=torch.int64).numpy()
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


class FineWebReplayTrainReader:
    """Replay several source ranks as one concatenated single-GPU batch."""

    def __init__(
        self,
        readers: Sequence[Any],
        *,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if len(readers) < 2:
            raise ValueError("FineWeb replay requires at least two source readers")
        if batch_size <= 0 or batch_size % len(readers) != 0:
            raise ValueError("Replay batch_size must divide evenly across source readers")

        source_batch_size = batch_size // len(readers)
        for source_rank, reader in enumerate(readers):
            if int(reader.batch_size) != source_batch_size:
                raise ValueError(
                    f"Replay source rank {source_rank} has batch_size={reader.batch_size}; "
                    f"expected {source_batch_size}"
                )
            if int(reader.sequence_length) != sequence_length:
                raise ValueError("Replay source sequence length does not match the run")

        self.readers = list(readers)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.source_world_size = len(readers)
        self.step = 0
        self.requires_checkpoint_state = any(
            getattr(reader, "requires_checkpoint_state", False)
            for reader in self.readers
        )
        self._hash_steps = int(os.environ.get("FINEWEB_BATCH_HASH_STEPS", "0"))
        if self._hash_steps < 0:
            raise ValueError("FINEWEB_BATCH_HASH_STEPS must be non-negative")

    def set_step(self, step: int) -> None:
        for reader in self.readers:
            reader.set_step(step)
        self.step = step

    def sample_batch(self):
        source_batches = [reader.sample_batch() for reader in self.readers]
        x = torch.cat([batch[0] for batch in source_batches], dim=0)
        y = torch.cat([batch[1] for batch in source_batches], dim=0)
        if x.shape != (self.batch_size, self.sequence_length):
            raise RuntimeError(f"Unexpected replay batch shape: {tuple(x.shape)}")

        if self.step < self._hash_steps:
            source_hashes = [
                blocks_sha256(_blocks_from_xy(source_x, source_y))
                for source_x, source_y in source_batches
            ]
            combined_hash = blocks_sha256(_blocks_from_xy(x, y))
            source_text = " ".join(
                f"source_rank{rank}={digest}"
                for rank, digest in enumerate(source_hashes)
            )
            print(
                f"FINEWEB_REPLAY_BATCH_SHA256 microstep={self.step} "
                f"combined={combined_hash} {source_text}",
                flush=True,
            )

        self.step += 1
        return x, y

    def state_dict(self) -> dict[str, Any]:
        return {
            "reader_type": "fineweb_replay_train_reader_v1",
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "source_world_size": self.source_world_size,
            "step": self.step,
            "source_states": [reader.state_dict() for reader in self.readers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("reader_type") != "fineweb_replay_train_reader_v1":
            raise RuntimeError("Unsupported FineWeb replay checkpoint format")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Checkpoint batch_size does not match the replay reader")
        if int(state["sequence_length"]) != self.sequence_length:
            raise ValueError("Checkpoint sequence length does not match the replay reader")
        if int(state["source_world_size"]) != self.source_world_size:
            raise ValueError("Checkpoint source world size does not match the replay reader")

        source_states = state["source_states"]
        if len(source_states) != self.source_world_size:
            raise ValueError("Checkpoint has the wrong number of replay source states")
        for reader, source_state in zip(self.readers, source_states, strict=True):
            reader.load_state_dict(source_state)
        self.step = int(state["step"])


class PermutedFineWebReplayTrainReader:
    """Read the same packed microsteps in a seed-dependent order."""

    requires_checkpoint_state = False

    def __init__(self, reader: FineWebReplayTrainReader, *, steps: int, seed: int):
        if steps <= 0:
            raise ValueError("Packed FineWeb permutation needs a positive step count")
        if any(source._num_steps < steps for source in reader.readers):
            raise ValueError("Packed FineWeb has fewer microsteps than the requested run")
        self.reader = reader
        self.batch_size = reader.batch_size
        self.sequence_length = reader.sequence_length
        self.order = np.random.default_rng(seed).permutation(steps)
        self.step = 0
        digest = hashlib.sha256(self.order.astype("<u4").tobytes()).hexdigest()
        print(
            f"FINEWEB_PACKED_STEP_PERMUTATION seed={seed} steps={steps} "
            f"sha256={digest}",
            flush=True,
        )

    def set_step(self, step: int) -> None:
        if step < 0 or step > len(self.order):
            raise ValueError("Permuted packed FineWeb step is out of range")
        self.step = step

    def sample_batch(self):
        if self.step >= len(self.order):
            raise RuntimeError("Permuted packed FineWeb train reader exhausted")
        self.reader.set_step(int(self.order[self.step]))
        batch = self.reader.sample_batch()
        self.step += 1
        return batch


class FineWebSerialReplayTrainReader:
    """Replay source ranks round-robin while preserving the run batch size."""

    def __init__(
        self,
        readers: Sequence[Any],
        *,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if len(readers) < 2:
            raise ValueError("Serial FineWeb replay requires at least two source readers")
        for source_rank, reader in enumerate(readers):
            if int(reader.batch_size) != batch_size:
                raise ValueError(
                    f"Serial replay source rank {source_rank} has "
                    f"batch_size={reader.batch_size}; expected {batch_size}"
                )
            if int(reader.sequence_length) != sequence_length:
                raise ValueError("Serial replay source sequence length does not match the run")

        self.readers = list(readers)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.source_world_size = len(readers)
        self.step = 0
        self.requires_checkpoint_state = any(
            getattr(reader, "requires_checkpoint_state", False)
            for reader in self.readers
        )
        self._hash_steps = int(os.environ.get("FINEWEB_BATCH_HASH_STEPS", "0"))
        if self._hash_steps < 0:
            raise ValueError("FINEWEB_BATCH_HASH_STEPS must be non-negative")

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("Serial FineWeb replay step must be non-negative")
        complete_rounds, remainder = divmod(step, self.source_world_size)
        for source_rank, reader in enumerate(self.readers):
            reader.set_step(complete_rounds + int(source_rank < remainder))
        self.step = step

    def sample_batch(self):
        source_rank = self.step % self.source_world_size
        x, y = self.readers[source_rank].sample_batch()
        if x.shape != (self.batch_size, self.sequence_length):
            raise RuntimeError(f"Unexpected serial replay batch shape: {tuple(x.shape)}")
        if self.step < self._hash_steps:
            digest = blocks_sha256(_blocks_from_xy(x, y))
            print(
                f"FINEWEB_SERIAL_REPLAY_BATCH_SHA256 microstep={self.step} "
                f"source_rank={source_rank} digest={digest}",
                flush=True,
            )
        self.step += 1
        return x, y

    def state_dict(self) -> dict[str, Any]:
        return {
            "reader_type": "fineweb_serial_replay_train_reader_v1",
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "source_world_size": self.source_world_size,
            "step": self.step,
            "source_states": [reader.state_dict() for reader in self.readers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("reader_type") != "fineweb_serial_replay_train_reader_v1":
            raise RuntimeError("Unsupported serial FineWeb replay checkpoint format")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Checkpoint batch_size does not match serial replay")
        if int(state["sequence_length"]) != self.sequence_length:
            raise ValueError("Checkpoint sequence length does not match serial replay")
        if int(state["source_world_size"]) != self.source_world_size:
            raise ValueError("Checkpoint source world size does not match serial replay")
        source_states = state["source_states"]
        if len(source_states) != self.source_world_size:
            raise ValueError("Checkpoint has the wrong number of serial replay states")
        for reader, source_state in zip(self.readers, source_states, strict=True):
            reader.load_state_dict(source_state)
        self.step = int(state["step"])
