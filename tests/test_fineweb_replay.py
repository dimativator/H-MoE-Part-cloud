from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from src.data.fineweb_packed import (
    EXPECTED_MANIFEST_FINGERPRINT,
    EXPECTED_SPLIT_PLAN_FINGERPRINT,
    EXPECTED_VAL_BLOCKS_SHA256,
    PackedFineWebTrainReader,
    build_packed_fineweb_readers,
)
from src.data.fineweb_replay import (
    FineWebReplayTrainReader,
    FineWebSerialReplayTrainReader,
    blocks_sha256,
)
from src.distributed.ddp import DataParallelDistributedBackend


class FineWebReplayTrainReaderTest(unittest.TestCase):
    def _build_reader(self, root: Path, rank_blocks: list[np.ndarray]):
        ranks = []
        for rank, blocks in enumerate(rank_blocks):
            path = root / f"rank{rank}.uint16"
            payload = np.asarray(blocks, dtype="<u2").tobytes()
            path.write_bytes(payload)
            ranks.append(
                {
                    "rank": rank,
                    "file": path.name,
                    "blocks": len(blocks),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        metadata = {
            "world_size": 2,
            "batch_size": 2,
            "sequence_length": 2,
            "ranks": ranks,
        }
        sources = [
            PackedFineWebTrainReader(
                root,
                metadata,
                rank=rank,
                world_size=2,
                batch_size=2,
                sequence_length=2,
            )
            for rank in range(2)
        ]
        return FineWebReplayTrainReader(
            sources,
            batch_size=4,
            sequence_length=2,
        )

    def test_concatenates_source_rank_batches_in_rank_order(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            reader = self._build_reader(Path(temporary), rank_blocks)
            x, y = reader.sample_batch()

        expected_blocks = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [101, 102, 103], [104, 105, 106]]
        )
        torch.testing.assert_close(x, expected_blocks[:, :-1])
        torch.testing.assert_close(y, expected_blocks[:, 1:])

    def test_logs_stable_hashes_and_restores_both_sources(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(os.environ, {"FINEWEB_BATCH_HASH_STEPS": "1"}):
                reader = self._build_reader(root, rank_blocks)
                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    reader.sample_batch()
                state = reader.state_dict()

                restored = self._build_reader(root, rank_blocks)
                restored.load_state_dict(state)
                restored_x, restored_y = restored.sample_batch()

        first_blocks = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [101, 102, 103], [104, 105, 106]]
        )
        self.assertIn(f"combined={blocks_sha256(first_blocks)}", output.getvalue())
        expected_next = torch.tensor(
            [[7, 8, 9], [10, 11, 12], [107, 108, 109], [110, 111, 112]]
        )
        torch.testing.assert_close(restored_x, expected_next[:, :-1])
        torch.testing.assert_close(restored_y, expected_next[:, 1:])


class FineWebSerialReplayTrainReaderTest(FineWebReplayTrainReaderTest):
    def _build_reader(self, root: Path, rank_blocks: list[np.ndarray]):
        ranks = []
        for rank, blocks in enumerate(rank_blocks):
            path = root / f"rank{rank}.uint16"
            payload = np.asarray(blocks, dtype="<u2").tobytes()
            path.write_bytes(payload)
            ranks.append(
                {
                    "rank": rank,
                    "file": path.name,
                    "blocks": len(blocks),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        metadata = {
            "world_size": 2,
            "batch_size": 2,
            "sequence_length": 2,
            "ranks": ranks,
        }
        sources = [
            PackedFineWebTrainReader(
                root,
                metadata,
                rank=rank,
                world_size=2,
                batch_size=2,
                sequence_length=2,
            )
            for rank in range(2)
        ]
        return FineWebSerialReplayTrainReader(
            sources,
            batch_size=2,
            sequence_length=2,
        )

    def test_concatenates_source_rank_batches_in_rank_order(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            reader = self._build_reader(Path(temporary), rank_blocks)
            rank0_x, rank0_y = reader.sample_batch()
            rank1_x, rank1_y = reader.sample_batch()

        expected_rank0 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_rank1 = torch.tensor([[101, 102, 103], [104, 105, 106]])
        torch.testing.assert_close(rank0_x, expected_rank0[:, :-1])
        torch.testing.assert_close(rank0_y, expected_rank0[:, 1:])
        torch.testing.assert_close(rank1_x, expected_rank1[:, :-1])
        torch.testing.assert_close(rank1_y, expected_rank1[:, 1:])

    def test_logs_stable_hashes_and_restores_both_sources(self):
        rank_blocks = [
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.asarray(
                [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = self._build_reader(root, rank_blocks)
            reader.sample_batch()
            reader.sample_batch()
            reader.sample_batch()
            state = reader.state_dict()

            restored = self._build_reader(root, rank_blocks)
            restored.load_state_dict(state)
            restored_x, restored_y = restored.sample_batch()

        expected = torch.tensor([[107, 108, 109], [110, 111, 112]])
        torch.testing.assert_close(restored_x, expected[:, :-1])
        torch.testing.assert_close(restored_y, expected[:, 1:])


class PackedFineWebMultiGpuParityTest(unittest.TestCase):
    def _write_snapshot(self, root: Path) -> None:
        rank_blocks = [
            np.asarray(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18],
                    [19, 20, 21],
                    [22, 23, 24],
                ]
            ),
            np.asarray(
                [
                    [101, 102, 103],
                    [104, 105, 106],
                    [107, 108, 109],
                    [110, 111, 112],
                    [113, 114, 115],
                    [116, 117, 118],
                    [119, 120, 121],
                    [122, 123, 124],
                ]
            ),
        ]
        ranks = []
        for rank, blocks in enumerate(rank_blocks):
            path = root / f"rank{rank}.uint16"
            payload = np.asarray(blocks, dtype="<u2").tobytes()
            path.write_bytes(payload)
            ranks.append(
                {
                    "rank": rank,
                    "file": path.name,
                    "blocks": len(blocks),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        validation = np.asarray([[201, 202, 203]], dtype="<u2")
        validation_path = root / "validation.uint16"
        validation_payload = validation.tobytes()
        validation_path.write_bytes(validation_payload)
        metadata = {
            "format": "packed_fineweb_h200_v1",
            "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
            "split_plan_fingerprint": EXPECTED_SPLIT_PLAN_FINGERPRINT,
            "validation_blocks_sha256": EXPECTED_VAL_BLOCKS_SHA256,
            "world_size": 2,
            "batch_size": 4,
            "sequence_length": 2,
            "ranks": ranks,
            "validation": {
                "file": validation_path.name,
                "blocks": 1,
                "bytes": len(validation_payload),
                "sha256": hashlib.sha256(validation_payload).hexdigest(),
            },
        }
        (root / "packed_metadata.json").write_text(json.dumps(metadata))

    def _reader(self, root: Path, *, rank: int, world_size: int):
        args = SimpleNamespace(
            datasets_dir=str(root),
            fineweb_replay_world_size=2 if world_size == 1 else 1,
            fineweb_replay_layout="concat",
            batch_size=8 // world_size,
            sequence_length=2,
            eval_batch_size=1,
        )
        return build_packed_fineweb_readers(
            args,
            rank=rank,
            world_size=world_size,
        )["train"]

    @staticmethod
    def _blocks(reader) -> torch.Tensor:
        x, y = reader.sample_batch()
        return torch.cat((x, y[:, -1:]), dim=1)

    def test_one_two_and_four_gpu_global_batches_are_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_snapshot(root)
            one_gpu = self._reader(root, rank=0, world_size=1)
            two_gpu = [
                self._reader(root, rank=rank, world_size=2) for rank in range(2)
            ]
            four_gpu = [
                self._reader(root, rank=rank, world_size=4) for rank in range(4)
            ]

            for _ in range(2):
                expected = self._blocks(one_gpu)
                actual_two = torch.cat([self._blocks(reader) for reader in two_gpu])
                actual_four = torch.cat([self._blocks(reader) for reader in four_gpu])
                torch.testing.assert_close(actual_two, expected)
                torch.testing.assert_close(actual_four, expected)
                self.assertEqual(blocks_sha256(actual_two), blocks_sha256(expected))
                self.assertEqual(blocks_sha256(actual_four), blocks_sha256(expected))

    def test_launcher_values_become_the_expected_physical_batches(self):
        for world_size in (1, 2, 4):
            with self.subTest(world_size=world_size):
                backend = DataParallelDistributedBackend.__new__(
                    DataParallelDistributedBackend
                )
                backend.local_rank = 0
                backend.get_world_size = lambda: world_size
                args = SimpleNamespace(
                    batch_size=32 // world_size,
                    acc_steps=4 * world_size,
                    device="cuda",
                    seed=0,
                    data_seed=1337,
                )
                adjusted = backend.get_adjusted_args_for_process(args)
                self.assertEqual(adjusted.batch_size, 32 // world_size)
                self.assertEqual(adjusted.acc_steps, 4)
                self.assertEqual(
                    world_size * adjusted.batch_size * adjusted.acc_steps,
                    128,
                )

    def test_four_gpu_reader_restores_the_same_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_snapshot(root)
            reader = self._reader(root, rank=3, world_size=4)
            reader.sample_batch()
            state = reader.state_dict()
            restored = self._reader(root, rank=3, world_size=4)
            restored.load_state_dict(state)
            blocks = self._blocks(restored)

        torch.testing.assert_close(
            blocks,
            torch.tensor([[119, 120, 121], [122, 123, 124]]),
        )

    def test_segmented_snapshot_crosses_the_one_x_c_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_snapshot(root)
            metadata_path = root / "packed_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["format"] = "packed_fineweb_h200_v2"
            for rank_metadata in metadata["ranks"]:
                rank = int(rank_metadata["rank"])
                first = dict(rank_metadata)
                continuation_blocks = np.asarray(
                    [
                        [1001 + rank * 100, 1002 + rank * 100, 1003 + rank * 100],
                        [1004 + rank * 100, 1005 + rank * 100, 1006 + rank * 100],
                        [1007 + rank * 100, 1008 + rank * 100, 1009 + rank * 100],
                        [1010 + rank * 100, 1011 + rank * 100, 1012 + rank * 100],
                    ],
                    dtype="<u2",
                )
                continuation_path = root / f"rank{rank}.continuation.uint16"
                payload = continuation_blocks.tobytes()
                continuation_path.write_bytes(payload)
                continuation = {
                    "file": continuation_path.name,
                    "blocks": len(continuation_blocks),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                rank_metadata["segments"] = [first, continuation]
                rank_metadata["blocks"] = first["blocks"] + continuation["blocks"]
                rank_metadata["bytes"] = first["bytes"] + continuation["bytes"]
                rank_metadata.pop("file")
                rank_metadata.pop("sha256")
            metadata_path.write_text(json.dumps(metadata))

            readers = [self._reader(root, rank=rank, world_size=2) for rank in range(2)]
            for reader in readers:
                reader.set_step(2)
            boundary_batch = torch.cat([self._blocks(reader) for reader in readers])

        torch.testing.assert_close(
            boundary_batch,
            torch.tensor(
                [
                    [1001, 1002, 1003],
                    [1004, 1005, 1006],
                    [1007, 1008, 1009],
                    [1010, 1011, 1012],
                    [1101, 1102, 1103],
                    [1104, 1105, 1106],
                    [1107, 1108, 1109],
                    [1110, 1111, 1112],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
