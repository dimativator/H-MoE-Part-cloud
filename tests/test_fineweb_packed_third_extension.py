from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.cloud import extend_fineweb_h200_packed_for_slimadam_4xc as extension


class FakeStream:
    def __init__(self, blocks: list[list[int]]) -> None:
        self.blocks = blocks
        self.position = 0

    def __next__(self) -> list[int]:
        block = self.blocks[self.position]
        self.position += 1
        return block

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = int(state["position"])


class PackedThirdExtensionTest(unittest.TestCase):
    def test_fast_forward_matches_existing_segment_and_preserves_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            payload = np.asarray(blocks, dtype="<u2").tobytes()
            packed = root / "continuation.uint16"
            packed.write_bytes(payload)
            stream = FakeStream(blocks + [[13, 14, 15]])
            with (
                mock.patch.object(extension, "BLOCKS_PER_RANK", 4),
                mock.patch.object(extension, "BLOCK_TOKENS", 3),
                mock.patch.object(extension, "BLOCK_BYTES", 6),
                mock.patch.object(extension, "CHECKPOINT_EVERY_BLOCKS", 2),
            ):
                extension.restore_end_of_v2(
                    root, 0, stream, packed, hashlib.sha256(payload).hexdigest()
                )
                self.assertEqual(stream.position, 4)
                restored = FakeStream(blocks + [[13, 14, 15]])
                extension.restore_end_of_v2(
                    root, 0, restored, packed, hashlib.sha256(payload).hexdigest()
                )
                self.assertEqual(next(restored), [13, 14, 15])

    def test_fast_forward_rejects_different_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = np.asarray([[1, 2, 3]], dtype="<u2").tobytes()
            packed = root / "continuation.uint16"
            packed.write_bytes(payload)
            with (
                mock.patch.object(extension, "BLOCKS_PER_RANK", 1),
                mock.patch.object(extension, "BLOCK_TOKENS", 3),
                mock.patch.object(extension, "BLOCK_BYTES", 6),
            ):
                with self.assertRaisesRegex(RuntimeError, "continuation mismatch"):
                    extension.restore_end_of_v2(
                        root,
                        0,
                        FakeStream([[9, 9, 9]]),
                        packed,
                        hashlib.sha256(payload).hexdigest(),
                    )

    def test_finalize_keeps_v2_backup_and_appends_third_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ranks = []
            for rank in (0, 1):
                segments = []
                for index, blocks in ((0, 2), (1, 2), (2, 1)):
                    path = root / f"rank{rank}.segment{index}.uint16"
                    payload = np.asarray(
                        [[rank + index + 1, rank + index + 2, rank + index + 3]] * blocks,
                        dtype="<u2",
                    ).tobytes()
                    path.write_bytes(payload)
                    segments.append(
                        {
                            "file": path.name,
                            "blocks": blocks,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                (root / f"train_rank{rank}.third.json").write_text(
                    json.dumps({"rank": rank, **segments[2]})
                )
                ranks.append(
                    {
                        "rank": rank,
                        "segments": segments[:2],
                        "blocks": 4,
                        "bytes": 24,
                    }
                )
            original = {
                "format": "packed_fineweb_h200_v2",
                "manifest_fingerprint": extension.EXPECTED_MANIFEST_FINGERPRINT,
                "split_plan_fingerprint": extension.EXPECTED_SPLIT_PLAN_FINGERPRINT,
                "world_size": 2,
                "batch_size": extension.BATCH_SIZE,
                "blocks_per_rank": 4,
                "ranks": ranks,
            }
            metadata_path = root / "packed_metadata.json"
            metadata_path.write_text(json.dumps(original))
            with (
                mock.patch.object(extension, "BLOCKS_PER_RANK", 2),
                mock.patch.object(extension, "BLOCK_TOKENS", 3),
                mock.patch.object(extension, "BLOCK_BYTES", 6),
                mock.patch.object(extension, "EXTRA_BLOCKS", 1),
                mock.patch.object(extension, "TARGET_BLOCKS", 5),
            ):
                extension.finalize(root)

            updated = json.loads(metadata_path.read_text())
            self.assertEqual(updated["format"], "packed_fineweb_h200_v3")
            self.assertEqual([len(rank["segments"]) for rank in updated["ranks"]], [3, 3])
            self.assertEqual(json.loads((root / "packed_metadata.v2.json").read_text()), original)


if __name__ == "__main__":
    unittest.main()
