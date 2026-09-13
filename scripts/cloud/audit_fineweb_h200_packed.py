#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_FORMATS = {"packed_fineweb_h200_v1", "packed_fineweb_h200_v2"}
EXPECTED_MANIFEST_FINGERPRINT = (
    "7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b"
)
EXPECTED_SPLIT_PLAN_FINGERPRINT = (
    "550b33876a3810f4bee389f6b897584017bf9f7a6ac4456aa0de9cf043c09455"
)
EXPECTED_VALIDATION_SHA256_UINT32 = (
    "d9b18bcef1a4ef61a493dbcf2ebb2afadd8fa2a207111dd004d34761e406448e"
)
EXPECTED_PREVIEW_SHA256 = {
    0: "fef09aaf0c5d7056e2421b34a5e5e9e761932721a6d0b14f65ab2050c4690393",
    1: "f7ae4a8aefabb183355afaa4b7b23193b8e8faf42c634a1bb131e99b3c9819a3",
}
EXPECTED_CONTINUATION_PREVIEW_SHA256 = {
    0: "00ea50b3e5bff4225d8f502fe81bc5f3668ef2062a19e1f3bc3344207c7bfd59",
    1: "dc0b936666849907e1bef994b98e9f20b021dcd766775e910bc4ab63cfb8d545",
}
PREVIEW_BLOCKS = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    metadata_path = args.dataset / "packed_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["format"] in EXPECTED_FORMATS
    assert metadata["manifest_fingerprint"] == EXPECTED_MANIFEST_FINGERPRINT
    assert metadata["split_plan_fingerprint"] == EXPECTED_SPLIT_PLAN_FINGERPRINT
    assert (
        metadata["validation_blocks_sha256"]
        == EXPECTED_VALIDATION_SHA256_UINT32
    )

    block_tokens = int(metadata["block_tokens"])
    for rank_metadata in metadata["ranks"]:
        rank = int(rank_metadata["rank"])
        segments = rank_metadata.get("segments", [rank_metadata])
        assert sum(int(segment["blocks"]) for segment in segments) == int(
            rank_metadata["blocks"]
        )
        assert sum(int(segment["bytes"]) for segment in segments) == int(
            rank_metadata["bytes"]
        )
        for segment in segments:
            path = args.dataset / segment["file"]
            assert path.stat().st_size == int(segment["bytes"])
            digest = sha256_file(path)
            assert digest == segment["sha256"]
            print(
                f"rank={rank} segment={path.name} blocks={segment['blocks']} "
                f"bytes={path.stat().st_size} sha256={digest}",
                flush=True,
            )
        first_path = args.dataset / segments[0]["file"]
        preview_bytes = PREVIEW_BLOCKS * block_tokens * 2
        with first_path.open("rb") as source:
            preview_digest = hashlib.sha256(source.read(preview_bytes)).hexdigest()
        assert preview_digest == EXPECTED_PREVIEW_SHA256[rank]
        if len(segments) > 1:
            continuation_path = args.dataset / segments[1]["file"]
            with continuation_path.open("rb") as source:
                continuation_preview_digest = hashlib.sha256(
                    source.read(preview_bytes)
                ).hexdigest()
            assert (
                continuation_preview_digest
                == EXPECTED_CONTINUATION_PREVIEW_SHA256[rank]
            )
            print(
                f"rank={rank} continuation_preview_sha256="
                f"{continuation_preview_digest}",
                flush=True,
            )
        print(
            f"rank={rank} total_blocks={rank_metadata['blocks']} "
            f"total_bytes={rank_metadata['bytes']} preview_sha256={preview_digest}",
            flush=True,
        )

    validation_metadata = metadata["validation"]
    validation_path = args.dataset / validation_metadata["file"]
    assert validation_path.stat().st_size == int(validation_metadata["bytes"])
    validation_file_digest = sha256_file(validation_path)
    assert validation_file_digest == validation_metadata["sha256"]
    validation = np.memmap(validation_path, mode="r", dtype="<u2")
    validation_uint32_digest = hashlib.sha256(
        np.asarray(validation, dtype="<u4").tobytes()
    ).hexdigest()
    assert validation_uint32_digest == EXPECTED_VALIDATION_SHA256_UINT32
    print(
        f"validation_blocks={validation_metadata['blocks']} "
        f"sha256={validation_file_digest} "
        f"token_sha256_uint32={validation_uint32_digest}",
        flush=True,
    )
    print("PACKED_AUDIT=ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
