#!/usr/bin/env python3
"""Delete only Frugal Muon checkpoints matching a verified relay manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


RESULTS_ROOT = Path(
    "/workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-20260912"
)
GROUP = "2xChinchilla_500M_frugal_muon_2gpu_cloud"
EXPERIMENTS = (
    "llama500M_frugal_muon_adamw_bf16_2xC_2gpu",
    "llama500M_frugal_muon_adamw_fp8_full_2xC_2gpu",
)
CHECKPOINT_NAMES = ("67911", "latest")
REMOTE_PREFIX = "frugal-muon-500m-2gpu-20260912"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    expected_manifest_sha256 = os.environ["EXPECTED_MANIFEST_SHA256"]
    roots: list[Path] = []
    files: list[Path] = []
    for experiment in EXPERIMENTS:
        for checkpoint_name in CHECKPOINT_NAMES:
            root = RESULTS_ROOT / GROUP / experiment / "ckpts" / checkpoint_name
            resolved = root.resolve(strict=True)
            if resolved != root or RESULTS_ROOT not in resolved.parents:
                raise RuntimeError(f"unsafe checkpoint root: {root}")
            roots.append(root)
            for path in sorted(root.iterdir()):
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise RuntimeError(f"refusing non-regular checkpoint entry: {path}")
                files.append(path)

    manifest = {
        "format": 1,
        "source_root": str(RESULTS_ROOT),
        "files": [
            {
                "path": f"{REMOTE_PREFIX}/{path.relative_to(RESULTS_ROOT).as_posix()}",
                "relative_path": path.relative_to(RESULTS_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "checkpoint manifest changed after relay: "
            f"expected={expected_manifest_sha256} actual={actual_manifest_sha256}"
        )

    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"DELETE_VALIDATED files={len(files)} bytes={total_bytes} "
        f"manifest_sha256={actual_manifest_sha256}",
        flush=True,
    )
    for root in roots:
        shutil.rmtree(root)
        print(f"DELETED={root}", flush=True)
    for experiment in EXPERIMENTS:
        ckpts = RESULTS_ROOT / GROUP / experiment / "ckpts"
        if ckpts.is_dir() and not any(ckpts.iterdir()):
            ckpts.rmdir()
    print(f"CHECKPOINT_DELETE_COMPLETE bytes={total_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
