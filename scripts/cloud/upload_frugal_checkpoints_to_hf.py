#!/usr/bin/env python3
"""Upload the exact Frugal Muon recovery checkpoints to a private HF relay."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


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


def checkpoint_roots() -> list[Path]:
    roots = []
    for experiment in EXPERIMENTS:
        for checkpoint_name in CHECKPOINT_NAMES:
            root = RESULTS_ROOT / GROUP / experiment / "ckpts" / checkpoint_name
            resolved = root.resolve(strict=True)
            if resolved != root or RESULTS_ROOT not in resolved.parents:
                raise RuntimeError(f"unsafe checkpoint root: {root}")
            roots.append(root)
    return roots


def regular_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for path in sorted(root.iterdir()):
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"refusing non-regular checkpoint entry: {path}")
            files.append(path)
    return files


def remote_path(path: Path) -> str:
    return f"{REMOTE_PREFIX}/{path.relative_to(RESULTS_ROOT).as_posix()}"


def verify_remote(api: HfApi, repo_id: str, manifest: dict) -> None:
    paths = [item["path"] for item in manifest["files"]]
    entries = api.get_paths_info(
        repo_id=repo_id, paths=paths, repo_type="model", revision="main"
    )
    by_path = {entry.path: entry for entry in entries}
    for item in manifest["files"]:
        entry = by_path.get(item["path"])
        if entry is None:
            raise RuntimeError(f"remote file missing: {item['path']}")
        if entry.size != item["size"]:
            raise RuntimeError(
                f"remote size mismatch for {item['path']}: "
                f"expected={item['size']} actual={entry.size}"
            )
        lfs = getattr(entry, "lfs", None)
        remote_sha256 = getattr(lfs, "sha256", None) if lfs else None
        if remote_sha256 != item["sha256"]:
            raise RuntimeError(
                f"remote SHA-256 mismatch for {item['path']}: "
                f"expected={item['sha256']} actual={remote_sha256}"
            )


def main() -> int:
    repo_id = os.environ["HF_RELAY_REPO_ID"]
    roots = checkpoint_roots()
    files = regular_files(roots)
    manifest = {
        "format": 1,
        "source_root": str(RESULTS_ROOT),
        "files": [
            {
                "path": remote_path(path),
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
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    total_bytes = sum(item["size"] for item in manifest["files"])
    print(
        f"MANIFEST files={len(files)} bytes={total_bytes} sha256={manifest_sha256}",
        flush=True,
    )

    api = HfApi(endpoint="https://huggingface.co")
    identity = api.whoami().get("name")
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not info.private:
        raise RuntimeError(f"relay repository must be private: {repo_id}")
    print(f"AUTH identity={identity} repo={repo_id} private={info.private}", flush=True)

    operations = [
        CommitOperationAdd(path_in_repo=item["path"], path_or_fileobj=str(path))
        for item, path in zip(manifest["files"], files, strict=True)
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=manifest_bytes)
    )
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message="Relay Frugal Muon recovery checkpoints to h200_mipt",
            )
            break
        except Exception as error:
            last_error = error
            if attempt == 6:
                raise
            delay = min(300, 10 * 2 ** (attempt - 1))
            print(
                f"UPLOAD_RETRY attempt={attempt}/6 delay={delay}s "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(delay)
    else:  # pragma: no cover
        raise RuntimeError("upload failed") from last_error

    verify_remote(api, repo_id, manifest)
    remote_manifest = hf_hub_download(
        repo_id=repo_id, filename="manifest.json", repo_type="model"
    )
    if hashlib.sha256(Path(remote_manifest).read_bytes()).hexdigest() != manifest_sha256:
        raise RuntimeError("remote manifest SHA-256 mismatch")
    print(f"RELAY_UPLOAD_VERIFIED={repo_id}", flush=True)
    print(f"MANIFEST_SHA256={manifest_sha256}", flush=True)
    print(f"TOTAL_BYTES={total_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
