#!/usr/bin/env python3
"""Download and verify a private Hugging Face relay using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def request(url: str, token: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
        timeout=300,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--token-file", default=Path.home() / ".cache/huggingface/token", type=Path
    )
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    base_url = f"https://huggingface.co/{args.repo_id}/resolve/main"
    manifest_url = f"{base_url}/manifest.json"
    with request(manifest_url, token) as response:
        manifest_bytes = response.read()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    args.destination.mkdir(parents=True, exist_ok=True)
    expected_paths: set[Path] = set()
    for index, item in enumerate(manifest["files"], start=1):
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe manifest path: {relative}")
        target = args.destination / relative
        if args.destination.resolve() not in target.parent.resolve().parents and target.parent.resolve() != args.destination.resolve():
            raise RuntimeError(f"target escapes destination: {target}")
        expected_paths.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            target.is_file()
            and target.stat().st_size == item["size"]
            and sha256_file(target) == item["sha256"]
        ):
            print(f"ALREADY_VERIFIED {index}/{len(manifest['files'])} {target}", flush=True)
            continue
        incoming = target.with_name(f".{target.name}.incoming")
        url = f"{base_url}/{urllib.parse.quote(item['path'], safe='/')}"
        digest = hashlib.sha256()
        size = 0
        with request(url, token) as response, incoming.open("wb") as handle:
            while chunk := response.read(16 * 1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != item["size"] or digest.hexdigest() != item["sha256"]:
            incoming.unlink(missing_ok=True)
            raise RuntimeError(
                f"download verification failed for {item['path']}: "
                f"size={size}/{item['size']} sha256={digest.hexdigest()}/{item['sha256']}"
            )
        os.replace(incoming, target)
        print(f"DOWNLOADED_VERIFIED {index}/{len(manifest['files'])} {target}", flush=True)

    local_manifest = args.destination / "manifest.json"
    local_manifest.write_bytes(manifest_bytes)
    total_bytes = sum(item["size"] for item in manifest["files"])
    print(f"H200_RELAY_VERIFIED files={len(expected_paths)} bytes={total_bytes}", flush=True)
    print(f"MANIFEST_SHA256={manifest_sha256}", flush=True)
    print(f"DESTINATION={args.destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
