#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Hugging Face token file is empty")

    with tempfile.TemporaryDirectory(prefix="slimadam-checkpoint-") as temp_dir:
        incoming = Path(temp_dir)
        manifest_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=f"{args.remote_dir}/manifest.json",
                repo_type="model",
                token=token,
                local_dir=incoming,
            )
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["iteration"]) != args.expected_iteration:
            raise RuntimeError("Unexpected checkpoint iteration in manifest")
        if int(manifest["world_size"]) != 1:
            raise RuntimeError("Expected a one-worker checkpoint")

        source_dir = manifest_path.parent
        for name, expected in manifest["files"].items():
            path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=f"{args.remote_dir}/{name}",
                    repo_type="model",
                    token=token,
                    local_dir=incoming,
                )
            )
            if path.stat().st_size != int(expected["size"]):
                raise RuntimeError(f"Size mismatch for {name}")
            if sha256(path) != str(expected["sha256"]):
                raise RuntimeError(f"SHA256 mismatch for {name}")

        checkpoint = torch.load(
            source_dir / "main.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if int(checkpoint["itr"]) != args.expected_iteration:
            raise RuntimeError("Unexpected iteration in main.pt")
        del checkpoint

        if args.destination.exists():
            existing_manifest = args.destination / "manifest.json"
            if existing_manifest.is_file() and json.loads(
                existing_manifest.read_text(encoding="utf-8")
            ) == manifest:
                print(f"CHECKPOINT_ALREADY_READY destination={args.destination}")
                return 0
            raise FileExistsError(f"Destination already exists: {args.destination}")

        args.destination.parent.mkdir(parents=True, exist_ok=True)
        staged = args.destination.with_name(f".{args.destination.name}.incoming")
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(source_dir, staged)
        staged.rename(args.destination)

    print(
        f"CHECKPOINT_READY iteration={args.expected_iteration} "
        f"destination={args.destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
