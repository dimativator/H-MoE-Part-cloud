#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi


def checkpoint_iteration(checkpoint_dir: Path, world_size: int) -> int | None:
    expected = [checkpoint_dir / "main.pt"] + [
        checkpoint_dir / f"worker_{rank}.pt" for rank in range(world_size)
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return None
    try:
        checkpoint = torch.load(
            checkpoint_dir / "main.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        return int(checkpoint["itr"])
    except (OSError, EOFError, RuntimeError, KeyError):
        return None


def hardlink_snapshot(source: Path, destination: Path, world_size: int) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    names = ["main.pt"] + [f"worker_{rank}.pt" for rank in range(world_size)]
    paths = []
    for name in names:
        target = destination / name
        os.link(source / name, target)
        paths.append(target)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    token = None
    if args.token_file is not None:
        token = args.token_file.read_text().strip()
        if not token:
            raise RuntimeError("Hugging Face token file is empty")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    uploaded = set()

    while True:
        iteration = checkpoint_iteration(args.checkpoint_dir, args.world_size)
        if iteration is not None and iteration not in uploaded:
            snapshot = args.staging_dir / str(iteration)
            if not snapshot.exists():
                files = hardlink_snapshot(
                    args.checkpoint_dir, snapshot, args.world_size
                )
                snapshot_iteration = checkpoint_iteration(snapshot, args.world_size)
                if snapshot_iteration != iteration:
                    raise RuntimeError(
                        f"Checkpoint changed while snapshotting: "
                        f"expected {iteration}, got {snapshot_iteration}"
                    )
                manifest = {
                    "iteration": iteration,
                    "world_size": args.world_size,
                    "files": {
                        path.name: {
                            "size": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                        for path in files
                    },
                }
                (snapshot / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=str(snapshot),
                path_in_repo=f"{args.remote_prefix}/{iteration}",
                commit_message=f"Relay SlimAdam checkpoint at iteration {iteration}",
            )
            for path in snapshot.iterdir():
                path.unlink()
            snapshot.rmdir()
            uploaded.add(iteration)
            print(f"RELAY_UPLOADED iteration={iteration}", flush=True)

        if args.stop_file.exists():
            final_iteration = checkpoint_iteration(
                args.checkpoint_dir, args.world_size
            )
            if final_iteration is None or final_iteration in uploaded:
                print("RELAY_COMPLETE", flush=True)
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
