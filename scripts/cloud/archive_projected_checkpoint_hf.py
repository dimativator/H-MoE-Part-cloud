"""Archive one campaign checkpoint and verify a pinned HF commit. Never delete sources."""

import argparse
import fcntl
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_url

ROOT = Path("/home/dimativator/checkpoints/projected-500m-precisions-20261013")
REPO = "DimaTivator/effpretrain_ckpts"
FILES = ("main.pt", "worker_0.pt", "worker_1.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_bytes(token: str, path: str, revision: str) -> bytes:
    request = Request(hf_hub_url(REPO, path, revision=revision),
                      headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    pattern = r"llama500M_(apollo|galore)_(bf16|fp8_mixed|fp8_full|w16a16g32_fp32_states)_wd1e-2_2xC_warmup2000_2gpu_20261013-iter-67911"
    if not re.fullmatch(pattern, args.name):
        raise ValueError("Not a campaign checkpoint name")
    folder = ROOT / args.name
    if folder.is_symlink() or folder.resolve().parent != ROOT.resolve():
        raise ValueError("Unsafe checkpoint folder")
    if set(path.name for path in folder.iterdir()) != set(FILES):
        raise ValueError("Expected exactly main and two worker checkpoints")
    records = {}
    identities = {}
    for name in FILES:
        path = folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint: {name}")
        info = path.stat()
        identities[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        records[name] = {"size": info.st_size, "sha256": sha256(path)}
    manifest = {"campaign": ROOT.name, "iteration": 67911, "name": args.name,
                "files": records}
    destination = f"{ROOT.name}/{args.name}"
    with Path("/home/dimativator/.codex/archive_precision_hf.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = Path("/home/dimativator/.hf_token").read_text().strip()
        api = HfApi(token=token)
        if api.whoami()["name"] != "DimaTivator":
            raise ValueError("Unexpected HF identity")
        revision = api.repo_info(REPO).sha
        existing = api.list_repo_files(REPO, revision=revision)
        if any(path.startswith(destination + "/") for path in existing):
            if json.loads(remote_bytes(token, destination + "/SHA256.json", revision)) != manifest:
                raise ValueError("Existing HF destination differs; refusing overwrite")
        else:
            operations = [CommitOperationAdd(path_in_repo=f"{destination}/{name}",
                                            path_or_fileobj=str(folder / name)) for name in FILES]
            operations.append(CommitOperationAdd(path_in_repo=destination + "/SHA256.json",
                                                path_or_fileobj=json.dumps(manifest, indent=2).encode()))
            revision = api.create_commit(REPO, operations=operations, num_threads=1,
                                         commit_message=f"Archive {args.name}").oid
        paths = [f"{destination}/{name}" for name in FILES]
        remote = {item.path: item for item in api.get_paths_info(REPO, paths=paths, revision=revision)}
        for name in FILES:
            info = remote[f"{destination}/{name}"]
            lfs = info.lfs
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if not digest:
                digest = hashlib.sha256(remote_bytes(token, f"{destination}/{name}", revision)).hexdigest()
            if info.size != records[name]["size"] or digest != records[name]["sha256"]:
                raise ValueError(f"Pinned HF verification failed: {name}")
        if json.loads(remote_bytes(token, destination + "/SHA256.json", revision)) != manifest:
            raise ValueError("Pinned HF manifest differs")
        for name in FILES:
            info = (folder / name).stat()
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != identities[name]:
                raise ValueError("Source changed during upload")
        print(json.dumps({"event": "HF_SHA256_VERIFIED", "commit": revision,
                          "destination": destination, "files": records,
                          "source_retained": True}), flush=True)


if __name__ == "__main__":
    main()
