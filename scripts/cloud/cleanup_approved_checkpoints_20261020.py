"""Delete only the five checkpoint directories approved on 2026-10-20.

Default is read-only. APPLY=1 repeats every check before unlinking exact files.
No recursive deletion, downloads, credentials, or checkpoint deserialization.
"""

import hashlib
import json
import os
from pathlib import Path
import stat
import time
import urllib.request


REVISION = "ba5df6e50532422704b1d447d58b83eeedb70f70"
FIGURE_ROOT = Path("/workspace-SR006.nfs3/dimativator/figure6a-adamw-fp8-retrain-20261005")
FIGURE = FIGURE_ROOT / "figure6a_correct_checkpoint_20261005/llama500M_adamw_fp8_states_1xC_2gpu_global128/ckpts/67911"
HORIZON_ROOT = Path("/home/jovyan/dimativator/horizon-adam-257m-lr-20261014")
HORIZON_GROUP = HORIZON_ROOT / "horizon_adam_257m_lr_20261014"
TARGETS = [FIGURE] + [
    HORIZON_GROUP / f"horizon_adam_257m_b8_h32_lr{lr}_wd1e-1_s0_attempt1/ckpts/latest"
    for lr in ("2e-3", "1e-3", "5e-4", "1e-4")
]
EXPECTED = {
    "main.pt": (3060216509, "580067125fd9f5cf11bb62fc072d95ff04a2556d58f77ae764a5c4f2cfcd82ab"),
    "worker_0.pt": (14765, "5456cfcfab1011651ba5d332ed789be6bfda14c87f0b5b7b7c72500e2da5f406"),
    "worker_1.pt": (14829, "f1a05dfe38220a480cd8c697687ff69c76416a18ef64e229c6048d1c2a02a2f9"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_path(path: Path) -> None:
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"Symlink refused: {component}")
    require(path.resolve(strict=True) == path, f"Non-canonical path: {path}")


def identity(path: Path) -> tuple:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"Not a regular file: {path}")
    require(info.st_nlink == 1, f"Hardlinked file refused: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def protected_files() -> dict:
    result = {}
    for root in (FIGURE_ROOT, HORIZON_ROOT):
        for parent, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if name != "ckpts"]
            for name in filenames:
                path = Path(parent) / name
                if name.endswith((".log", ".jsonl")):
                    checked_path(path.parent)
                    if path.is_symlink():
                        info = path.lstat()
                        result[str(path)] = ("symlink", os.readlink(path), info.st_ino, info.st_mtime_ns)
                    else:
                        result[str(path)] = identity(path)
    require(result, "No protected logs/metrics found")
    return result


def storage() -> dict:
    result = {}
    for mount in ("/home/jovyan", "/workspace-SR006.nfs3"):
        info = os.statvfs(mount)
        result[mount] = {"free_bytes": info.f_bavail * info.f_frsize, "total_bytes": info.f_blocks * info.f_frsize}
    return result


def main() -> None:
    if os.environ.get("VERIFY_ONLY", "0") == "1":
        receipt = Path("/home/jovyan/dimativator/approved-checkpoint-cleanup-20261020.json")
        checked_path(receipt)
        with receipt.open() as stream:
            report = json.load(stream)
        require(report["status"] == "verified_deleted", "Cleanup receipt not complete")
        require([item["path"] for item in report["targets"]] == [str(path) for path in TARGETS], "Receipt targets mismatch")
        for target in TARGETS:
            checked_path(target.parent)
            require(not os.path.lexists(target), f"Target still present: {target}")
        print("CLEANUP_VERIFIED=" + json.dumps({"storage": storage(), "receipt": report}), flush=True)
        return
    apply = os.environ.get("APPLY", "0") == "1"
    url = f"https://huggingface.co/api/models/DimaTivator/stage3_dense/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        remote = json.load(response)
    require(remote["sha"] == REVISION, "HF revision mismatch")
    entries = {item["rfilename"]: item for item in remote["siblings"]}
    for name, (size, sha256) in EXPECTED.items():
        entry = entries[f"adamw_fp8_states_fix/pre_decay/{name}"]
        require(entry["size"] == size and entry["lfs"]["size"] == size, f"HF size mismatch: {name}")
        require(entry["lfs"]["sha256"] == sha256, f"HF SHA256 mismatch: {name}")
    print(f"HF_REMOTE_VERIFIED revision={REVISION}", flush=True)
    protected = protected_files()
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "apply": apply, "hf_revision": REVISION, "before": storage(), "targets": []}
    audited = {}
    for target in TARGETS:
        checked_path(target)
        require(target.is_dir(), f"Missing checkpoint directory: {target}")
        names = {path.name for path in target.iterdir()}
        allowed = set(EXPECTED) if target == FIGURE else {"main.pt", "worker_0.pt"}
        require(names == allowed, f"Unexpected checkpoint contents: {target}: {sorted(names)}")
        files = []
        for name in sorted(names):
            path = target / name
            checked_path(path)
            before = identity(path)
            sha256 = digest(path)
            require(identity(path) == before, f"File changed during audit: {path}")
            if target == FIGURE:
                require((before[2], sha256) == EXPECTED[name], f"Local/HF mismatch: {path}")
            elif name == "main.pt":
                require(before[2] == 1302694921, f"Horizon checkpoint size changed: {path}")
            audited[path] = before
            files.append({"name": name, "bytes": before[2], "sha256": sha256})
        report["targets"].append({"path": str(target), "files": files, "bytes": sum(item["bytes"] for item in files)})
        print("CHECKPOINT_AUDITED " + json.dumps(report["targets"][-1]), flush=True)
    require(protected_files() == protected, "Logs/metrics changed during audit")
    # Validate every target again before the first destructive operation.
    for path, before in audited.items():
        checked_path(path)
        require(identity(path) == before, f"Checkpoint changed before deletion: {path}")
    if apply:
        receipt = Path("/home/jovyan/dimativator/approved-checkpoint-cleanup-20261020.json")
        require(not receipt.exists() and not receipt.is_symlink(), "Existing cleanup receipt; inspect before retry")
        checked_path(receipt.parent)
        with receipt.open("x") as stream:
            json.dump(report, stream, indent=2)
        for target in TARGETS:
            for path in sorted(path for path in audited if path.parent == target):
                require(identity(path) == audited[path], f"Changed before unlink: {path}")
                path.unlink()
            target.rmdir()
            require(not target.exists(), f"Deletion incomplete: {target}")
            print(f"DELETED_APPROVED_CHECKPOINT={target}", flush=True)
        require(protected_files() == protected, "Protected logs/metrics changed")
        report["after"] = storage()
        report["deleted_bytes"] = sum(item["bytes"] for item in report["targets"])
        report["protected_logs_metrics_count"] = len(protected)
        report["status"] = "verified_deleted"
        with receipt.open("w") as stream:
            json.dump(report, stream, indent=2)
        print(f"CLEANUP_RECEIPT={receipt}", flush=True)
    print("CLEANUP_RESULT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
