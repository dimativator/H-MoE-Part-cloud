"""Inventory and synchronize existing W&B data. Never starts training."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import zlib

BASE_URL = "https://wandb-radfan.ru"
LOGS = "/home/jovyan/shares/SR006.nfs3/mlsub-logs"
RESERVE = 5_000_000_000
PATHS = ("/workspace-SR006.nfs3", LOGS, "/home/jovyan", "/tmp", "/")
PRUNE = {"ckpts", "checkpoints", ".git", ".cache", "datasets", "artifacts", "media"}
FAMILIES = ("galore", "apollo", "frugal", "slim_adam", "slimadam")


def emit(kind: str, value: object) -> None:
    text = json.dumps(value, sort_keys=True)
    key = os.environ.get("WANDB_API_KEY")
    if key:
        text = text.replace(key, "[REDACTED]")
    print(kind + "=" + text, flush=True)


def guard(pending: int = 0) -> dict:
    available = {p: os.statvfs(p).f_bavail * os.statvfs(p).f_frsize for p in PATHS}
    value = {"timestamp": time.time(), "available_bytes": available, "logs_dir": LOGS,
             "ready": all(v > RESERVE for v in available.values()), "pending_tmp_bytes": pending}
    emit("SPACE_SNAPSHOT", value)
    if not value["ready"] or available["/tmp"] - pending <= RESERVE:
        raise RuntimeError("DISK_RESERVE_BLOCKED")
    return value


def watchdog(stop: threading.Event) -> None:
    while not stop.wait(30):
        try:
            guard()
        except Exception as exc:
            emit("DISK_GUARD_STOP", {"error": type(exc).__name__})
            os._exit(72)


def walk(root: Path):
    if root.is_symlink() or not root.is_dir():
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        path = Path(current)
        dirs[:] = [d for d in dirs if d not in PRUNE and not (path / d).is_symlink()
                   and len((path / d).relative_to(root).parts) <= 8]
        yield path, files


def roots() -> list[Path]:
    selected = []
    for parent in ("/home/jovyan/dimativator", "/workspace-SR006.nfs2/dimativator",
                   "/workspace-SR006.nfs3/dimativator"):
        p = Path(parent)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_dir() and not child.is_symlink() and any(
                    word in child.name.lower() for word in ("500m", "projected", "wd-tun")
                ):
                    selected.append(child)
    return selected


def record_config(updates) -> dict:
    result = {}
    for item in updates:
        key = item.key or ".".join(item.nested_key)
        if any(word in key.lower() for word in ("api_key", "password", "access_token")):
            continue
        try:
            result[key] = json.loads(item.value_json)
        except (ValueError, TypeError):
            pass
    return result


def binary_header(path: Path) -> dict:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    reader = DataStore()
    reader.open_for_scan(str(path))
    result = {"id": path.name[4:-6], "binary": str(path), "bytes": path.stat().st_size}
    try:
        for _ in range(200):
            data = reader.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("run"):
                run = record.run
                result.update(id=run.run_id, name=run.display_name, group=run.run_group,
                              entity=run.entity, project=run.project,
                              config=record_config(run.config.update))
                break
    finally:
        reader.close()
    return result


def metric_inventory(path: Path) -> dict | None:
    last = None
    final = None
    count = 0
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            checksum.update(raw)
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            last = row
            count += 1
            if "final-val/loss" in row:
                final = row
    if not last:
        return None
    name = last.get("experiment", str(path))
    if not any(word in name.lower() for word in FAMILIES):
        return None
    return {"path": str(path), "experiment": name, "rows": count,
            "sha256": checksum.hexdigest(), "last_iter": last.get("iter"), "final": final}


def inventory(target_ids: set[str]) -> None:
    import wandb

    emit("SDK", {"version": wandb.__version__})
    for root in roots():
        emit("ROOT", str(root))
        for path, files in walk(root):
            if "metrics.jsonl" in files:
                value = metric_inventory(path / "metrics.jsonl")
                if value:
                    emit("METRICS", value)
            for filename in files:
                if filename.startswith("run-") and filename.endswith(".wandb"):
                    file = path / filename
                    if file.is_symlink():
                        continue
                    try:
                        header = binary_header(file)
                        name = header.get("name", "").lower()
                        if header["id"] in target_ids or any(word in name for word in FAMILIES):
                            emit("OFFLINE", header)
                    except Exception as exc:
                        emit("BINARY_ERROR", {"path": str(file), "error": type(exc).__name__})


def sync_binary(target: dict) -> None:
    """Sync a temporary copy only, and reject all artifact-bearing records."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    binary = Path(target["binary"])
    if binary.is_symlink() or not binary.is_file():
        raise RuntimeError("Missing regular source binary")
    if binary.name != "run-" + target["id"] + ".wandb":
        raise RuntimeError("Source ID mismatch")
    reader = DataStore()
    reader.open_for_scan(str(binary))
    try:
        while True:
            data = reader.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("artifact"):
                raise RuntimeError("Artifact-bearing binary requires metrics-only recovery")
    finally:
        reader.close()
    guard(binary.stat().st_size + 100_000_000)
    with tempfile.TemporaryDirectory(prefix="wandb-table-sync-", dir="/tmp") as tmp:
        run_dir = Path(tmp) / binary.parent.name
        run_dir.mkdir()
        shutil.copyfile(binary, run_dir / binary.name)
        # Native binaries contain history/config. Never copy checkpoint/media files.
        files = run_dir / "files"
        files.mkdir()
        for name in ("config.yaml", "wandb-summary.json", "wandb-metadata.json"):
            source = binary.parent / "files" / name
            if source.is_file() and not source.is_symlink():
                shutil.copyfile(source, files / name)
        guard()
        result = subprocess.run(["wandb", "sync", "--entity", "andrey", "--project",
                                 "fp8-pretrain", str(run_dir)], cwd=tmp, capture_output=True,
                                text=True, timeout=600)
        emit("SYNC_RESULT", {"id": target["id"], "returncode": result.returncode,
                             "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        if result.returncode:
            raise RuntimeError("Native W&B sync failed")


def main() -> None:
    os.environ.update(WANDB_BASE_URL=BASE_URL, WANDB_DIR="/tmp", WANDB_CACHE_DIR="/tmp/wandb-sync-cache",
                      WANDB_DATA_DIR="/tmp/wandb-sync-data", WANDB_CONSOLE="off", WANDB_DISABLE_CODE="true")
    if os.environ.get("VERIFIED_BOOTSTRAP_LOGS_DIR") != LOGS:
        raise RuntimeError("NFS3 bootstrap policy proof missing")
    guard()
    for p in ("/workspace-SR006.nfs2", "/home/jovyan/shares/SR006.nfs2"):
        if Path(p).exists():
            s = os.statvfs(p)
            emit("NFS2_READ_ONLY", {"path": p, "available_bytes": s.f_bavail * s.f_frsize,
                                    "used_for_writes": False})
    raw_payload = base64.b64decode(os.environ["SYNC_TARGETS_B64"])
    if os.environ.get("SYNC_TARGETS_COMPRESSED") == "1":
        raw_payload = zlib.decompress(raw_payload)
    payload = json.loads(raw_payload)
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(stop,), daemon=True).start()
    try:
        if os.environ.get("SYNC_MODE", "inventory") == "inventory":
            inventory(set(payload.get("run_ids", [])))
        else:
            failures = []
            for target in payload["targets"]:
                try:
                    sync_binary(target)
                except Exception as exc:
                    failures.append(target["id"])
                    emit("SYNC_ERROR", {"id": target["id"], "error": str(exc)})
            if failures:
                emit("SYNC_FAILURES", failures)
                raise RuntimeError("Some existing runs could not be synchronized")
    finally:
        stop.set()
    guard()
    emit("COMPLETE", {"mode": os.environ.get("SYNC_MODE", "inventory")})


if __name__ == "__main__":
    main()
