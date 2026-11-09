"""Read-only storage and runtime diagnostics; never writes model checkpoints."""

import importlib.metadata
import json
import os
import time


def snapshot() -> dict:
    records = []
    paths = ("/", "/tmp", "/home/jovyan", "/workspace-SR006.nfs3",
             "/home/jovyan/shares/SR006.nfs3/mlsub-logs",
             "/workspace-SR006.nfs2")
    for path in paths:
        try:
            stat = os.statvfs(path)
            records.append(dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize,
                                device=os.stat(path).st_dev,
                                writable_by_campaign="nfs2" not in path))
        except OSError as error:
            records.append(dict(path=path, error=str(error),
                                writable_by_campaign="nfs2" not in path))
    return dict(timestamp=time.time(), records=records,
                reserve_bytes=5_000_000_000, checkpoints=False)


if __name__ == "__main__":
    print("REBUTTAL_STORAGE=" + json.dumps(snapshot()), flush=True)
    versions = {}
    for name in ("torch", "transformer-engine", "triton", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    print("REBUTTAL_RUNTIME=" + json.dumps(versions), flush=True)
