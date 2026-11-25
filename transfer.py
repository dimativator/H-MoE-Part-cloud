import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import urllib.request

nfs = Path("/workspace-SR006.nfs3")
home = Path("/home/jovyan")
nfs2 = Path("/workspace-SR006.nfs2")
logs = home / "mlsub-logs"
dest = nfs / "dimativator" / "fineweb-validation-32768-20261125"
files = {
    "validation.uint16": (67174400, "bc23cac94158ff98af16de2b31bf113752b0e5de1891786a27e9d279ef98c232"),
    "manifest.json": (46333, "abdf437800fbd4fa00915bc48d5c9778748850bbb7b4199e5c569c7fbfe822c3"),
}
reserve = 5_000_000_000
total = sum(size for size, _ in files.values())

def measure(phase, planned):
    for mount in (nfs, nfs2):
        if not mount.resolve(strict=True).is_mount():
            raise RuntimeError(f"Missing mount {mount}")
    for path in (Path("/"), Path("/tmp"), home, logs, nfs2, nfs):
        size = shutil.disk_usage(path)
        allowance = planned if path.stat().st_dev == nfs.stat().st_dev else 0
        ok = size.free > reserve + allowance + 1048576
        print(json.dumps(dict(phase=phase, time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              path=str(path),total_bytes=size.total,used_bytes=size.used,available_bytes=size.free,
                              planned_bytes=allowance,status="OK" if ok else "STOP")), flush=True)
        if not ok:
            raise RuntimeError(f"Insufficient reserve on {path}")

def watch():
    while True:
        time.sleep(30)
        try:
            measure("monitor", 0)
        except Exception as e:
            print("TRANSFER_STOP", repr(e), flush=True)
            os._exit(1)

def verify(folder):
    for name, (size, digest) in files.items():
        path = folder / name
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Verification failed for {path}")

def run():
    measure("before", total)
    threading.Thread(target=watch, daemon=True).start()
    if not dest.exists():
        with tempfile.TemporaryDirectory(prefix=".validation-copy-",dir=dest.parent) as tmp:
            stage = Path(tmp) / "dataset"
            stage.mkdir()
            for name,(size, _) in files.items():
                measure("before_file", size)
                request = urllib.request.Request(os.environ["SOURCE_URL"] + "/" + name,
                            headers={"Authorization":"Bearer " + os.environ["SOURCE_KEY"]})
                with urllib.request.urlopen(request, timeout=45) as source, (stage / name).open("xb") as target:
                    if int(source.headers.get("Content-Length",-1)) != size:
                        raise RuntimeError("Source length mismatch")
                    remaining = size
                    while remaining:
                        data = source.read(min(remaining, 1024*1024))
                        if not data:
                            raise EOFError(name)
                        target.write(data)
                        remaining -= len(data)
                    target.flush()
                    os.fsync(target.fileno())
            verify(stage)
            stage.rename(dest)
    verify(dest)
    measure("after", 0)
    print("TRANSFER_VERIFIED", dest, json.dumps(files), flush=True)

if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print("TRANSFER_FAILED", type(exc).__name__, str(exc), flush=True)
        sys.exit(1)
