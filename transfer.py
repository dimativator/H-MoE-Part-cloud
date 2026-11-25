import base64
import subprocess
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
            link_parts = sorted(k for k in os.environ if k.startswith("LINK_PART_"))
            link = "".join(os.environ[k] for k in link_parts)
            url = base64.urlsafe_b64decode(link + "=" * (-len(link) % 4)).decode()
            with urllib.request.urlopen(url, timeout=60) as source:
                ciphertext = source.read(total + 1)
            if len(ciphertext) != total:
                raise RuntimeError("Source length mismatch")
            plain = subprocess.run(["openssl", "enc", "-d", "-aes-256-ctr",
                                    "-K", os.environ["SOURCE_KEY"], "-iv", os.environ["SOURCE_IV"]],
                                   input=ciphertext, stdout=subprocess.PIPE, check=True).stdout
            offset = 0
            for name, (size, expected_hash) in files.items():
                payload = plain[offset:offset+size]
                if hashlib.sha256(payload).hexdigest() != expected_hash:
                    raise RuntimeError("Source digest mismatch")
                measure("before_file", size)
                with (stage / name).open("xb") as target:
                    target.write(payload)
                    target.flush()
                    os.fsync(target.fileno())
                offset += size
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
