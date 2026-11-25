"""Cloud.ru four-rank runner for each of the two 1xC validation-size experiments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

NFS = Path("/workspace-SR006.nfs3")
ROOT = NFS / "dimativator/validation-size-257m-20261125"
PACKED = NFS / "dimativator/fineweb-h200-packed"
VALIDATION = NFS / "dimativator/fineweb-validation-32768-20261125"
PATHS = ["/", "/tmp", "/home/jovyan", "/home/jovyan/mlsub-logs", "/workspace-SR006.nfs2", str(NFS)]
METHODS = ("adamw", "muon")
RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
REPO = Path(__file__).resolve().parents[2]
GROUP = "val-ci-257m"


class StorageEmergency(RuntimeError):
    pass


def space(extra=0):
    for mount in (NFS, Path(PATHS[-2])):
        if not mount.resolve(strict=True).is_mount():
            raise RuntimeError(f"Missing mount {mount}")
    values = {p: shutil.disk_usage(p).free for p in PATHS}
    ok = all(v > 5_000_000_000 + (extra if p == str(NFS) else 0) for p,v in values.items())
    status = dict(timestamp=time.time(), available_bytes=values, extra_bytes=extra, ok=ok)
    print("VAL_CI_DISK", json.dumps(status), flush=True)
    if min(values.values()) <= 5_000_000_000:
        print("КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU", values, flush=True)
        raise StorageEmergency(str(values))
    if not ok:
        raise RuntimeError("Cloud.ru storage reserve unavailable")
    return status


def checkpoints():
    return [ROOT / GROUP / method / "ckpts/39250" for method in METHODS]


def remove_owned(path):
    if path not in checkpoints() or path.resolve() != path:
        raise RuntimeError(f"Unexpected checkpoint path {path}")
    known = {"main.pt", *(f"worker_{i}.pt" for i in range(4))}
    if any(f.name not in known or f.is_symlink() or not f.is_file() for f in path.iterdir()):
        raise RuntimeError("Unexpected checkpoint contents")
    shutil.rmtree(path)
    print("VAL_CI_DELETED", path, flush=True)


def finish():
    with (ROOT / "cleanup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if all((ROOT / GROUP / m / "bootstrap.json").is_file() for m in METHODS):
            for path in checkpoints():
                if path.exists():
                    remove_owned(path)
            space()
            print("VAL_CI_CLEANUP_COMPLETE", flush=True)


def main():
    space(6_500_000_000)
    meta = json.loads((PACKED / "packed_metadata.json").read_text())
    if meta["world_size"] != 2 or meta["batch_size"] != 16:
        raise ValueError("Unexpected packed layout")
    digest = hashlib.sha256((VALIDATION / "validation.uint16").read_bytes()).hexdigest()
    if digest != "bc23cac94158ff98af16de2b31bf113752b0e5de1891786a27e9d279ef98c232":
        raise ValueError("Validation digest mismatch")
    if sys.argv[1:] == ["preflight"]:
        print("VAL_CI_PREFLIGHT_OK", flush=True)
        return
    method = sys.argv[1]
    if method not in METHODS or int(os.environ["OMPI_COMM_WORLD_SIZE"]) != 4:
        raise ValueError("Expected AdamW or Muon and four MPI ranks")
    if ROOT.resolve() != ROOT:
        raise RuntimeError("Campaign root contains a symlink")
    ROOT.mkdir(exist_ok=True, parents=True)
    rendezvous = ROOT / (method + "-address")
    if RANK == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = os.environ.get("MASTER_ADDR", sock.getsockname()[0])
        tmp = rendezvous.with_suffix(".tmp")
        with tmp.open("x") as f:
            f.write(address)
        if rendezvous.exists():
            raise RuntimeError("Run already initialized")
        tmp.rename(rendezvous)
    while not rendezvous.exists():
        time.sleep(1)
    name = f"valci257m-{method}-20261125"
    tmp = Path("/tmp") / name
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, RANK=str(RANK), WORLD_SIZE="4",
               LOCAL_RANK=os.getenv("OMPI_COMM_WORLD_LOCAL_RANK","0"),
               MASTER_ADDR=rendezvous.read_text(), MASTER_PORT="29500" if method=="adamw" else "29501",
               CLOUD_DISK_GUARD_ROOT=str(ROOT), CLOUD_DISK_GUARD_PATHS=json.dumps(PATHS),
               FINAL_VALIDATION_32768="1", EXPECTED_GLOBAL_BATCH="128", PYTHONUNBUFFERED="1",
               TRITON_CACHE_DIR=str(tmp/"triton"), WANDB_RUN_ID=name, WANDB_RESUME="never",
               WANDB_BASE_URL="https://wandb-radfan.ru", WANDB_ENTITY="andrey", WANDB_DIR=str(ROOT))
    cmd=[sys.executable, "src/main.py", "--experiment-name",method,"--distributed-backend","nccl",
         "--seed","0","--data-seed","1337","--dataset","fineweb","--datasets-dir",str(PACKED),
         "--streaming","--workers","8","--sequence-length","1024","--fineweb-replay-world-size","1",
         "--model","llama","--n-layer","12","--n-embd","1024","--n-head","8","--multiple-of","256",
         "--dtype","bfloat16","--model-parameter-dtype","float32","--dropout","0",
         "--opt",method,"--lr","0.001","--weight-decay","0.1","--beta1","0.9","--beta2","0.99",
         "--grad-clip","1","--eps","1e-7","--lite-ns-steps","6","--lite-muon-theta","0.95",
         "--batch-size","8","--acc-steps","16","--iterations","39250","--scheduler","wsd",
         "--warmup-steps","2000","--wsd-fract-decay","0.1","--wsd-final-lr-scale","0","--decay-type","cosine",
         "--latest-ckpt-interval","0","--permanent-ckpt-interval","39250",
         "--results-base-folder",str(ROOT),"--wandb","--wandb-group",GROUP,"--wandb-project","fp8-pretrain",
         "--eval-batches","32","--eval-batch-size","32","--eval-interval","500","--log-interval","50",
         "--metrics-jsonl",str(ROOT / GROUP / method / "metrics.jsonl")]
    print("VAL_CI_COMMAND",json.dumps(cmd),flush=True)
    child = subprocess.Popen(cmd,cwd=REPO,env=env,start_new_session=True)
    try:
        while child.poll() is None:
            if (ROOT / "storage-stop").exists():
                raise RuntimeError("Storage emergency in this campaign")
            space()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
        if child.returncode:
            raise RuntimeError(f"Training child exited {child.returncode}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    if RANK == 0:
        finish()


if __name__ == "__main__":
    try:
        main()
    except StorageEmergency:
        if ROOT.is_dir() and ROOT.resolve() == ROOT:
            with (ROOT / "cleanup.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                (ROOT / "storage-stop").touch()
                owned = sorted((p for p in checkpoints() if p.is_dir()), key=lambda p:p.stat().st_mtime)
                for path in owned:
                    before = shutil.disk_usage(NFS).free
                    if before > 5_000_000_000:
                        break
                    remove_owned(path)
                    print("VAL_CI_EMERGENCY_BYTES", before, shutil.disk_usage(NFS).free, flush=True)
        raise
