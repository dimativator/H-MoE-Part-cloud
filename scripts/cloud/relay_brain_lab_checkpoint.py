"""Transfer a verified intermediate checkpoint between Cloud.ru and brain_lab."""

import argparse
import base64
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import uuid


HOST = "proxy2.cod.phystech.edu"
PORT = "10210"
USER = "dimativator"
HOST_KEY = (
    "[proxy2.cod.phystech.edu]:10210 ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIIHHLCvWHzxMi+m7XnSqdhq3qWPOmGt+88zaEcWPDTui"
)
REMOTE_ROOT = "/home/dimativator/checkpoints/500m-slimadam-precisions-20261002"
FILES = ("main.pt", "worker_0.pt", "worker_1.pt", "worker_2.pt", "worker_3.pt")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args, *, capture=False):
    return subprocess.run(args, check=True, text=True, capture_output=capture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=("upload", "download"))
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("remote_name")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.remote_name):
        raise ValueError("Unsafe remote checkpoint name")
    secret = os.environ.get("BRAIN_LAB_RELAY_KEY_B64")
    if not secret:
        raise RuntimeError("BRAIN_LAB_RELAY_KEY_B64 is required")

    with tempfile.TemporaryDirectory(prefix="brain-lab-relay-") as temporary:
        private_key = Path(temporary) / "id_ed25519"
        known_hosts = Path(temporary) / "known_hosts"
        private_key.write_bytes(base64.b64decode(secret, validate=True))
        private_key.chmod(0o600)
        known_hosts.write_text(HOST_KEY + "\n")
        options = [
            "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}", "-i", str(private_key),
        ]
        ssh = ["ssh", *options, "-p", PORT, f"{USER}@{HOST}"]
        scp = ["scp", "-q", *options, "-P", PORT]
        final = f"{REMOTE_ROOT}/{args.remote_name}"
        if args.direction == "upload":
            if not args.local_dir.is_dir():
                raise FileNotFoundError(args.local_dir)
            hashes = {}
            for name in FILES:
                path = args.local_dir / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Incomplete checkpoint: {path}")
                hashes[name] = checksum(path)
            staging = f"{final}.upload-{uuid.uuid4().hex}"
            run([*ssh, f"mkdir -p {shlex.quote(REMOTE_ROOT)} && test ! -e {shlex.quote(final)}"])
            run([*scp, "-r", str(args.local_dir), f"{USER}@{HOST}:{staging}"])
            remote = staging
        else:
            if args.local_dir.exists():
                raise FileExistsError(args.local_dir)
            run([*ssh, f"test -d {shlex.quote(final)}"])
            args.local_dir.parent.mkdir(parents=True, exist_ok=True)
            run([*scp, "-r", f"{USER}@{HOST}:{final}", str(args.local_dir)])
            hashes = {name: checksum(args.local_dir / name) for name in FILES}
            remote = final

        for name, expected in hashes.items():
            actual = run(
                [*ssh, f"sha256sum {shlex.quote(remote + '/' + name)}"], capture=True
            ).stdout.split()[0]
            if actual != expected:
                raise RuntimeError(f"Checkpoint hash mismatch for {name}")
        if args.direction == "upload":
            run([*ssh, f"test ! -e {shlex.quote(final)} && mv {shlex.quote(staging)} {shlex.quote(final)}"])
        print(f"BRAIN_LAB_CHECKPOINT_VERIFIED direction={args.direction} name={args.remote_name}", flush=True)


if __name__ == "__main__":
    main()
