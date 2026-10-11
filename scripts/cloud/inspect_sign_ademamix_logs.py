#!/usr/bin/env python3
"""Read only Sign-AdEMAMix campaign metrics and diagnostic log tails."""
import json
from pathlib import Path

for root in [Path("/home/jovyan/dimativator/sign-ademamix-20261007"), Path("/home/jovyan/dimativator/sign-ademamix-precision-20261010"), Path("/home/jovyan/dimativator/sign-ademamix-clip1-20261011")]:
    print("ROOT", str(root), root.exists(), flush=True)
    for path in sorted(root.glob("*/*/metrics.jsonl")):
        latest = None
        validation = None
        final = None
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest = row
                if "val/loss" in row or "final-val/loss" in row:
                    validation = row
                if "final-val/loss" in row:
                    final = row
        print("SIGN_LOCAL_METRICS " + json.dumps({"path": str(path), "mtime": path.stat().st_mtime, "latest": latest, "validation": validation, "final": final}), flush=True)
        log = path.parent / "rank0.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                tail = stream.read().decode("utf-8", errors="replace")
            print("SIGN_LOG_TAIL " + json.dumps({"path": str(log), "tail": tail}), flush=True)

    for log in sorted(root.glob("*/*/rank0.log")):
        if not (log.parent / "metrics.jsonl").exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8000))
                tail = stream.read().decode("utf-8", errors="replace")
            print("SIGN_STARTUP_LOG " + json.dumps({"path": str(log), "tail": tail}), flush=True)
