#!/usr/bin/env bash
set -euo pipefail
echo "INSPECT_UTC=$(date -u --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
python - <<'PY'
import json
from pathlib import Path

roots = {
    "bf16": Path("/home/jovyan/dimativator/500m-slimadam-precisions-20261002"),
    "fp8_act": Path("/workspace-SR006.nfs2/dimativator/500m-slimadam-precisions-20261002"),
    "fp8_full": Path("/home/jovyan/dimativator/500m-slimadam-precisions-20261002"),
    "w16a16g32_fp8_states": Path("/workspace-SR006.nfs2/dimativator/500m-slimadam-precisions-20261002"),
}
for precision in ("bf16", "fp8_act", "fp8_full", "w16a16g32_fp8_states"):
    root = roots[precision]
    for phase, group, suffix in (
        ("trunk", "2xChinchilla_500M_slimadam_wd1e-4_precisions_2gpu", "2xC"),
        ("decay", "1xChinchilla_decay_500M_slimadam_wd1e-4_precisions_2gpu", "1xC_decay"),
    ):
        run = f"llama500M_slim_adam_{precision}_wd1e-4_{suffix}_warmup2000_2gpu"
        path = root / group / run / "metrics.jsonl"
        latest = None
        last_eval = None
        if path.is_file():
            with path.open() as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    latest = item
                    if "val/loss" in item:
                        last_eval = item
        print("PRECISION_STATUS", json.dumps({
            "precision": precision, "phase": phase,
            "latest": latest, "last_eval": last_eval,
        }, sort_keys=True), flush=True)
    ckpt = root / "2xChinchilla_500M_slimadam_wd1e-4_precisions_2gpu" / (
        f"llama500M_slim_adam_{precision}_wd1e-4_2xC_warmup2000_2gpu"
    ) / "ckpts/67911"
    print("CLOUD_CHECKPOINT", precision, ckpt.exists(), flush=True)
    files = {}
    for name in ("main.pt", "worker_0.pt", "worker_1.pt"):
        path = ckpt / name
        files[name] = path.stat().st_size if path.is_file() else None
    print("CLOUD_CHECKPOINT_FILES", precision, json.dumps(files, sort_keys=True), flush=True)

    log = root / "logs" / (
        f"llama500M_slim_adam_{precision}_wd1e-4_2xC_warmup2000_2gpu_full_rank0.log"
    )
    markers = (
        b"BRAIN_LAB_CHECKPOINT_VERIFIED",
        b"Uploaded intermediate checkpoint",
        b"Deleted local intermediate checkpoint",
        b"WARNING: failed to upload intermediate checkpoint",
    )
    events = []
    if log.is_file():
        with log.open("rb") as stream:
            for line in stream:
                for marker in markers:
                    offset = line.find(marker)
                    if offset >= 0:
                        detail = line[offset:offset + 400].decode("utf-8", errors="replace")
                        events.append(detail.split("\r", 1)[0].strip())
    print("CHECKPOINT_RELAY_EVENTS", precision, json.dumps(events[-12:]), flush=True)
PY
