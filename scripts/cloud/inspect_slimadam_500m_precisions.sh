#!/usr/bin/env bash
set -euo pipefail
readonly root=/home/jovyan/dimativator/500m-slimadam-precisions-20261002
echo "INSPECT_UTC=$(date -u --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
ROOT="${root}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
for precision in ("bf16", "fp8_act", "fp8_full", "w16a16g32_fp8_states"):
    for phase, group, suffix in (
        ("trunk", "2xChinchilla_500M_slimadam_wd1e-4_precisions_4gpu", "2xC"),
        ("decay", "1xChinchilla_decay_500M_slimadam_wd1e-4_precisions_4gpu", "1xC_decay"),
    ):
        run = f"llama500M_slim_adam_{precision}_wd1e-4_{suffix}_warmup2000_4gpu"
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
                    if "val_loss" in item:
                        last_eval = item
        print("PRECISION_STATUS", json.dumps({
            "precision": precision, "phase": phase,
            "latest": latest, "last_eval": last_eval,
        }, sort_keys=True), flush=True)
    ckpt = root / "2xChinchilla_500M_slimadam_wd1e-4_precisions_4gpu" / (
        f"llama500M_slim_adam_{precision}_wd1e-4_2xC_warmup2000_4gpu"
    ) / "ckpts/67911"
    print("CLOUD_CHECKPOINT", precision, ckpt.exists(), flush=True)
PY
