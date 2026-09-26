#!/usr/bin/env bash
set -euo pipefail

echo "INSPECT_DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3

group=2xChinchilla_500M_slimadam_bf16_wd_sweep_2gpu_cloud
decay_group=1xChinchilla_decay_500M_slimadam_bf16_wd_sweep_2gpu_cloud
for wd in 1e-2 1e-3 1e-4; do
    if [[ "${wd}" == 1e-4 ]]; then
        root=/workspace-SR006.nfs2/dimativator/500m-slimadam-bf16-wd-2xc-20260926
    else
        root=/home/jovyan/dimativator/500m-slimadam-bf16-wd-2xc-20260926
    fi
    trunk=llama500M_slim_adam_bf16_wd${wd}_2xC_2gpu
    decay=llama500M_slim_adam_bf16_wd${wd}_1xC_decay_2gpu
    echo "WD=${wd} ROOT=${root}"
    log=${root}/logs/${trunk}_full_rank0.log
    if [[ -f "${log}" ]]; then
        tr '\r' '\n' < "${log}" |
            grep -E 'RUN_START|ENVIRONMENT_AND_DATA_CHECK|Train: Iter=|Eval: Iter=|SLIMADAM_|Traceback|Error|RuntimeError' |
            tail -n 14 || true
    else
        echo "LOG=missing"
    fi
    for spec in "${group}/${trunk}" "${decay_group}/${decay}"; do
        metrics=${root}/${spec}/metrics.jsonl
        if [[ -f "${metrics}" ]]; then
            echo "METRICS=${metrics} LINES=$(wc -l < "${metrics}")"
            tail -n 3 "${metrics}"
            python - "${metrics}" "${wd}" "${spec}" <<'PY'
import json
import sys

metrics, wd, spec = sys.argv[1:]
points = {}
with open(metrics, encoding="utf-8") as source:
    for line in source:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A training process may still be appending its final line.
        step = row.get("iter")
        loss = row.get("val/loss")
        if isinstance(step, int) and isinstance(loss, (int, float)):
            points[step] = loss
print("CURVE_JSON=" + json.dumps({
    "wd": wd,
    "phase": "decay" if "decay" in spec else "trunk",
    "points": sorted(points.items()),
}, separators=(",", ":")))
PY
        else
            echo "METRICS=${metrics} missing"
        fi
    done
    ckpt=${root}/${group}/${trunk}/ckpts/67911
    if [[ -d "${ckpt}" ]]; then
        for file in main.pt worker_0.pt worker_1.pt; do
            if [[ -f "${ckpt}/${file}" ]]; then
                stat -c 'PRE_DECAY=%n BYTES=%s MODIFIED=%y' "${ckpt}/${file}"
            fi
        done
    fi
done
