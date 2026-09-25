#!/usr/bin/env bash
set -euo pipefail

echo "INSPECT_DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3

group=500M_time_matched_muon_fp8_1xC_20260925
for optimizer in slim_adam frugal adamw; do
    case "${optimizer}" in
        slim_adam)
            root=/home/jovyan/dimativator/500m-time-matched-20260925/slim_adam
            experiment=llama500M_slim_adam_bf16_time_matched_4gpu
            ;;
        frugal)
            root=/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/frugal
            experiment=llama500M_frugal_bf16_time_matched_4gpu
            ;;
        adamw)
            root=/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/adamw
            experiment=llama500M_adamw_fp8_time_matched_1gpu
            ;;
    esac
    echo "OPTIMIZER=${optimizer}"
    if [[ "${optimizer}" == adamw ]]; then
        log=${root}/logs/${experiment}_full.log
    else
        log=${root}/logs/${experiment}_rank0.log
    fi
    metrics=${root}/${group}/${experiment}/metrics.jsonl
    if [[ -f "${log}" ]]; then
        echo "LOG=${log}"
        tail -n 28 "${log}"
    else
        echo "LOG=missing"
    fi
    if [[ -f "${metrics}" ]]; then
        echo "METRICS=${metrics} LINES=$(wc -l < "${metrics}")"
        tail -n 8 "${metrics}"
    else
        echo "METRICS=missing"
    fi
    latest=${root}/${group}/${experiment}/ckpts/latest
    if [[ -f "${latest}/main.pt" ]]; then
        stat -c 'LATEST_MAIN_BYTES=%s MODIFIED=%y' "${latest}/main.pt"
        for worker in "${latest}"/worker_*.pt; do
            if [[ -f "${worker}" ]]; then
                stat -c 'LATEST_WORKER=%n BYTES=%s MODIFIED=%y' "${worker}"
            fi
        done
    fi
done
