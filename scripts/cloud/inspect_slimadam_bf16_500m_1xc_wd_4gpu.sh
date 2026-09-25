#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/slimadam-bf16-500m-1xc-wd-sweep-20260925}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
WANDB_GROUP=1xChinchilla_500M_slimadam_bf16_wd_sweep_4gpu_cloud

date --iso-8601=seconds
df -h /home/jovyan /workspace-SR006.nfs3
if [[ -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "PACKED_FINEWEB=present"
else
    echo "PACKED_FINEWEB=missing"
fi
if [[ -f "${HOME}/.netrc" ]]; then
    echo "NETRC=present"
else
    echo "NETRC=missing"
fi
for wd in 1e-2 1e-3 1e-4; do
    experiment="llama500M_slim_adam_bf16_wd${wd}_1xC_4gpu"
    metrics="${RESULTS_DIR}/${WANDB_GROUP}/${experiment}/metrics.jsonl"
    rank0="${RESULTS_DIR}/logs/${experiment}_rank0.log"
    echo "EXPERIMENT=${experiment}"
    if [[ -f "${metrics}" ]]; then
        echo "METRIC_LINES=$(wc -l < "${metrics}")"
        tail -n 8 "${metrics}"
    else
        echo "METRICS_JSONL=missing"
    fi
    if [[ -f "${rank0}" ]]; then
        tail -n 16 "${rank0}"
    else
        echo "RANK0_LOG=missing"
    fi
    if [[ -d "${RESULTS_DIR}/${WANDB_GROUP}/${experiment}/ckpts" ]]; then
        echo "UNEXPECTED_CHECKPOINT_DIRECTORY=present"
    else
        echo "CHECKPOINT_DIRECTORY=absent"
    fi
done
