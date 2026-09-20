#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-bf16-500m-1xc-wd1e-4-4gpu-20260920}
EXPERIMENT_NAME=llama500M_frugal_adamw_bf16_wd1e-4_1xC_4gpu
WANDB_GROUP=1xChinchilla_500M_frugal_bf16_wd1e-4_4gpu_cloud
METRICS_JSONL=${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}/metrics.jsonl

echo "DATE=$(date --iso-8601=seconds)"
echo "METRICS_JSONL=${METRICS_JSONL}"
if [[ "${SHOW_GPU_START:-0}" == "1" ]]; then
    rank0_log=${RESULTS_DIR}/logs/${EXPERIMENT_NAME}_rank0.log
    if [[ -f "${rank0_log}" ]]; then
        echo "GPU_START_SNAPSHOT"
        sed -n '1,30p' "${rank0_log}"
    fi
fi
if [[ -f "${METRICS_JSONL}" ]]; then
    echo "METRIC_LINES=$(wc -l < "${METRICS_JSONL}")"
    tail -n 10 "${METRICS_JSONL}"
else
    echo "METRICS_JSONL=missing"
fi

for rank in 0 1 2 3; do
    log_file=${RESULTS_DIR}/logs/${EXPERIMENT_NAME}_rank${rank}.log
    if [[ -f "${log_file}" ]]; then
        echo "RANK_LOG=${log_file}"
        tail -n 12 "${log_file}"
    else
        echo "RANK_LOG_${rank}=missing"
    fi
done

if [[ -d "${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}/ckpts" ]]; then
    echo "UNEXPECTED_CHECKPOINT_DIRECTORY=present"
else
    echo "CHECKPOINT_DIRECTORY=absent"
fi
