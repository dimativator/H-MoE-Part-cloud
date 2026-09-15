#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-20260912}
GROUP=2xChinchilla_500M_frugal_muon_2gpu_cloud
EXPERIMENTS=(
    llama500M_frugal_muon_adamw_bf16_2xC_2gpu
    llama500M_frugal_muon_adamw_fp8_full_2xC_2gpu
)

echo "INSPECT_ROOT=${RESULTS_DIR}"
for experiment in "${EXPERIMENTS[@]}"; do
    experiment_dir="${RESULTS_DIR}/${GROUP}/${experiment}"
    metrics_file="${experiment_dir}/metrics.jsonl"
    echo "EXPERIMENT=${experiment}"
    if [[ -f "${metrics_file}" ]]; then
        tail -n 256 "${metrics_file}" \
            | grep '"event": "train"' \
            | tail -n 1 \
            | sed 's/^/LATEST_TRAIN=/' || true
        tail -n 256 "${metrics_file}" \
            | grep '"event": "validation"' \
            | tail -n 1 \
            | sed 's/^/LATEST_VALIDATION=/' || true
    else
        echo "METRICS_MISSING=${metrics_file}"
    fi

    for checkpoint_file in main.pt worker_0.pt worker_1.pt; do
        checkpoint_path="${experiment_dir}/ckpts/latest/${checkpoint_file}"
        if [[ -f "${checkpoint_path}" ]]; then
            stat --printf='LATEST_CHECKPOINT=%s %Y %n\n' "${checkpoint_path}"
        fi
    done
done
