#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-20260912}
GROUP=2xChinchilla_500M_frugal_muon_2gpu_cloud
EXPERIMENTS=(
    llama500M_frugal_muon_adamw_fp8_act_2xC_2gpu
    llama500M_frugal_muon_adamw_bf16_fp8_states_2xC_2gpu
)

for experiment in "${EXPERIMENTS[@]}"; do
    experiment_dir="${RESULTS_DIR}/${GROUP}/${experiment}"
    echo "EXPERIMENT=${experiment}"
    for checkpoint_file in main.pt worker_0.pt worker_1.pt; do
        checkpoint_path="${experiment_dir}/ckpts/latest/${checkpoint_file}"
        if [[ -f "${checkpoint_path}" ]]; then
            stat --printf='LATEST_CHECKPOINT=%s %Y %n\n' "${checkpoint_path}"
        else
            echo "LATEST_CHECKPOINT_MISSING=${checkpoint_path}"
        fi
    done
done
