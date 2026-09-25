#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/slimadam-bf16-500m-1xc-wd-sweep-20260925}
WANDB_GROUP=1xChinchilla_500M_slimadam_bf16_wd_sweep_4gpu_cloud
export WANDB_BASE_URL=https://wandb-radfan.ru
export WANDB_ENTITY=andrey
if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
    echo "W&B credentials are missing; cannot sync" >&2
    exit 2
fi

for wd in 1e-2 1e-3 1e-4; do
    experiment="llama500M_slim_adam_bf16_wd${wd}_1xC_4gpu"
    log_file="${RESULTS_DIR}/logs/${experiment}_rank0.log"
    metrics="${RESULTS_DIR}/${WANDB_GROUP}/${experiment}/metrics.jsonl"
    if [[ ! -f "${log_file}" ]] || ! grep -q "SLIMADAM_BF16_500M_1XC_COMPLETE wd=${wd} iter=75457" "${log_file}"; then
        echo "Run ${experiment} is not confirmed complete" >&2
        exit 3
    fi
    if [[ ! -f "${metrics}" ]] || ! grep -q '"final-val/loss"' "${metrics}"; then
        echo "Final validation metric is missing for ${experiment}" >&2
        exit 4
    fi
done

shopt -s nullglob
offline_runs=("${RESULTS_DIR}"/wandb/offline-run-*)
if (( ${#offline_runs[@]} != 3 )); then
    echo "Expected exactly three offline W&B runs, found ${#offline_runs[@]}" >&2
    exit 5
fi

for run_dir in "${offline_runs[@]}"; do
    echo "SYNC_RUN=${run_dir}"
    wandb sync --entity andrey --project fp8-pretrain --mark-synced "${run_dir}"
done
echo "SLIMADAM_BF16_500M_WD_SWEEP_SYNC_COMPLETE"
