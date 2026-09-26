#!/usr/bin/env bash
set -euo pipefail

export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey
[[ -n "${WANDB_API_KEY:-}" || -r "${HOME}/.netrc" ]] || {
    echo 'W&B credentials are missing' >&2
    exit 2
}
command -v wandb >/dev/null

readonly adamw_root=/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/adamw
readonly slim_root=/home/jovyan/dimativator/500m-time-matched-20260925/slim_adam
readonly frugal_root=/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/frugal
readonly scale_root=/home/jovyan/dimativator/stage3_257m_scale_20260925

grep -q 'TIME_MATCHED_ADAMW_FULL_COMPLETE iter=92807' "${adamw_root}/logs/llama500M_adamw_fp8_time_matched_1gpu_full.log"
grep -q 'TIME_MATCHED_COMPLETE optimizer=slim_adam iter=90333' "${slim_root}/logs/llama500M_slim_adam_bf16_time_matched_4gpu_rank0.log"
grep -q 'TIME_MATCHED_COMPLETE optimizer=frugal iter=91258' "${frugal_root}/logs/llama500M_frugal_bf16_time_matched_4gpu_rank0.log"
for lr in 1e-4 5e-4 1e-3 2e-3; do
    grep -q "SCALE_SWEEP_COMPLETE lr=${lr} iter=39250 early_stop=none" \
        "${scale_root}/logs/257m_scale_bf16_1xC_lr${lr}_cloud_2gpu_rank0.log"
done

sync_run() {
    local root=$1 run_id=$2 run_dir
    local matches=("${root}"/wandb/offline-run-*-"${run_id}")
    (( ${#matches[@]} == 1 )) && [[ -d "${matches[0]}" ]] || {
        echo "Expected exactly one offline run for ${run_id}" >&2
        exit 3
    }
    run_dir=${matches[0]}
    [[ -s "${run_dir}/run-${run_id}.wandb" ]] || {
        echo "Missing W&B data for ${run_id}" >&2
        exit 4
    }
    echo "SYNC_START id=${run_id}"
    wandb sync --entity andrey --project fp8-pretrain --mark-synced "${run_dir}"
    echo "SYNC_LINK=https://wandb-radfan.ru/andrey/fp8-pretrain/runs/${run_id}"
}

sync_run "${adamw_root}/wandb" u91eypyp
sync_run "${slim_root}/wandb" t4yfqnan
sync_run "${frugal_root}/wandb" j1l970l8
for run_id in 2mzhzp6d c0ad718l moxb61fy yxgwpvkr; do
    sync_run "${scale_root}/wandb_offline" "${run_id}"
done
echo 'STAGE3_COMPLETED_WANDB_SYNC_COMPLETE count=7'
