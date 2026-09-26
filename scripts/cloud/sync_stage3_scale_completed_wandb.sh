#!/usr/bin/env bash
set -euo pipefail

export WANDB_BASE_URL=https://wandb-radfan.ru
[[ -n "${WANDB_API_KEY:-}" ]] || { echo 'W&B credentials are missing' >&2; exit 2; }
command -v wandb >/dev/null

readonly root=/home/jovyan/dimativator/stage3_257m_scale_20260925
grep -q 'SCALE_4XC_COMPLETE lr=1e-3 iter=157000 mode=full' \
    "${root}/logs/257m_scale_bf16_4xC_cloud_4gpu_full_rank0.log"
grep -q 'FINAL_VAL_LOSS_EXACT iter=157000' \
    "${root}/logs/257m_scale_bf16_4xC_cloud_4gpu_full_rank0.log"
for scale in 1 2; do
    target=$((39250 * scale))
    grep -q "SCALE_DECAY_COMPLETE scale=${scale} lr=1e-3 iter=${target}" \
        "${root}/logs/257m_scale_bf16_${scale}xC_decay_cloud_4gpu_rank0.log"
    grep -q "FINAL_VAL_LOSS_EXACT iter=${target}" \
        "${root}/logs/257m_scale_bf16_${scale}xC_decay_cloud_4gpu_rank0.log"
done

sync_run() {
    local run_id=$1 run_dir
    local matches=("${root}/wandb_offline/wandb"/offline-run-*-"${run_id}")
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

sync_run stipjs1d
sync_run ud8u9lo7
sync_run 2pppfl82

python - <<'PY'
import wandb

api = wandb.Api()
for run_id in ("stipjs1d", "ud8u9lo7", "2pppfl82"):
    run = api.run(f"andrey/fp8-pretrain/{run_id}")
    assert run.id == run_id, run.id
    print(f"WANDB_VERIFIED id={run_id} url={run.url}", flush=True)
PY
echo 'SCALE_COMPLETED_WANDB_SYNC_COMPLETE count=3'
