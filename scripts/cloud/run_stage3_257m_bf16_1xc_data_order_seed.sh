#!/usr/bin/env bash
set -euo pipefail

: "${OPTIMIZER:?OPTIMIZER must be adamw or muon}"
: "${SEED:?SEED must be 1, 2, or 3}"
[[ "${SEED}" == 1 || "${SEED}" == 2 || "${SEED}" == 3 ]] || {
    echo "SEED must be 1, 2, or 3" >&2
    exit 2
}
case "${OPTIMIZER}" in
    adamw|muon) ;;
    *) echo "OPTIMIZER must be adamw or muon" >&2; exit 2 ;;
esac

readonly data_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly results_dir=/home/jovyan/dimativator/stage3_257m_bf16_1xc_data_order_20260924
readonly eval_cache_dir=/home/jovyan/evals_cache
readonly group=1xChinchilla_257M_bf16_native_states_data_order_seed_sweep
readonly experiment="257m_${OPTIMIZER}_bf16_1xC_data_order_seed${SEED}_cloud_1gpu"
readonly log_file="${results_dir}/logs/${experiment}.log"

[[ -s "${data_dir}/packed_metadata.json" ]] || {
    echo "Missing packed FineWeb metadata" >&2
    exit 3
}
mkdir -p "${results_dir}/logs" "${results_dir}/wandb_offline" "${eval_cache_dir}"
exec > >(tee -a "${log_file}") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${results_dir}/wandb_offline"

echo "DATA_ORDER_SEED_RUN_START optimizer=${OPTIMIZER} model_seed=${SEED} data_seed=${SEED}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
df -h /home/jovyan /workspace-SR006.nfs3

torchrun --standalone --nproc_per_node=1 src/main.py \
    --distributed-backend nccl \
    --experiment-name "${experiment}" \
    --seed "${SEED}" --data-seed "${SEED}" \
    --dataset fineweb --datasets-dir "${data_dir}" \
    --fineweb-replay-world-size 2 --fineweb-replay-layout concat \
    --fineweb-packed-shuffle-steps \
    --eval-cache-dir "${eval_cache_dir}" \
    --sequence-length 1024 --streaming --workers 8 \
    --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 \
    --dtype bfloat16 \
    --opt "${OPTIMIZER}" --lr 1e-3 --weight-decay 1e-1 \
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --scheduler wsd --wsd-final-lr-scale 0 --wsd-fract-decay 0.1 --decay-type cosine \
    --iterations 39250 --warmup-steps 2000 \
    --batch-size 32 --eval-batch-size 32 --acc-steps 4 \
    --eval-interval 500 --eval-batches 32 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --log-interval 50 --no-local-save \
    --results-base-folder "${results_dir}" \
    --wandb --wandb-project fp8-pretrain --wandb-group "${group}" \
    --wandb-tags bf16_model native_optimizer_states no_fp8_optim 257M \
        "${OPTIMIZER}" full_1xc "model_seed_${SEED}" "data_order_seed_${SEED}" \
        cloudru 1gpu local_offline

echo "DATA_ORDER_SEED_RUN_COMPLETE optimizer=${OPTIMIZER} seed=${SEED} iter=39250"
