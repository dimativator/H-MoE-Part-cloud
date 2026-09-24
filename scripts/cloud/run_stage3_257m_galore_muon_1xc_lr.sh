#!/usr/bin/env bash
set -euo pipefail

: "${LR:?LR must be one of 1e-4, 5e-4, 1e-2, 2e-3}"
case "${LR}" in
    1e-4|5e-4|1e-2|2e-3) ;;
    *) echo "Unsupported LR: ${LR}" >&2; exit 2 ;;
esac
case "${SMOKE_TEST:-0}" in
    0|1) ;;
    *) echo "SMOKE_TEST must be 0 or 1" >&2; exit 2 ;;
esac

readonly data_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly eval_cache_dir=/home/jovyan/evals_cache
readonly group=1xChinchilla_257M_bf16_galore_muon_lr_sweep
readonly experiment="257m_galore_muon_bf16_1xC_lr${LR}_cloud_1gpu"
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    readonly results_dir=/home/jovyan/dimativator/stage3_257m_galore_muon_smoke_20260924
    readonly iterations=2
    readonly warmup_steps=1
else
    readonly results_dir=/home/jovyan/dimativator/stage3_257m_galore_muon_1xc_20260924
    readonly iterations=39250
    readonly warmup_steps=2000
fi
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

echo "GALORE_MUON_START lr=${LR} iterations=${iterations} model_seed=0 data_seed=1337"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
df -h /home/jovyan /workspace-SR006.nfs3

torchrun --standalone --nproc_per_node=1 src/main.py \
    --distributed-backend nccl \
    --experiment-name "${experiment}" \
    --seed 0 --data-seed 1337 \
    --dataset fineweb --datasets-dir "${data_dir}" \
    --fineweb-replay-world-size 2 --fineweb-replay-layout concat \
    --eval-cache-dir "${eval_cache_dir}" \
    --sequence-length 1024 --streaming --workers 8 \
    --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 \
    --dtype bfloat16 \
    --opt galore_muon --lr "${LR}" --weight-decay 1e-1 \
    --non_proj_opt adamw --inactive_lr_scale 0 \
    --density 0.25 --update_gap 50 --proj_side std --proj_type svd \
    --momentum 0.95 --muon_ns_steps 5 \
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --scheduler wsd --wsd-final-lr-scale 0 --wsd-fract-decay 0.1 --decay-type cosine \
    --iterations "${iterations}" --warmup-steps "${warmup_steps}" \
    --batch-size 32 --eval-batch-size 32 --acc-steps 4 \
    --eval-interval 500 --eval-batches 32 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --log-interval 50 --no-local-save \
    --results-base-folder "${results_dir}" \
    --wandb --wandb-project fp8-pretrain --wandb-group "${group}" \
    --wandb-tags bf16_model galore_muon pure_projection native_optimizer_states \
        no_fp8_optim 257M full_1xc "lr_${LR}" cloudru 1gpu local_offline

echo "GALORE_MUON_COMPLETE lr=${LR} iter=${iterations}"
