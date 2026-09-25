#!/usr/bin/env bash
set -euo pipefail

: "${LR:?Set LR to the winning 1xC learning rate}"
: "${SCALE:?SCALE must be 1 or 2}"
case "${LR}" in 1e-4|5e-4|1e-3|2e-3) ;; *) exit 2 ;; esac
case "${SCALE}" in
    1) start=35325; end=39250 ;;
    2) start=70650; end=78500 ;;
    *) echo "SCALE must be 1 or 2" >&2; exit 2 ;;
esac

readonly packed_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly results_dir=/home/jovyan/dimativator/stage3_257m_scale_20260925
readonly eval_cache_dir=/home/jovyan/evals_cache
readonly checkpoint="${results_dir}/4xChinchilla_257M_bf16_scale/257m_scale_bf16_4xC_cloud_4gpu/ckpts/${start}"
readonly experiment="257m_scale_bf16_${SCALE}xC_decay_cloud_4gpu"
[[ -s "${checkpoint}/main.pt" ]] || { echo "Missing pre-decay checkpoint" >&2; exit 3; }
for source_rank in 0 1 2 3; do [[ -s "${checkpoint}/worker_${source_rank}.pt" ]] || exit 3; done

rank=${OMPI_COMM_WORLD_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
if (( world_size > 1 )); then
    [[ "${world_size}" == 4 ]] || exit 3
    export RANK=${RANK:-${rank}} WORLD_SIZE=${WORLD_SIZE:-${world_size}}
    export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    launcher=(python)
else
    launcher=(torchrun --standalone --nproc_per_node=4)
fi

mkdir -p "${results_dir}/logs" "${results_dir}/wandb_offline" "${eval_cache_dir}"
exec > >(tee -a "${results_dir}/logs/${experiment}_rank${rank}.log") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${results_dir}/wandb_offline"
echo "SCALE_DECAY_START scale=${SCALE} lr=${LR} start=${start} end=${end} rank=${rank}/${world_size}"
"${launcher[@]}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "${experiment}" \
    --seed 0 --data-seed 1337 \
    --dataset fineweb --datasets-dir "${packed_dir}" \
    --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
    --eval-cache-dir "${eval_cache_dir}" \
    --sequence-length 1024 --streaming --workers 8 \
    --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 \
    --dtype bfloat16 \
    --opt scale --lr "${LR}" --weight-decay 0.1 --momentum 0.9 \
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --resume-from "${checkpoint}" --decay-from-checkpoint \
    --scheduler wsd --wsd-final-lr-scale 0 --wsd-fract-decay 1.0 --decay-type cosine \
    --iterations "${end}" --warmup-steps 0 \
    --batch-size 8 --eval-batch-size 8 --acc-steps 16 \
    --eval-interval 500 --eval-batches 128 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --log-interval 50 --no-local-save \
    --results-base-folder "${results_dir}" \
    --wandb --wandb-project fp8-pretrain --wandb-group 4xChinchilla_257M_bf16_scale \
    --wandb-tags fineweb bf16 native_states 257M scale "decay_${SCALE}xc" "lr_${LR}" cloudru 4gpu local_offline

if (( rank == 0 )); then echo "SCALE_DECAY_COMPLETE scale=${SCALE} lr=${LR} iter=${end}"; fi
