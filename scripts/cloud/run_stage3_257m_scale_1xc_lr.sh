#!/usr/bin/env bash
set -euo pipefail

: "${LR:?LR must be one of 1e-4, 5e-4, 1e-3, 2e-3}"
case "${LR}" in
    1e-4|5e-4|1e-3|2e-3) ;;
    *) echo "Unsupported LR: ${LR}" >&2; exit 2 ;;
esac
case "${SMOKE_TEST:-0}" in
    0|1) ;;
    *) echo "SMOKE_TEST must be 0 or 1" >&2; exit 2 ;;
esac

readonly packed_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly eval_cache_dir=/home/jovyan/evals_cache
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    readonly results_dir=/home/jovyan/dimativator/stage3_257m_scale_smoke_20260925
    readonly experiment_suffix=_smoke
else
    readonly results_dir=/home/jovyan/dimativator/stage3_257m_scale_20260925
    readonly experiment_suffix=
fi
readonly group=1xChinchilla_257M_bf16_scale_lr_sweep
readonly experiment="257m_scale_bf16_1xC_lr${LR}_cloud_2gpu${experiment_suffix}"
readonly iterations=39250
readonly warmup_steps=2000
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    readonly early_stop=2
else
    readonly early_stop=${EARLY_STOP_ITERATION:-}
fi
[[ -s "${packed_dir}/packed_metadata.json" ]] || { echo "Packed FineWeb missing" >&2; exit 3; }

rank=${OMPI_COMM_WORLD_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
if (( world_size > 1 )); then
    [[ "${world_size}" == 2 ]] || { echo "Exactly two MPI ranks required" >&2; exit 3; }
    export RANK=${RANK:-${rank}} WORLD_SIZE=${WORLD_SIZE:-${world_size}}
    export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    launcher=(python)
else
    launcher=(torchrun --standalone --nproc_per_node=2)
fi

mkdir -p "${results_dir}/logs" "${results_dir}/wandb_offline" "${eval_cache_dir}"
exec > >(tee -a "${results_dir}/logs/${experiment}_rank${rank}.log") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${results_dir}/wandb_offline"
echo "SCALE_SWEEP_START lr=${LR} rank=${rank}/${world_size} iterations=${iterations}"
df -h /home/jovyan /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

extra_args=()
if [[ -n "${early_stop}" ]]; then extra_args=(--early-stop-iteration "${early_stop}"); fi
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
    --scheduler wsd --wsd-final-lr-scale 0 --wsd-fract-decay 0.1 --decay-type cosine \
    --iterations "${iterations}" --warmup-steps "${warmup_steps}" \
    --batch-size 16 --eval-batch-size 16 --acc-steps 8 \
    --eval-interval 500 --eval-batches 64 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --log-interval 50 --no-local-save \
    --results-base-folder "${results_dir}" \
    --wandb --wandb-project fp8-pretrain --wandb-group "${group}" \
    --wandb-tags fineweb bf16 native_states 257M scale full_1xc "lr_${LR}" cloudru 2gpu local_offline \
    "${extra_args[@]}"

if (( rank == 0 )); then echo "SCALE_SWEEP_COMPLETE lr=${LR} iter=${iterations} early_stop=${early_stop:-none}"; fi
