#!/usr/bin/env bash
set -euo pipefail

: "${LR:?Set LR to the winning 1xC learning rate}"
case "${LR}" in
    1e-4|5e-4|1e-3|2e-3) ;;
    *) echo "Unsupported LR: ${LR}" >&2; exit 2 ;;
esac
MODE=${MODE:-full}
case "${MODE}" in
    full|resume|smoke) ;;
    *) echo "MODE must be full, resume, or smoke" >&2; exit 2 ;;
esac

readonly packed_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly eval_cache_dir=/home/jovyan/evals_cache
readonly results_dir=/home/jovyan/dimativator/stage3_257m_scale_20260925
readonly group=4xChinchilla_257M_bf16_scale
readonly experiment=257m_scale_bf16_4xC_cloud_4gpu
readonly iterations=157000
readonly resume_from=${RESUME_FROM:-}
[[ -s "${packed_dir}/packed_metadata.json" ]] || { echo "Packed FineWeb missing" >&2; exit 3; }
if [[ "${MODE}" == resume ]]; then
    [[ -s "${resume_from}/main.pt" ]] || { echo "Verified RESUME_FROM is required" >&2; exit 4; }
    for source_rank in 0 1 2 3; do
        [[ -s "${resume_from}/worker_${source_rank}.pt" ]] || exit 4
    done
fi

rank=${OMPI_COMM_WORLD_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
if (( world_size > 1 )); then
    [[ "${world_size}" == 4 ]] || { echo "Exactly four MPI ranks required" >&2; exit 3; }
    export RANK=${RANK:-${rank}} WORLD_SIZE=${WORLD_SIZE:-${world_size}}
    export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    launcher=(python)
else
    launcher=(torchrun --standalone --nproc_per_node=4)
fi

mkdir -p "${results_dir}/logs" "${results_dir}/wandb_offline" "${eval_cache_dir}"
available_bytes=$(df -B1 --output=avail "${results_dir}" | tail -n 1 | tr -d ' ')
if [[ "${MODE}" != smoke ]] && (( available_bytes < 8000000000 )); then
    echo "Insufficient checkpoint headroom: ${available_bytes} bytes" >&2
    exit 5
fi
exec > >(tee -a "${results_dir}/logs/${experiment}_${MODE}_rank${rank}.log") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${results_dir}/wandb_offline"
echo "SCALE_4XC_START mode=${MODE} lr=${LR} rank=${rank}/${world_size}"
df -h /home/jovyan /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

extra_args=(--latest-ckpt-interval 5000 --inter-ckpts 35325 70650 141300)
if [[ "${MODE}" == resume ]]; then extra_args+=(--resume-from "${resume_from}"); fi
if [[ "${MODE}" == smoke ]]; then
    extra_args=(--early-stop-iteration 2 --no-local-save)
    experiment_name="${experiment}_smoke"
else
    experiment_name="${experiment}"
fi
"${launcher[@]}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "${experiment_name}" \
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
    --iterations "${iterations}" --warmup-steps 2000 \
    --batch-size 8 --eval-batch-size 8 --acc-steps 16 \
    --eval-interval 500 --eval-batches 128 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --log-interval 50 \
    --results-base-folder "${results_dir}" \
    --wandb --wandb-project fp8-pretrain --wandb-group "${group}" \
    --wandb-tags fineweb bf16 native_states 257M scale full_4xc "lr_${LR}" cloudru 4gpu local_offline \
    "${extra_args[@]}"

if (( rank == 0 )); then echo "SCALE_4XC_COMPLETE lr=${LR} iter=${iterations} mode=${MODE}"; fi
