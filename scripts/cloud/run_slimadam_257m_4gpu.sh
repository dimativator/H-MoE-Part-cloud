#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
if [[ "${NPROC_PER_NODE}" != "4" ]]; then
    echo "This launcher requires exactly four GPUs" >&2
    exit 2
fi

DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LOG_DIR=${LOG_DIR:-/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud}
WANDB_DIR=${WANDB_DIR:-${RESULTS_DIR}/wandb_offline}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-257m_slim_adam_fp8_states_8xC_cloud_4gpu}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-5000}
HF_RELAY_REPO_ID=${HF_RELAY_REPO_ID:-}
HF_RELAY_PREFIX=${HF_RELAY_PREFIX:-slimadam_257m_fp8_states_8xC}
WANDB_GROUP=8xChinchilla_257M_fp8_states

mkdir -p "${RESULTS_DIR}" "${EVAL_CACHE_DIR}" "${LOG_DIR}" "${WANDB_DIR}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FINEWEB_LOG_DATA_HASHES=1
export WANDB_MODE=offline
export WANDB_DIR

rank=${OMPI_COMM_WORLD_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
if (( world_size > 1 )); then
    if [[ "${world_size}" != "${NPROC_PER_NODE}" ]]; then
        echo "mlsub MPI world size must equal ${NPROC_PER_NODE}" >&2
        exit 3
    fi
    export RANK=${RANK:-${rank}}
    export WORLD_SIZE=${WORLD_SIZE:-${world_size}}
    export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("$(command -v python)")
else
    TRAIN_LAUNCHER=("$(command -v torchrun)" --standalone --nproc_per_node=4)
fi

if (( rank == 0 )); then
    echo "GPU_STATUS_BEFORE_LAUNCH"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
fi

if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Missing exact packed FineWeb metadata: ${DATASETS_DIR}/packed_metadata.json" >&2
    exit 4
fi

common_args=(
    --distributed-backend nccl
    --seed 0
    --dataset fineweb
    --datasets-dir "${DATASETS_DIR}"
    --fineweb-replay-world-size 1
    --fineweb-replay-layout concat
    --eval-cache-dir "${EVAL_CACHE_DIR}"
    --sequence-length 1024
    --data-seed 1337
    --streaming
    --workers 8
    --model llama
    --n-layer 12
    --n-embd 1024
    --n-head 8
    --multiple-of 256
    --dtype bfloat16
    --opt slim_adam
    --lr 5e-4
    --weight-decay 1e-1
    --beta1 0.9
    --beta2 0.99
    --grad-clip 1.0
    --scheduler wsd
    --wsd-final-lr-scale 0
    --decay-type cosine
    --batch-size 8
    --eval-batch-size 8
    --acc-steps 16
    --fp8-optim
    --fp8-qgroup-size 128
    --fp8-first-order-bit E4M3
    --fp8-second-order-bit E4M3
    --fp8-expansion expand
    --results-base-folder "${RESULTS_DIR}"
)

if [[ "${MODE}" == "smoke" ]]; then
    experiment_name="${EXPERIMENT_NAME}_smoke"
    run_args=(
        --iterations 3
        --warmup-steps 1
        --wsd-fract-decay 0.1
        --eval-interval 3
        --eval-batches 1
        --log-interval 1
        --no-local-save
    )
elif [[ "${MODE}" == "full" ]]; then
    experiment_name="${EXPERIMENT_NAME}"
    run_args=(
        --iterations 314000
        --warmup-steps 2000
        --wsd-fract-decay 0.1
        --eval-interval 500
        --eval-batches 128
        --downstream-eval-enabled
        --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled
        --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
        --log-interval 50
        --inter-ckpts 35325 70650 141300
        --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
        --wandb
        --wandb-project fp8-pretrain
        --wandb-group "${WANDB_GROUP}"
        --wandb-tags optimizer_fp8 bf16_model 257M slim_adam full_8xc cloudru 4gpu local_offline h200_data_parity
    )
else
    echo "MODE must be smoke or full" >&2
    exit 2
fi

if [[ "${MODE}" == "full" && -n "${HF_RELAY_REPO_ID}" ]]; then
    run_args+=(
        --upload-inter-ckpts-to huggingface
        --hf-inter-ckpt-repo-id "${HF_RELAY_REPO_ID}"
        --delete-local-inter-ckpts-after-upload
    )
fi

experiment_dir="${RESULTS_DIR}/${WANDB_GROUP}/${experiment_name}"
stop_file="${experiment_dir}/.relay_stop"
relay_pid=""
if [[ "${MODE}" == "full" && "${rank}" == "0" && -n "${HF_RELAY_REPO_ID}" ]]; then
    rm -f "${stop_file}"
    python scripts/cloud/relay_latest_checkpoint_to_hf.py \
        --checkpoint-dir "${experiment_dir}/ckpts/latest" \
        --staging-dir "${experiment_dir}/ckpts/.relay" \
        --repo-id "${HF_RELAY_REPO_ID}" \
        --remote-prefix "${HF_RELAY_PREFIX}" \
        --world-size 4 \
        --stop-file "${stop_file}" &
    relay_pid=$!
fi

echo "MODE=${MODE} EXPERIMENT=${experiment_name} rank=${rank}/${world_size}"
echo "PARITY_CONFIG physical_microbatch=8 physical_acc_steps=4 global_batch=128"
set +e
"${TRAIN_LAUNCHER[@]}" src/main.py \
    --experiment-name "${experiment_name}" \
    "${common_args[@]}" \
    "${run_args[@]}"
train_status=$?
set -e

if [[ -n "${relay_pid}" ]]; then
    touch "${stop_file}"
    wait "${relay_pid}"
fi
echo "TRAIN_EXIT=${train_status}"
echo "LOG=${LOG_FILE}"
exit "${train_status}"
