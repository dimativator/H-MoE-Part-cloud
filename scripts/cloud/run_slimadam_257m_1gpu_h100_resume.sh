#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-remote}
FINEWEB_MANIFEST=${FINEWEB_MANIFEST:-/home/jovyan/fineweb_h200_manifest.json}
RESUME_FROM=${RESUME_FROM:-/workspace-SR006.nfs3/dimativator/checkpoints/slimadam_257m_fp8_states_8xC/160000}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LOG_DIR=${LOG_DIR:-/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud}
WANDB_DIR=${WANDB_DIR:-${RESULTS_DIR}/wandb_offline}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-5000}
HF_RELAY_REPO_ID=${HF_RELAY_REPO_ID:-}
HF_RELAY_PREFIX=${HF_RELAY_PREFIX:-slimadam_257m_fp8_states_cloud_1gpu}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-}
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

echo "GPU_STATUS_BEFORE_LAUNCH"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

for path in "${FINEWEB_MANIFEST}" "${RESUME_FROM}/main.pt" "${RESUME_FROM}/worker_0.pt"; do
    if [[ ! -s "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 4
    fi
done

common_args=(
    --distributed-backend nccl
    --seed 0
    --dataset fineweb
    --datasets-dir "${DATASETS_DIR}"
    --fineweb-manifest "${FINEWEB_MANIFEST}"
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
    --iterations 314000
    --warmup-steps 2000
    --wsd-final-lr-scale 0
    --wsd-fract-decay 0.1
    --decay-type cosine
    --batch-size 32
    --eval-batch-size 32
    --acc-steps 4
    --fp8-optim
    --fp8-qgroup-size 128
    --fp8-first-order-bit E4M3
    --fp8-second-order-bit E4M3
    --fp8-expansion expand
    --resume-from "${RESUME_FROM}"
    --results-base-folder "${RESULTS_DIR}"
)

if [[ "${MODE}" == "smoke" ]]; then
    experiment_name="${EXPERIMENT_NAME}_smoke"
    run_args=(
        --early-stop-iteration 160003
        --eval-interval 500
        --eval-batches 32
        --log-interval 1
        --no-local-save
    )
elif [[ "${MODE}" == "full" ]]; then
    experiment_name="${EXPERIMENT_NAME}"
    run_args=(
        --eval-interval 500
        --eval-batches 32
        --downstream-eval-enabled
        --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled
        --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
        --log-interval 50
        --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
        --wandb
        --wandb-project fp8-pretrain
        --wandb-group "${WANDB_GROUP}"
        --wandb-tags optimizer_fp8 bf16_model 257M slim_adam full_8xc cloudru 1gpu h100 local_offline h200_data_cursor_resume
    )
else
    echo "MODE must be smoke or full" >&2
    exit 2
fi

experiment_dir="${RESULTS_DIR}/${WANDB_GROUP}/${experiment_name}"
stop_file="${experiment_dir}/.relay_stop"
relay_pid=""
if [[ "${MODE}" == "full" && -n "${HF_RELAY_REPO_ID}" ]]; then
    if [[ ! -s "${HF_TOKEN_FILE}" ]]; then
        echo "HF_TOKEN_FILE is required when HF_RELAY_REPO_ID is set" >&2
        exit 5
    fi
    rm -f "${stop_file}"
    python scripts/cloud/relay_latest_checkpoint_to_hf.py \
        --checkpoint-dir "${experiment_dir}/ckpts/latest" \
        --staging-dir "${experiment_dir}/ckpts/.relay" \
        --repo-id "${HF_RELAY_REPO_ID}" \
        --remote-prefix "${HF_RELAY_PREFIX}" \
        --world-size 1 \
        --stop-file "${stop_file}" \
        --token-file "${HF_TOKEN_FILE}" &
    relay_pid=$!
fi

echo "MODE=${MODE} EXPERIMENT=${experiment_name}"
echo "PARITY_CONFIG microbatch=32 acc_steps=4 effective_batch=128 resume=160000 scheduler_target=314000"
set +e
torchrun --standalone --nproc_per_node=1 src/main.py \
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
