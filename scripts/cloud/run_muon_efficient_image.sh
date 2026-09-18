#!/usr/bin/env bash
set -uo pipefail

# Run an FP8 optimizer-state experiment directly in the Cloud.ru
# `efficient` image.  Unlike run_optimizer_fp8_cloud.sh, this entrypoint does
# not install packages or switch to the persistent Torch 2.5.1 environment.

MODE=${MODE:-smoke}
OPTIMIZER=${OPTIMIZER:-muon}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
GRAD_CLIP=${GRAD_CLIP:-1.0}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-milestones}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-10000}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
case "${NPROC_PER_NODE}" in
    1|2|4) ;;
    *)
        echo "H200 parity supports NPROC_PER_NODE=1, 2, or 4" >&2
        exit 2
        ;;
esac
BATCH_SIZE=${BATCH_SIZE:-$((32 / NPROC_PER_NODE))}
if (( NPROC_PER_NODE == 1 )); then
    FINEWEB_REPLAY_WORLD_SIZE=${FINEWEB_REPLAY_WORLD_SIZE:-2}
else
    FINEWEB_REPLAY_WORLD_SIZE=${FINEWEB_REPLAY_WORLD_SIZE:-1}
fi
FINEWEB_REPLAY_LAYOUT=${FINEWEB_REPLAY_LAYOUT:-concat}
RUN_SEED=${SEED:-0}
FULL_WARMUP_STEPS=${WARMUP_STEPS:-2000}
FULL_ACC_STEPS=${ACC_STEPS:-}
FULL_ITERATIONS=${ITERATIONS:-75457}
HORIZON_TAG=${HORIZON_TAG:-1xChinchilla}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/exps}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LOG_DIR=${LOG_DIR:-/workspace-SR006.nfs3/dimativator/logs/optimizer_fp8_cloud}
WANDB_PROJECT=${WANDB_PROJECT:-fp8-pretrain}
WANDB_ENTITY=${WANDB_ENTITY:-andrey}
WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
WANDB_GROUP=${WANDB_GROUP:-1xChinchilla_optimizer_fp8_cloud}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-500m_${OPTIMIZER}_optimizer_fp8_1xC_cloud_${NPROC_PER_NODE}gpu_torch291_h200_data_parity_v1}
SMOKE_MARKER=${SMOKE_MARKER:-${LOG_DIR}/.${EXPERIMENT_NAME}_smoke_ok}
TEST_ITERATIONS=${TEST_ITERATIONS:-10}
TEST_SCHEDULER_ITERATIONS=${TEST_SCHEDULER_ITERATIONS:-75457}
TEST_EVAL_BATCHES=${TEST_EVAL_BATCHES:-32}

if [[ "${MODE}" == "inspect_home_inodes" ]]; then
    echo "MODE=${MODE}"
    df -h /home/jovyan 2>&1 || true
    df -i /home/jovyan 2>&1 || true
    echo "HOME_INODE_USAGE"
    du --inodes -x -d 2 /home/jovyan 2>/dev/null | sort -n | tail -n 80
    exit 0
fi

if [[ "${MODE}" == "cleanup_triton_cache" ]]; then
    TRITON_CACHE_DIR=/home/jovyan/.cache/triton
    echo "MODE=${MODE}"
    echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    df -i /home/jovyan 2>&1 || true
    if [[ ! -d "${TRITON_CACHE_DIR}" ]]; then
        echo "Triton cache does not exist; nothing to clean"
        exit 0
    fi
    cache_inodes=$(du --inodes -s "${TRITON_CACHE_DIR}" 2>/dev/null | awk '{print $1}')
    echo "TRITON_CACHE_INODES_BEFORE=${cache_inodes:-unknown}"
    find "${TRITON_CACHE_DIR}" -depth -mindepth 1 -delete
    cache_inodes=$(du --inodes -s "${TRITON_CACHE_DIR}" 2>/dev/null | awk '{print $1}')
    echo "TRITON_CACHE_INODES_AFTER=${cache_inodes:-unknown}"
    df -i /home/jovyan 2>&1 || true
    exit 0
fi

if [[ "${MODE}" == "cleanup_mlspace_log_links" ]]; then
    MLS_LOG_LINK_DIR=/home/jovyan/mlspace-logs
    echo "MODE=${MODE}"
    echo "MLS_LOG_LINK_DIR=${MLS_LOG_LINK_DIR}"
    df -h /home/jovyan 2>&1 || true
    df -i /home/jovyan 2>&1 || true
    if [[ ! -d "${MLS_LOG_LINK_DIR}" ]]; then
        echo "Cloud log-link directory does not exist; nothing to clean"
        exit 0
    fi
    link_count=$(find "${MLS_LOG_LINK_DIR}" -mindepth 1 -maxdepth 1 -type l -print 2>/dev/null | wc -l)
    regular_count=$(find "${MLS_LOG_LINK_DIR}" -mindepth 1 -maxdepth 1 ! -type l -print 2>/dev/null | wc -l)
    echo "MLS_LOG_SYMLINKS_BEFORE=${link_count}"
    echo "MLS_LOG_NON_SYMLINKS_PRESERVED=${regular_count}"
    find "${MLS_LOG_LINK_DIR}" -mindepth 1 -maxdepth 1 -type l -delete
    remaining_links=$(find "${MLS_LOG_LINK_DIR}" -mindepth 1 -maxdepth 1 -type l -print 2>/dev/null | wc -l)
    echo "MLS_LOG_SYMLINKS_AFTER=${remaining_links}"
    df -h /home/jovyan 2>&1 || true
    df -i /home/jovyan 2>&1 || true
    exit 0
fi

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${EVAL_CACHE_DIR}"
LOG_FILE="${LOG_DIR}/${OPTIMIZER}_efficient_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
    status=$?
    echo "EXIT=${status}"
    echo "LOG=${LOG_FILE}"
    tail -n 100 "${LOG_FILE}" || true
    trap - EXIT
    exit "${status}"
}
trap finish EXIT

echo "MODE=${MODE} OPTIMIZER=${OPTIMIZER} EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "SEED=${RUN_SEED}"
echo "FINEWEB_REPLAY_WORLD_SIZE=${FINEWEB_REPLAY_WORLD_SIZE}"
echo "FINEWEB_REPLAY_LAYOUT=${FINEWEB_REPLAY_LAYOUT}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /tmp 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true

if [[ "${MODE}" == "inspect" ]]; then
    EXPERIMENT_DIR="${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}"
    CHECKPOINT_DIR="${EXPERIMENT_DIR}/ckpts"
    LATEST_DIR="${CHECKPOINT_DIR}/latest"
    echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
    echo "CHECKPOINT_FILES"
    find "${CHECKPOINT_DIR}" -maxdepth 2 -type f \
        -printf '%T@ %s %p\n' 2>/dev/null | sort -n || true
    echo "CHECKPOINT_DIRS"
    find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d \
        -printf '%f\n' 2>/dev/null | sort || true

    CHECKPOINT_PATH="${LATEST_DIR}/main.pt" \
    WORKER_PATH="${LATEST_DIR}/worker_0.pt" \
        "$(command -v python)" - <<'PY'
import os
from pathlib import Path

import torch

main_path = Path(os.environ["CHECKPOINT_PATH"])
worker_path = Path(os.environ["WORKER_PATH"])
if not main_path.is_file() or not worker_path.is_file():
    raise FileNotFoundError(
        f"incomplete latest checkpoint: main={main_path.is_file()} "
        f"worker={worker_path.is_file()}"
    )
checkpoint = torch.load(
    main_path, map_location="cpu", mmap=True, weights_only=False
)
torch.load(worker_path, map_location="cpu", weights_only=False)
iteration = int(checkpoint["itr"])
if iteration <= 0:
    raise ValueError(f"invalid checkpoint iteration: {iteration}")
print(f"LATEST_CHECKPOINT={main_path}")
print(f"LATEST_CHECKPOINT_ITER={iteration}")
print(f"LATEST_CHECKPOINT_BYTES={main_path.stat().st_size}")
print(f"LATEST_WORKER={worker_path}")
print(f"LATEST_WORKER_BYTES={worker_path.stat().st_size}")
PY
    exit 0
fi

PYTHON_BIN=$(command -v python)
TORCHRUN_BIN=$(command -v torchrun)
TRAIN_LAUNCHER=("${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}")
if (( ${OMPI_COMM_WORLD_SIZE:-1} > 1 )); then
    if [[ "${OMPI_COMM_WORLD_SIZE}" != "${NPROC_PER_NODE}" ]]; then
        echo "mlsub MPI world size must match NPROC_PER_NODE" >&2
        exit 8
    fi
    export RANK=${RANK:-"${OMPI_COMM_WORLD_RANK}"}
    export WORLD_SIZE=${WORLD_SIZE:-"${OMPI_COMM_WORLD_SIZE}"}
    export LOCAL_RANK=${LOCAL_RANK:-"${OMPI_COMM_WORLD_LOCAL_RANK:-0}"}
    export MASTER_ADDR=${MASTER_ADDR:-"$(hostname -f)"}
    export MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("${PYTHON_BIN}")
    echo "MLSUB_DDP rank=${RANK}/${WORLD_SIZE} local_rank=${LOCAL_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
fi
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

import torch
import torchao
import triton

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
assert torch.cuda.is_available()
assert torchao.__version__.startswith("0.15.0"), torchao.__version__
assert triton.__version__ == "3.5.1", triton.__version__
print("python_environment", "torch", torch.__version__, "cuda", torch.version.cuda)
print("python_environment", "torchao", torchao.__version__, "triton", triton.__version__)
print("python_environment", "wandb", metadata.version("wandb"))
print("gpu", torch.cuda.get_device_name(0))
PY

if [[ -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "FINEWEB_PACKED_METADATA=${DATASETS_DIR}/packed_metadata.json"
else
    parquet_files=("${DATASETS_DIR}"/*.parquet)
    if [[ ! -f "${parquet_files[0]:-}" ]]; then
        echo "FineWeb dataset is missing from ${DATASETS_DIR}" >&2
        exit 5
    fi
    echo "FINEWEB_PARQUET_ROOT=${DATASETS_DIR} files=${#parquet_files[@]}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT WANDB_ENTITY WANDB_BASE_URL

# Cloud.ru's persistent credentials live under /home/jovyan, while the newer
# base image sets HOME=/home/user.  Point standard netrc discovery at the
# persistent home without copying or printing the API key.
if [[ -z "${WANDB_API_KEY:-}" && -r /home/jovyan/.netrc ]]; then
    export HOME=/home/jovyan
    echo "WANDB_AUTH_SOURCE=/home/jovyan/.netrc"
fi

if [[ "${MODE}" == "smoke" ]]; then
    rm -f -- "${SMOKE_MARKER}"
    ITERATIONS=3
    WARMUP_STEPS=1
    ACC_STEPS=$((4 * NPROC_PER_NODE))
    EVAL_INTERVAL=3
    EVAL_BATCHES=1
    RUN_LOG_INTERVAL=1
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
    EXTRA_ARGS=(--no-local-save)
elif [[ "${MODE}" == "test" ]]; then
    if (( NPROC_PER_NODE * BATCH_SIZE != 32 )); then
        echo "H200 parity requires NPROC_PER_NODE * BATCH_SIZE = 32" >&2
        exit 7
    fi
    if (( TEST_ITERATIONS <= 0 || TEST_EVAL_BATCHES <= 0 || TEST_SCHEDULER_ITERATIONS <= TEST_ITERATIONS )); then
        echo "Test lengths are invalid" >&2
        exit 2
    fi
    ITERATIONS=${TEST_SCHEDULER_ITERATIONS}
    WARMUP_STEPS=2000
    ACC_STEPS=$((4 * NPROC_PER_NODE))
    EVAL_INTERVAL=${TEST_ITERATIONS}
    EVAL_BATCHES=${TEST_EVAL_BATCHES}
    RUN_LOG_INTERVAL=1
    EXTRA_ARGS=(
        --early-stop-iteration "${TEST_ITERATIONS}"
        --no-local-save
        --wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-tags fineweb optimizer_fp8 bf16_model 1xChinchilla 0.5B "${NPROC_PER_NODE}gpu" cloudru h100 "${OPTIMIZER}" torch291 efficient-image h200-data-parity test
    )
elif [[ "${MODE}" == "full" ]]; then
    if [[ "${REQUIRE_SMOKE_MARKER:-0}" == "1" && ! -f "${SMOKE_MARKER}" ]]; then
        echo "Required smoke marker is missing: ${SMOKE_MARKER}" >&2
        exit 6
    fi
    "${PYTHON_BIN}" - <<'PY'
import os

import wandb

assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is required for a full run"
viewer = wandb.Api(timeout=30).viewer
assert viewer, "W&B authentication returned an empty viewer"
print("WANDB_AUTH=ok")
PY
    ITERATIONS=${FULL_ITERATIONS}
    WARMUP_STEPS=${FULL_WARMUP_STEPS}
    if [[ -n "${FULL_ACC_STEPS}" ]]; then
        ACC_STEPS=${FULL_ACC_STEPS}
    else
        ACC_STEPS=$((4 * NPROC_PER_NODE))
    fi
    EVAL_INTERVAL=500
    EVAL_BATCHES=32
    RUN_LOG_INTERVAL=50
    if [[ "${CHECKPOINT_MODE}" == "milestones" ]]; then
        CHECKPOINT_ARGS=(
            --inter-ckpts 10000 20000 30000 40000 50000 60000 67911 70000
            --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
            --upload-inter-ckpts-to wandb
            --delete-local-inter-ckpts-after-upload
        )
    elif [[ "${CHECKPOINT_MODE}" == "latest" ]]; then
        if (( LATEST_CKPT_INTERVAL <= 0 )); then
            echo "CHECKPOINT_MODE=latest requires LATEST_CKPT_INTERVAL > 0" >&2
            exit 2
        fi
        CHECKPOINT_ARGS=(--latest-ckpt-interval "${LATEST_CKPT_INTERVAL}")
    elif [[ "${CHECKPOINT_MODE}" == "latest_relay" ]]; then
        if (( LATEST_CKPT_INTERVAL <= 0 )); then
            echo "CHECKPOINT_MODE=latest_relay requires LATEST_CKPT_INTERVAL > 0" >&2
            exit 2
        fi
        CHECKPOINT_ARGS=(
            --inter-ckpts 67911
            --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
            --upload-latest-ckpt-to-wandb
        )
    else
        echo "Unsupported CHECKPOINT_MODE=${CHECKPOINT_MODE}" >&2
        exit 2
    fi
    EXTRA_ARGS=(
        --downstream-eval-enabled
        --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled
        --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
        "${CHECKPOINT_ARGS[@]}"
        --wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-tags fineweb optimizer_fp8 bf16_model "${HORIZON_TAG}" 0.5B "${NPROC_PER_NODE}gpu" cloudru h100 "${OPTIMIZER}" "seed${RUN_SEED}" torch291 efficient-image h200-data-parity
    )
else
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
fi

expected_batch_size=$((32 / NPROC_PER_NODE))
expected_acc_steps=$((4 * NPROC_PER_NODE))
expected_replay_world_size=1
if (( NPROC_PER_NODE == 1 )); then
    expected_replay_world_size=2
fi
if (( BATCH_SIZE != expected_batch_size || ACC_STEPS != expected_acc_steps )); then
    echo "H200 parity requires pre-DDP batch_size=${expected_batch_size} " \
         "and acc_steps=${expected_acc_steps} for ${NPROC_PER_NODE} GPU(s)" >&2
    exit 7
fi
if (( BATCH_SIZE * ACC_STEPS != 128 )); then
    echo "1xChinchilla pre-DDP effective batch must be 128" >&2
    exit 7
fi
if (( FINEWEB_REPLAY_WORLD_SIZE != expected_replay_world_size )); then
    echo "H200 parity requires FINEWEB_REPLAY_WORLD_SIZE=${expected_replay_world_size} " \
         "for ${NPROC_PER_NODE} GPU(s)" >&2
    exit 7
fi
echo "H200_PARITY_CONFIG=ok pre_ddp_batch=${BATCH_SIZE} " \
     "pre_ddp_acc_steps=${ACC_STEPS} physical_batch=${expected_batch_size} " \
     "physical_acc_steps=4 global_batch=128"

if [[ "${OPTIMIZER}" != "muon" && "${OPTIMIZER}" != "soap" ]]; then
    echo "Unsupported OPTIMIZER=${OPTIMIZER}" >&2
    exit 2
fi

"${TRAIN_LAUNCHER[@]}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    --seed "${RUN_SEED}" \
    --dataset fineweb \
    --datasets-dir "${DATASETS_DIR}" \
    --fineweb-replay-world-size "${FINEWEB_REPLAY_WORLD_SIZE}" \
    --fineweb-replay-layout "${FINEWEB_REPLAY_LAYOUT}" \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length 1024 \
    --data-seed 1337 \
    --streaming \
    --workers 8 \
    --model llama \
    --n-layer 18 \
    --n-embd 1280 \
    --n-head 20 \
    --multiple-of 256 \
    --dtype bfloat16 \
    --opt "${OPTIMIZER}" \
    --lr 1e-3 \
    --weight-decay "${WEIGHT_DECAY}" \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad-clip "${GRAD_CLIP}" \
    --scheduler wsd \
    --warmup-steps "${WARMUP_STEPS}" \
    --iterations "${ITERATIONS}" \
    --wsd-fract-decay 0.1 \
    --wsd-final-lr-scale 0.0 \
    --decay-type cosine \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size 32 \
    --acc-steps "${ACC_STEPS}" \
    --fp8-optim \
    --fp8-qgroup-size 128 \
    --fp8-first-order-bit E4M3 \
    --fp8-second-order-bit E4M3 \
    --fp8-expansion expand \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-batches "${EVAL_BATCHES}" \
    --log-interval "${RUN_LOG_INTERVAL}" \
    --results-base-folder "${RESULTS_DIR}" \
    "${EXTRA_ARGS[@]}"
train_status=$?
if (( train_status == 0 )) && [[ "${MODE}" == "smoke" ]]; then
    touch "${SMOKE_MARKER}"
    echo "SMOKE_MARKER=${SMOKE_MARKER}"
fi
exit "${train_status}"
