#!/usr/bin/env bash
set -euo pipefail

# Two independent Cloud.ru queues run A=(bf16, fp8_act) and
# B=(fp8_full, bf16_fp8_states). Every precision first trains the 2xC trunk,
# then creates the independent 1xC decay branch from trunk step 67911.

QUEUE_ID=${QUEUE_ID:-A}
MODE=${MODE:-full}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-20260912}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-5000}
RUN_SEED=${SEED:-0}

if [[ "${NPROC_PER_NODE}" != "2" ]]; then
    echo "This entrypoint requires exactly two GPUs" >&2
    exit 2
fi
case "${RESULTS_DIR}" in
    /workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-*) ;;
    *)
        echo "Refusing checkpoint cleanup outside the dedicated experiment root: ${RESULTS_DIR}" >&2
        exit 2
        ;;
esac
case "${QUEUE_ID}" in
    A) PRECISIONS=(bf16 fp8_act) ;;
    B) PRECISIONS=(fp8_full bf16_fp8_states) ;;
    *) echo "QUEUE_ID must be A or B" >&2; exit 2 ;;
esac
case "${MODE}" in
    smoke|full) ;;
    *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
MPI_RANK=${OMPI_COMM_WORLD_RANK:-0}
MPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if (( MPI_SIZE > 1 && MPI_SIZE != NPROC_PER_NODE )); then
    echo "mlsub MPI world size ${MPI_SIZE} != ${NPROC_PER_NODE}" >&2
    exit 8
fi

mkdir -p "${RESULTS_DIR}/logs" "${EVAL_CACHE_DIR}"
QUEUE_LOG="${RESULTS_DIR}/logs/queue_${QUEUE_ID}_${MODE}_rank${MPI_RANK}.log"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "QUEUE_ID=${QUEUE_ID} MODE=${MODE} MPI_RANK=${MPI_RANK}/${MPI_SIZE}"
echo "DATASETS_DIR=${DATASETS_DIR} RESULTS_DIR=${RESULTS_DIR}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

PYTHON_BIN=$(command -v python)
TORCHRUN_BIN=$(command -v torchrun)
if (( MPI_SIZE > 1 )); then
    export RANK=${RANK:-${MPI_RANK}}
    export WORLD_SIZE=${WORLD_SIZE:-${MPI_SIZE}}
    export LOCAL_RANK=${LOCAL_RANK:-${MPI_LOCAL_RANK}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("${PYTHON_BIN}")
else
    TRAIN_LAUNCHER=("${TORCHRUN_BIN}" --standalone --nproc_per_node=2)
fi

"${PYTHON_BIN}" - <<'PY'
import torch
import torchao
import triton

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
assert torch.cuda.is_available()
assert torchao.__version__.startswith("0.15.0"), torchao.__version__
assert triton.__version__ == "3.5.1", triton.__version__
print("ENVIRONMENT_CHECK=ok", torch.__version__, torch.version.cuda)
PY

if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Packed H200 data is missing: ${DATASETS_DIR}" >&2
    exit 5
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# The platform home is shared by both MPI ranks and sometimes by concurrent
# jobs. Triton cache writes are not safe on that NFS mount.
export TRITON_CACHE_DIR="/tmp/triton-${QUEUE_ID}-${MODE}-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"

wait_for_marker() {
    local marker=$1
    if (( MPI_RANK == 0 )); then
        touch "${marker}"
        return
    fi
    for _ in $(seq 1 300); do
        [[ -f "${marker}" ]] && return
        sleep 1
    done
    echo "Timed out waiting for rank-0 marker: ${marker}" >&2
    exit 9
}

remove_checkpoint_tree() {
    local target=$1
    case "${target}" in
        "${RESULTS_DIR}/"*/ckpts|"${RESULTS_DIR}/"*/ckpts/latest) ;;
        *) echo "Refusing unsafe checkpoint removal: ${target}" >&2; exit 11 ;;
    esac
    rm -rf -- "${target}"
}

precision_args() {
    local precision=$1
    PRECISION_ARGS=()
    case "${precision}" in
        bf16) ;;
        fp8_act)
            PRECISION_ARGS=(
                --fp8 --fp8-fabit E4M3 --fp8-fwbit E4M3
                --fp8-babit E5M2 --fp8-bwbit E5M2 --fp8-group-size 16
            )
            ;;
        bf16_fp8_states)
            PRECISION_ARGS=(
                --fp8-optim --fp8-qgroup-size 128
                --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3
                --fp8-expansion expand
            )
            ;;
        fp8_full)
            PRECISION_ARGS=(
                --fp8 --fp8-fabit E4M3 --fp8-fwbit E4M3
                --fp8-babit E5M2 --fp8-bwbit E5M2 --fp8-group-size 16
                --fp8-optim --fp8-qgroup-size 128
                --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3
                --fp8-expansion expand
            )
            ;;
        *) echo "Unknown precision: ${precision}" >&2; exit 2 ;;
    esac
}

common_args() {
    local experiment_name=$1
    local group=$2
    local metrics_file=$3
    COMMON_ARGS=(
        --distributed-backend nccl
        --experiment-name "${experiment_name}"
        --seed "${RUN_SEED}"
        --data-seed 1337
        --dataset fineweb
        --datasets-dir "${DATASETS_DIR}"
        --fineweb-replay-world-size 1
        --fineweb-replay-layout concat
        --eval-cache-dir "${EVAL_CACHE_DIR}"
        --sequence-length 1024
        --streaming --workers 8
        --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256
        --dtype bfloat16
        --opt coord_muon --non_proj_opt adamw
        --lr 2e-3 --weight-decay 0.1
        --beta1 0.9 --beta2 0.99
        --momentum 0.95 --nesterov --muon_ns_steps 5
        --density 0.25 --update_gap 50 --coord_choice columns
        --grad-clip 1.0
        --batch-size 16 --eval-batch-size 32 --acc-steps 8
        --results-base-folder "${RESULTS_DIR}"
        --wandb-group "${group}"
        --metrics-jsonl "${metrics_file}"
    )
}

run_smoke() {
    local precision=$1
    local experiment="smoke_frugal_muon_500m_${precision}_2gpu"
    local group="smoke_frugal_muon_500m_2gpu"
    local exp_dir="${RESULTS_DIR}/${group}/${experiment}"
    local marker="${RESULTS_DIR}/.${experiment}.done"
    [[ -f "${marker}" ]] && return
    precision_args "${precision}"
    common_args "${experiment}" "${group}" "${exp_dir}/metrics.jsonl"
    if [[ -s "${exp_dir}/metrics.jsonl" ]]; then
        for required in main.pt worker_0.pt worker_1.pt; do
            [[ -f "${exp_dir}/ckpts/latest/${required}" ]] || {
                echo "Refusing to restart an existing trunk without a complete latest checkpoint: ${exp_dir}/ckpts/latest" >&2
                exit 12
            }
        done
    fi
    "${TRAIN_LAUNCHER[@]}" src/main.py \
        "${COMMON_ARGS[@]}" "${PRECISION_ARGS[@]}" \
        --scheduler none --warmup-steps 0 --iterations 3 \
        --eval-interval 3 --eval-batches 1 --log-interval 1 \
        --no-local-save
    wait_for_marker "${marker}"
}

run_trunk() {
    local precision=$1
    local experiment="llama500M_frugal_muon_adamw_${precision}_2xC_2gpu"
    local group="2xChinchilla_500M_frugal_muon_2gpu_cloud"
    local exp_dir="${RESULTS_DIR}/${group}/${experiment}"
    local marker="${RESULTS_DIR}/.${experiment}.done"
    [[ -f "${marker}" ]] && return
    precision_args "${precision}"
    common_args "${experiment}" "${group}" "${exp_dir}/metrics.jsonl"
    "${TRAIN_LAUNCHER[@]}" src/main.py \
        "${COMMON_ARGS[@]}" "${PRECISION_ARGS[@]}" \
        --scheduler wsd --warmup-steps 7000 --iterations 150914 \
        --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine \
        --eval-interval 500 --eval-batches 32 --log-interval 50 \
        --downstream-eval-enabled --downstream-eval-interval 2000 \
        --downstream-task-group basic_v2 \
        --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
        --inter-ckpts 67911 --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
    wait_for_marker "${marker}"
    if (( MPI_RANK == 0 )); then
        remove_checkpoint_tree "${exp_dir}/ckpts/latest"
    fi
    echo "TRUNK_COMPLETE precision=${precision}"
}

run_decay_branch() {
    local precision=$1
    local trunk_experiment="llama500M_frugal_muon_adamw_${precision}_2xC_2gpu"
    local trunk_group="2xChinchilla_500M_frugal_muon_2gpu_cloud"
    local trunk_dir="${RESULTS_DIR}/${trunk_group}/${trunk_experiment}"
    local experiment="llama500M_frugal_muon_adamw_${precision}_1xC_resume_2gpu"
    local group="1xChinchilla_resume_500M_frugal_muon_2gpu_cloud"
    local exp_dir="${RESULTS_DIR}/${group}/${experiment}"
    local phase_marker="${RESULTS_DIR}/.${experiment}.phase_done"
    local final_marker="${RESULTS_DIR}/.${experiment}.done"
    if [[ -f "${final_marker}" ]]; then
        return
    fi
    if [[ -f "${phase_marker}" ]]; then
        if (( MPI_RANK == 0 )); then
            remove_checkpoint_tree "${trunk_dir}/ckpts"
            remove_checkpoint_tree "${exp_dir}/ckpts"
        fi
        wait_for_marker "${final_marker}"
        return
    fi

    local source_ckpt="${trunk_dir}/ckpts/67911"
    local branch_latest="${exp_dir}/ckpts/latest"
    precision_args "${precision}"
    common_args "${experiment}" "${group}" "${exp_dir}/metrics.jsonl"
    RESUME_ARGS=()
    if [[ -f "${branch_latest}/main.pt" && -f "${branch_latest}/worker_0.pt" && -f "${branch_latest}/worker_1.pt" ]]; then
        RESUME_ARGS=(--resume-from "${branch_latest}")
    else
        for required in main.pt worker_0.pt worker_1.pt; do
            [[ -f "${source_ckpt}/${required}" ]] || {
                echo "Missing 1xC branch checkpoint file: ${source_ckpt}/${required}" >&2
                exit 10
            }
        done
        RESUME_ARGS=(--resume-from "${source_ckpt}" --decay-from-checkpoint)
    fi

    "${TRAIN_LAUNCHER[@]}" src/main.py \
        "${COMMON_ARGS[@]}" "${PRECISION_ARGS[@]}" "${RESUME_ARGS[@]}" \
        --scheduler wsd --warmup-steps 0 --iterations 75457 \
        --wsd-fract-decay 1.0 --wsd-final-lr-scale 0 --decay-type cosine \
        --eval-interval 500 --eval-batches 32 --log-interval 50 \
        --downstream-eval-enabled --downstream-eval-interval 2000 \
        --downstream-task-group basic_v2 \
        --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
        --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
    wait_for_marker "${phase_marker}"
    if (( MPI_RANK == 0 )); then
        remove_checkpoint_tree "${trunk_dir}/ckpts"
        remove_checkpoint_tree "${exp_dir}/ckpts"
    fi
    wait_for_marker "${final_marker}"
    echo "DECAY_COMPLETE precision=${precision}"
}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for precision in "${PRECISIONS[@]}"; do
    if [[ "${MODE}" == "smoke" ]]; then
        run_smoke "${precision}"
    else
        run_trunk "${precision}"
        run_decay_branch "${precision}"
    fi
done

echo "QUEUE_COMPLETE=${QUEUE_ID} MODE=${MODE}"
