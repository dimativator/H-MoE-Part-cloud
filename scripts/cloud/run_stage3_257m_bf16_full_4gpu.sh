#!/usr/bin/env bash
set -euo pipefail

: "${OPTIMIZER:?OPTIMIZER must be frugal or slim_adam}"
MODE=${MODE:-full}
case "${OPTIMIZER}" in
    frugal)
        label=frugal
        opt_args=(--opt coord_adamw --lr 1e-3 --weight-decay 1e-1 \
            --beta1 0.9 --beta2 0.99 --density 0.25 --update_gap 50 \
            --coord_choice columns --grad-clip 1.0)
        ;;
    slim_adam)
        label=slim_adam
        opt_args=(--opt slim_adam --lr 5e-4 --weight-decay 1e-1 \
            --beta1 0.9 --beta2 0.99 --grad-clip 1.0)
        ;;
    *)
        echo "Unsupported OPTIMIZER=${OPTIMIZER}" >&2
        exit 2
        ;;
esac

readonly packed_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly remote_dir=/workspace-SR006.nfs3/dimativator/fineweb-remote
readonly manifest_b64=scripts/cloud/fineweb_h200_manifest.json.gz.b64
readonly results_dir=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly log_dir="${results_dir}/logs"
readonly eval_cache_dir=/home/jovyan/evals_cache
readonly group=8xChinchilla_257M_bf16_native_states
readonly experiment="257m_${label}_bf16_native_states_8xC_cloud_4gpu"

rank=${OMPI_COMM_WORLD_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
if (( world_size > 1 )); then
    [[ "${world_size}" == 4 ]] || { echo "Exactly four MPI ranks required" >&2; exit 3; }
    export RANK=${RANK:-${rank}}
    export WORLD_SIZE=${WORLD_SIZE:-${world_size}}
    export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    launcher=(python)
else
    launcher=(torchrun --standalone --nproc_per_node=4)
fi

mkdir -p "${results_dir}" "${log_dir}" "${eval_cache_dir}" "${results_dir}/wandb_offline"
log_file="${log_dir}/${experiment}_${MODE}_rank${rank}.log"
exec > >(tee -a "${log_file}") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${results_dir}/wandb_offline"
export FINEWEB_LOG_DATA_HASHES=1

if (( rank == 0 )); then
    echo GPU_STATUS_BEFORE_LAUNCH
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
    df -h /home/jovyan /workspace-SR006.nfs3
fi

common_args=(
    --experiment-name "${experiment}"
    --distributed-backend nccl
    --seed 0 --data-seed 1337
    --dataset fineweb --eval-cache-dir "${eval_cache_dir}"
    --sequence-length 1024 --streaming --workers 8
    --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256
    --dtype bfloat16
    "${opt_args[@]}"
    --scheduler wsd --wsd-final-lr-scale 0 --wsd-fract-decay 0.1 --decay-type cosine
    --iterations 314000 --warmup-steps 2000
    --batch-size 8 --eval-batch-size 8 --acc-steps 16
    --eval-interval 500 --eval-batches 128
    --downstream-eval-enabled --downstream-eval-interval 2000
    --downstream-task-group basic_v2
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103
    --log-interval 50
    --results-base-folder "${results_dir}"
    --wandb --wandb-project fp8-pretrain --wandb-group "${group}"
    --wandb-tags bf16_model native_optimizer_states no_fp8_optim 257M "${label}" full_8xc cloudru 4gpu local_offline
)

if [[ "${MODE}" == smoke ]]; then
    "${launcher[@]}" src/main.py \
        "${common_args[@]}" \
        --datasets-dir "${packed_dir}" \
        --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
        --early-stop-iteration 3 --eval-interval 3 --eval-batches 1 \
        --no-local-save
    echo "BF16_SMOKE_COMPLETE optimizer=${OPTIMIZER} rank=${rank}"
    exit 0
fi
[[ "${MODE}" == full || "${MODE}" == resume ]] || {
    echo "MODE must be smoke, full, or resume" >&2
    exit 2
}

if [[ "${MODE}" == full ]]; then
    for source_rank in 0 1; do
        state_file="${packed_dir}/train_rank${source_rank}.third.state.json"
        [[ -s "${state_file}" ]] || { echo "Missing ${state_file}" >&2; exit 4; }
    done

    echo "FULL_PHASE1_START optimizer=${OPTIMIZER} rank=${rank}"
    "${launcher[@]}" src/main.py \
        "${common_args[@]}" \
        --datasets-dir "${packed_dir}" \
        --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
        --early-stop-iteration 157100 \
        --inter-ckpts 35325 70650 141300 \
        --latest-ckpt-interval 100
    echo "FULL_PHASE1_COMPLETE optimizer=${OPTIMIZER} rank=${rank}"
else
    latest_checkpoint="${results_dir}/${group}/${experiment}/ckpts/latest/main.pt"
    [[ -s "${latest_checkpoint}" ]] || {
        echo "Missing resume checkpoint ${latest_checkpoint}" >&2
        exit 5
    }
    echo "FULL_PHASE1_SKIPPED optimizer=${OPTIMIZER} rank=${rank} mode=resume"
fi

manifest_file="/dev/shm/fineweb_h200_manifest_${label}_${rank}.json"
python - "${manifest_b64}" "${manifest_file}" <<'PY'
import base64
import gzip
import sys
from pathlib import Path

payload = base64.b64decode(Path(sys.argv[1]).read_text().strip())
Path(sys.argv[2]).write_bytes(gzip.decompress(payload))
PY
trap 'rm -f "${manifest_file}"' EXIT

echo "FULL_PHASE2_START optimizer=${OPTIMIZER} rank=${rank}"
"${launcher[@]}" src/main.py \
    "${common_args[@]}" \
    --datasets-dir "${remote_dir}" --fineweb-manifest "${manifest_file}" \
    --fineweb-live-source-state-dir "${packed_dir}" \
    --fineweb-live-source-world-size 2 \
    --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
    --skip-train-reader-state-on-resume \
    --latest-ckpt-interval 5000
echo "FULL_8XC_COMPLETE optimizer=${OPTIMIZER} rank=${rank}"

if (( rank == 0 )); then
    python - "${results_dir}/${group}/${experiment}/ckpts/latest" <<'PY'
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1])
expected_parent = "ckpts"
if path.name != "latest" or path.parent.name != expected_parent:
    raise RuntimeError(f"Refusing unexpected cleanup path: {path}")
if path.exists():
    shutil.rmtree(path)
print("FINAL_LATEST_CHECKPOINT_REMOVED")
PY
fi
