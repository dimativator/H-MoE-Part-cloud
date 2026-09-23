#!/usr/bin/env bash
set -euo pipefail

: "${OPTIMIZER:?OPTIMIZER must be frugal or slim_adam}"
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
readonly results_dir=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly log_dir="${results_dir}/logs"
readonly eval_cache_dir=/home/jovyan/evals_cache
readonly group=8xChinchilla_257M_bf16_native_states
readonly full_experiment="257m_${label}_bf16_native_states_8xC_cloud_4gpu"
readonly checkpoint_root="${results_dir}/${group}/${full_experiment}/ckpts"
readonly start_scale=${START_SCALE:-1}
readonly run_suffix=${RUN_SUFFIX:-}
readonly keep_source_checkpoint=${KEEP_SOURCE_CHECKPOINT:-0}
[[ "${start_scale}" == 1 || "${start_scale}" == 2 || "${start_scale}" == 4 ]] || {
    echo "START_SCALE must be 1, 2, or 4" >&2
    exit 2
}
[[ "${run_suffix}" =~ ^[a-zA-Z0-9_-]*$ ]] || {
    echo "RUN_SUFFIX must contain only letters, digits, underscores, or hyphens" >&2
    exit 2
}
[[ "${keep_source_checkpoint}" == 0 || "${keep_source_checkpoint}" == 1 ]] || {
    echo "KEEP_SOURCE_CHECKPOINT must be 0 or 1" >&2
    exit 2
}

mkdir -p "${results_dir}" "${log_dir}" "${eval_cache_dir}" "${results_dir}/wandb_offline"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${results_dir}/wandb_offline"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo GPU_STATUS_BEFORE_LAUNCH
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

wait_for_checkpoint() {
    local iteration=$1
    local checkpoint_dir="${checkpoint_root}/${iteration}"
    echo "WAITING_FOR_CHECKPOINT optimizer=${OPTIMIZER} iteration=${iteration}"
    while true; do
        if [[ -s "${checkpoint_dir}/main.pt" ]] && \
           [[ -s "${checkpoint_dir}/worker_0.pt" ]] && \
           [[ -s "${checkpoint_dir}/worker_1.pt" ]] && \
           [[ -s "${checkpoint_dir}/worker_2.pt" ]] && \
           [[ -s "${checkpoint_dir}/worker_3.pt" ]]; then
            if python - "${checkpoint_dir}" "${iteration}" <<'PY'
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
expected = int(sys.argv[2])
main = torch.load(root / "main.pt", map_location="cpu", weights_only=False)
if int(main["itr"]) != expected:
    raise RuntimeError(f"checkpoint itr={main['itr']} expected={expected}")
for rank in range(4):
    torch.load(root / f"worker_{rank}.pt", map_location="cpu", weights_only=False)
PY
            then
                echo "CHECKPOINT_READY optimizer=${OPTIMIZER} iteration=${iteration}"
                return 0
            fi
        fi
        sleep 60
    done
}

run_decay() {
    local scale=$1 start=$2 end=$3
    local checkpoint_dir="${checkpoint_root}/${start}"
    local experiment="257m_${label}_bf16_native_states_${scale}xC_decay_cloud_1gpu${run_suffix}"
    local log_file="${log_dir}/${experiment}.log"
    wait_for_checkpoint "${start}"
    echo "DECAY_START optimizer=${OPTIMIZER} scale=${scale} start=${start} end=${end}" | tee -a "${log_file}"
    python src/main.py \
        --experiment-name "${experiment}" \
        --seed 0 --data-seed 1337 \
        --dataset fineweb --datasets-dir "${packed_dir}" \
        --fineweb-replay-world-size 2 --fineweb-replay-layout concat \
        --eval-cache-dir "${eval_cache_dir}" \
        --sequence-length 1024 --streaming --workers 8 \
        --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 \
        --dtype bfloat16 \
        "${opt_args[@]}" \
        --scheduler wsd --wsd-final-lr-scale 0 --decay-type cosine \
        --batch-size 32 --eval-batch-size 32 --acc-steps 4 \
        --resume-from "${checkpoint_dir}" --decay-from-checkpoint \
        --skip-train-reader-state-on-resume \
        --warmup-steps 0 --iterations "${end}" --wsd-fract-decay 1.0 \
        --eval-interval 500 --eval-batches 32 \
        --downstream-eval-enabled --downstream-eval-interval 2000 \
        --downstream-task-group basic_v2 \
        --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
        --log-interval 50 --no-local-save \
        --results-base-folder "${results_dir}" \
        --wandb --wandb-project fp8-pretrain --wandb-group "${group}" \
        --wandb-tags bf16_model native_optimizer_states no_fp8_optim 257M "${label}" "decay_${scale}xc" cloudru 1gpu local_offline \
        >> "${log_file}" 2>&1
    echo "DECAY_COMPLETE optimizer=${OPTIMIZER} scale=${scale} end=${end}" | tee -a "${log_file}"
    if [[ "${keep_source_checkpoint}" == 1 ]]; then
        echo "SOURCE_CHECKPOINT_PRESERVED=${start}"
        return 0
    fi
    python - "${checkpoint_dir}" "${start}" <<'PY'
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1])
expected = sys.argv[2]
if path.name != expected or path.parent.name != "ckpts":
    raise RuntimeError(f"Refusing unexpected cleanup path: {path}")
shutil.rmtree(path)
print(f"SOURCE_CHECKPOINT_REMOVED={expected}")
PY
}

if (( start_scale <= 1 )); then
    run_decay 1 35325 39250
fi
if (( start_scale <= 2 )); then
    run_decay 2 70650 78500
fi
run_decay 4 141300 157000
echo "DECAY_QUEUE_COMPLETE optimizer=${OPTIMIZER}"
