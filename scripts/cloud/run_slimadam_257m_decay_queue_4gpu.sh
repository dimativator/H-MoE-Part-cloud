#!/usr/bin/env bash
set -euo pipefail

readonly results_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud
readonly checkpoint_root="${results_dir}/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_4gpu/ckpts"
readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud
readonly data_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly eval_cache_dir=/home/jovyan/evals_cache

rank=${OMPI_COMM_WORLD_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
if [[ "${world_size}" != 4 ]]; then
    echo "This decay queue requires exactly four MPI ranks" >&2
    exit 2
fi
if [[ ! -f "${checkpoint_root}/slimadam_decay_sources_ready" ]]; then
    echo "Verified decay checkpoints are not ready" >&2
    exit 3
fi
for iteration in 35325 70650 141300; do
    for worker in 0 1 2 3; do
        [[ -s "${checkpoint_root}/${iteration}/worker_${worker}.pt" ]] || exit 4
    done
    [[ -s "${checkpoint_root}/${iteration}/main.pt" ]] || exit 4
done
[[ -f "${data_dir}/packed_metadata.json" ]] || exit 4

export RANK=${RANK:-${rank}}
export WORLD_SIZE=${WORLD_SIZE:-${world_size}}
export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DIR="${results_dir}/wandb_offline"
mkdir -p "${log_dir}" "${eval_cache_dir}" "${WANDB_DIR}"

run_decay() {
    local scale=$1 start=$2 end=$3
    local experiment="257m_slim_adam_fp8_states_${scale}xC_decay_cloud_4gpu"
    local log_file="${log_dir}/${experiment}_rank${rank}.log"
    echo "DECAY_START scale=${scale} start=${start} end=${end} rank=${rank} $(date -Is)" | tee -a "${log_file}"
    set +e
    python src/main.py \
        --experiment-name "${experiment}" \
        --distributed-backend nccl \
        --seed 0 --data-seed 1337 \
        --dataset fineweb --datasets-dir "${data_dir}" \
        --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
        --eval-cache-dir "${eval_cache_dir}" \
        --sequence-length 1024 --streaming --workers 8 \
        --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 \
        --dtype bfloat16 --opt slim_adam --lr 5e-4 --weight-decay 1e-1 \
        --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
        --scheduler wsd --wsd-final-lr-scale 0 --decay-type cosine \
        --batch-size 8 --eval-batch-size 8 --acc-steps 16 \
        --fp8-optim --fp8-qgroup-size 128 \
        --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3 \
        --fp8-expansion expand \
        --resume-from "${checkpoint_root}/${start}" --decay-from-checkpoint \
        --warmup-steps 0 --iterations "${end}" --wsd-fract-decay 1.0 \
        --eval-interval 500 --eval-batches 128 \
        --downstream-eval-enabled --downstream-eval-interval 2000 \
        --downstream-task-group basic_v2 \
        --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
        --log-interval 50 --latest-ckpt-interval 1000 \
        --results-base-folder "${results_dir}" \
        --wandb --wandb-project fp8-pretrain \
        --wandb-group 8xChinchilla_257M_fp8_states \
        --wandb-tags optimizer_fp8 bf16_model 257M slim_adam "decay_${scale}xc" cloudru 4gpu local_offline \
        >> "${log_file}" 2>&1
    local status=$?
    set -e
    echo "DECAY_EXIT scale=${scale} status=${status} rank=${rank} $(date -Is)" | tee -a "${log_file}"
    return "${status}"
}

run_decay 1 35325 39250
run_decay 2 70650 78500
run_decay 4 141300 157000
echo "SLIMADAM_DECAY_QUEUE_COMPLETE rank=${rank} $(date -Is)"
