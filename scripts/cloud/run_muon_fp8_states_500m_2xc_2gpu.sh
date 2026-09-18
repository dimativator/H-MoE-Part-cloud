#!/usr/bin/env bash
set -euo pipefail

# Train the 500M Llama model from scratch to the 2xChinchilla endpoint on two
# H100s. Model compute is BF16; only the Muon optimizer states use FP8.

export MODE=${MODE:-full}
export OPTIMIZER=muon
export NPROC_PER_NODE=2
export BATCH_SIZE=16
export ACC_STEPS=8
export ITERATIONS=150914
export WARMUP_STEPS=2000
export WEIGHT_DECAY=1e-4
export CHECKPOINT_MODE=${CHECKPOINT_MODE:-latest}
export LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-5000}
export DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
export RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/muon-fp8-states-500m-2xc-2gpu-20260918}
export LOG_DIR=${LOG_DIR:-${RESULTS_DIR}/logs}
export WANDB_GROUP=${WANDB_GROUP:-2xChinchilla_500M_muon_fp8_states_2gpu_cloud}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama500M_muon_fp8_states_bf16_wd1e-4_2xC_2gpu}
export HORIZON_TAG=2xChinchilla
export FINEWEB_REPLAY_WORLD_SIZE=1
export FINEWEB_REPLAY_LAYOUT=concat
export SEED=${SEED:-0}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${script_dir}/run_muon_efficient_image.sh"
