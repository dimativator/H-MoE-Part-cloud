#!/usr/bin/env bash
set -euo pipefail

readonly experiment_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_4gpu
readonly stop_file=$(mktemp)
trap 'rm -f "${stop_file}"' EXIT

python scripts/cloud/relay_latest_checkpoint_to_hf.py \
    --checkpoint-dir "${experiment_dir}/ckpts/latest" \
    --staging-dir "${experiment_dir}/ckpts/.relay" \
    --repo-id DimaTivator/slimadam-257m-cloud-relay-20260915 \
    --remote-prefix slimadam_257m_fp8_states_8xC \
    --world-size 4 \
    --stop-file "${stop_file}" \
    --poll-seconds 5
