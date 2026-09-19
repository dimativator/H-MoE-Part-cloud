#!/usr/bin/env bash
set -euo pipefail

readonly results_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud
readonly checkpoint_dir="${results_dir}/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100/ckpts/latest"
readonly manifest_path=/home/jovyan/fineweb_h200_manifest.json

for file in "${checkpoint_dir}/main.pt" "${checkpoint_dir}/worker_0.pt"; do
    if [[ ! -s "${file}" ]]; then
        echo "Missing checkpoint file: ${file}" >&2
        exit 4
    fi
done

base64 --decode scripts/cloud/fineweb_h200_manifest.json.gz.b64 \
    | gzip --decompress > "${manifest_path}"

export MODE=full
export FINEWEB_MANIFEST="${manifest_path}"
export RESUME_FROM="${checkpoint_dir}"
export HF_RELAY_REPO_ID=
exec bash scripts/cloud/run_slimadam_257m_1gpu_h100_resume.sh
