#!/usr/bin/env bash
set -euo pipefail

readonly nfs3_root=/workspace-SR006.nfs3/dimativator
readonly slim_root="${nfs3_root}/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states"
readonly muon_root="${nfs3_root}/muon-fp8-states-500m-2xc-2gpu-20260918/llama-18L20H_fineweb_150k/llama500M_muon_fp8_states_bf16_wd1e-4_2xC_2gpu"
readonly -a targets=(
    "${nfs3_root}/checkpoints/slimadam_257m_fp8_states_8xC/160000"
    "${slim_root}/257m_slim_adam_fp8_states_8xC_cloud_4gpu/ckpts"
    "${slim_root}/257m_slim_adam_fp8_states_1xC_decay_cloud_4gpu/ckpts"
    "${slim_root}/257m_slim_adam_fp8_states_2xC_decay_cloud_4gpu/ckpts"
    "${slim_root}/257m_slim_adam_fp8_states_4xC_decay_cloud_4gpu/ckpts"
    "${slim_root}/257m_slim_adam_fp8_states_4xC_decay_cloud_4gpu_extended/ckpts"
    "${slim_root}/257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100/ckpts"
    "${muon_root}/ckpts"
)

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs3

for target in "${targets[@]}"; do
    case "${target}" in
        "${nfs3_root}/checkpoints/slimadam_257m_fp8_states_8xC/160000"|\
        "${slim_root}"/257m_slim_adam_fp8_states_*/ckpts|\
        "${muon_root}/ckpts") ;;
        *)
            echo "REFUSE_UNEXPECTED_TARGET=${target}" >&2
            exit 2
            ;;
    esac
    if [[ -L "${target}" ]]; then
        echo "REFUSE_SYMLINK=${target}" >&2
        exit 2
    fi
    if [[ -d "${target}" ]]; then
        du -sh -- "${target}"
    else
        echo "ALREADY_MISSING=${target}"
    fi
done

for target in "${targets[@]}"; do
    if [[ -d "${target}" && ! -L "${target}" ]]; then
        rm -rf -- "${target}"
        echo "DELETED=${target}"
    fi
done

for target in "${targets[@]}"; do
    if [[ -e "${target}" ]]; then
        echo "DELETE_FAILED=${target}" >&2
        exit 3
    fi
done

echo "STAGE3_FP8_CHECKPOINT_CLEANUP_COMPLETE count=${#targets[@]}"
df -h /workspace-SR006.nfs3
