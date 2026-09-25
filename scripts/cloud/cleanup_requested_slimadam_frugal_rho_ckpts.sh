#!/usr/bin/env bash
set -euo pipefail

readonly slim_root=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921/8xChinchilla_257M_bf16_native_states
readonly slim_run=${slim_root}/257m_slim_adam_bf16_native_states_8xC_cloud_4gpu
readonly frugal_root=/workspace-SR006.nfs2/dimativator/frugal-rho-257m-1xc-wsd-20260915/1xChinchilla_257M_frugal_rho_wsd_cloud
readonly -a targets=(
    "${slim_run}/ckpts/latest"
    "${frugal_root}/llama257M_frugal_rho0p1_bf16_wsd_lr1e-3_wd0p1_1xC_1gpu/ckpts/latest"
    "${frugal_root}/llama257M_frugal_rho0p15_bf16_wsd_lr1e-3_wd0p1_1xC_1gpu/ckpts/latest"
    "${frugal_root}/llama257M_frugal_rho0p2_bf16_wsd_lr1e-3_wd0p1_1xC_1gpu/ckpts/latest"
    "${frugal_root}/llama257M_frugal_rho0p3_bf16_wsd_lr1e-3_wd0p1_1xC_1gpu/ckpts/latest"
)

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds) APPLY=${APPLY:-0}"
df -h /home/jovyan /workspace-SR006.nfs2

for target in "${targets[@]}"; do
    if [[ ! -d "${target}" ]]; then
        echo "MISSING=${target}"
        continue
    fi
    if [[ -L "${target}" || -L "$(dirname -- "${target}")" || ! -f "${target}/main.pt" ]]; then
        echo "UNSAFE_OR_NOT_CHECKPOINT=${target}" >&2
        exit 2
    fi
    resolved=$(realpath -e -- "${target}")
    if [[ "${resolved}" != "${target}" ]]; then
        echo "NON_CANONICAL=${target} RESOLVED=${resolved}" >&2
        exit 2
    fi
    du -sh -- "${target}"
    stat -c 'MAIN_BYTES=%s MAIN_MODIFIED=%y' -- "${target}/main.pt"
done

if [[ "${APPLY:-0}" != "1" ]]; then
    echo "AUDIT_ONLY: set APPLY=1 to remove these exact checkpoint directories"
    exit 0
fi

for target in "${targets[@]}"; do
    if [[ -d "${target}" ]]; then
        echo "DELETING=${target}"
        rm -rf -- "${target}"
    fi
    if [[ -e "${target}" ]]; then
        echo "DELETE_FAILED=${target}" >&2
        exit 3
    fi
    echo "DELETED_OR_ABSENT=${target}"
done

df -h /home/jovyan /workspace-SR006.nfs2
