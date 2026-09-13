#!/usr/bin/env bash
set -euo pipefail

readonly root=/workspace-SR006.nfs3/dimativator/galore-rho-257m-1xc-20260912
readonly group=${root}/1xChinchilla_257M_galore_rho_cloud

remove_checkpoint_tree() {
    local target=$1
    case "${target}" in
        "${group}/llama257M_galore_rho0p7_bf16_lr1e-3_wd0p1_1xC_1gpu/ckpts"|\
        "${group}/llama257M_galore_rho0p8_bf16_lr1e-3_wd0p1_1xC_1gpu/ckpts") ;;
        *) echo "Refusing unsafe checkpoint removal: ${target}" >&2; exit 2 ;;
    esac
    if [[ -d "${target}" && ! -L "${target}" ]]; then
        du -sh -- "${target}"
        rm -rf -- "${target}"
    fi
    test ! -e "${target}"
}

echo "GALORE_CLEANUP_BEFORE"
df -h /workspace-SR006.nfs3
remove_checkpoint_tree "${group}/llama257M_galore_rho0p7_bf16_lr1e-3_wd0p1_1xC_1gpu/ckpts"
remove_checkpoint_tree "${group}/llama257M_galore_rho0p8_bf16_lr1e-3_wd0p1_1xC_1gpu/ckpts"
echo "GALORE_CLEANUP=ok"
df -h /workspace-SR006.nfs3
du -sh -- "${root}"
