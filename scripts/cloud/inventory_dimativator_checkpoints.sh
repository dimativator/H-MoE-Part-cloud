#!/usr/bin/env bash
set -euo pipefail

readonly roots=(
    /workspace-SR006.nfs2/dimativator
    /workspace-SR006.nfs3/dimativator
)

echo "CHECKPOINT_INVENTORY_START $(date --iso-8601=seconds)"
for root in "${roots[@]}"; do
    echo "ROOT=${root}"
    if [[ ! -d "${root}" || -L "${root}" ]]; then
        echo "ROOT_MISSING_OR_UNSAFE=${root}"
        continue
    fi
    df -h "${root}"
    echo "CHECKPOINT_FILES_BEGIN"
    find "${root}" -type f \( \
        -name main.pt -o \
        -name 'worker_*.pt' -o \
        -name '*.ckpt' -o \
        -name '*.pth' -o \
        -iname '*checkpoint*.pt' -o \
        -iname '*checkpoint*.tar' -o \
        -iname '*checkpoint*.tar.gz' -o \
        -iname '*checkpoint*.tgz' \
    \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null \
        | sort -k3,3
    echo "CHECKPOINT_FILES_END"
    echo "CHECKPOINT_DIR_USAGE_BEGIN"
    while IFS= read -r directory; do
        [[ -n "${directory}" ]] || continue
        bytes=$(du -sb -- "${directory}" 2>/dev/null | awk '{print $1}')
        printf '%s\t%s\n' "${bytes:-unknown}" "${directory}"
    done < <(
        find "${root}" -type f \( -name main.pt -o -name 'worker_*.pt' -o -name '*.ckpt' \) \
            -printf '%h\n' 2>/dev/null | sort -u
    )
    echo "CHECKPOINT_DIR_USAGE_END"
done
echo "CHECKPOINT_INVENTORY_COMPLETE"
