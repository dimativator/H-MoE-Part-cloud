#!/usr/bin/env bash
set -euo pipefail

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs3

for path in \
    /home/jovyan/data \
    /home/jovyan/hmoe-checkpoints \
    /home/jovyan/hmoe-cloud \
    /home/jovyan/finewebedu_h200 \
    /home/jovyan/dimativator; do
    echo "PATH=${path}"
    if [[ ! -d "${path}" || -L "${path}" ]]; then
        echo "MISSING_OR_SYMLINK"
        continue
    fi
    stat -c 'OWNER=%U GROUP=%G MODIFIED=%y' -- "${path}"
    echo "DIRECT_CHILDREN"
    du -x -h --max-depth=1 -- "${path}" 2>/dev/null | sort -h | tail -n 35 || true
    echo "LARGE_FILES_BYTES_DATE_PATH"
    find "${path}" -xdev -type f -size +300M \
        -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null \
        | sort -nr | head -n 35 || true
done

echo "OTHER_LARGE_DIRECTORY_METADATA"
for path in /home/jovyan/monarch-pretrain /home/jovyan/xandi281; do
    if [[ -d "${path}" && ! -L "${path}" ]]; then
        stat -c '%n OWNER=%U GROUP=%G MODIFIED=%y' -- "${path}"
    fi
done
