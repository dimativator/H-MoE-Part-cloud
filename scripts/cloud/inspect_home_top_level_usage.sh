#!/usr/bin/env bash
set -euo pipefail

root=/home/jovyan
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h "${root}" /workspace-SR006.nfs3

echo HOME_TOP_LEVEL_USAGE
du -x -h --max-depth=1 "${root}" 2>/dev/null | sort -h || true

echo RECENT_LARGE_FILES
find "${root}" -xdev -type f -mmin -120 -size +100M \
    -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null \
    | sort -nr | head -n 50 || true
