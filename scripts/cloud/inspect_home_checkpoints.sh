#!/usr/bin/env bash
set -euo pipefail

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan

for root in \
    /home/jovyan/hmoe-checkpoints \
    /home/jovyan/dimativator \
    /home/jovyan/rl_muon; do
    if [[ ! -d "${root}" || -L "${root}" ]]; then
        echo "SKIP=${root}"
        continue
    fi
    echo "=== USAGE ${root} ==="
    du -x -h --max-depth=2 "${root}" 2>/dev/null | sort -h | tail -n 120
done

echo "=== CHECKPOINT_FILES ==="
find \
    /home/jovyan/hmoe-checkpoints \
    /home/jovyan/dimativator \
    /home/jovyan/rl_muon \
    -xdev -type f \
    \( -name 'main.pt' -o -name 'worker_*.pt' -o -name '*optim*.pt' \
       -o -name 'model_world_size_*_rank_*.pt' -o -name '*.ckpt' \) \
    -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null \
    | sort -nr | head -n 300
