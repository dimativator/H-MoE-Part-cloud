#!/usr/bin/env bash
set -euo pipefail

readonly root=/home/jovyan
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h "${root}"

echo HOME_TOP_LEVEL_USAGE
du -x -h --max-depth=1 -- "${root}" 2>/dev/null | sort -h || true

echo HOME_TOP_LEVEL_OWNERS
find "${root}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null |
    while IFS= read -r -d '' path; do
        stat -c '%n OWNER=%U UID=%u GROUP=%G MODIFIED=%y' -- "${path}"
    done

echo RL_MUON_MATCHES
for subtree in "${root}/xandi281" "${root}/monarch-pretrain"; do
    if [[ ! -d "${subtree}" || -L "${subtree}" ]]; then
        continue
    fi
    echo "SUBTREE=${subtree}"
    du -x -h --max-depth=2 -- "${subtree}" 2>/dev/null | sort -h | tail -n 35 || true
    find "${subtree}" -maxdepth 6 \
        \( -type d -o -type l \) \
        \( -iname '*rl*muon*' -o -iname '*muon*rl*' \) -print0 2>/dev/null |
        while IFS= read -r -d '' path; do
            stat -c '%n OWNER=%U UID=%u GROUP=%G MODIFIED=%y' -- "${path}"
            du -x -sh -- "${path}" 2>/dev/null || true
        done || true
done
