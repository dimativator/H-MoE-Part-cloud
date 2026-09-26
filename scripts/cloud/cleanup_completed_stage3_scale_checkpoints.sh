#!/usr/bin/env bash
set -euo pipefail

readonly checkpoint_root=/home/jovyan/dimativator/stage3_257m_scale_20260925/4xChinchilla_257M_bf16_scale/257m_scale_bf16_4xC_cloud_4gpu/ckpts
readonly log_root=/home/jovyan/dimativator/stage3_257m_scale_20260925/logs
readonly mode=${MODE:-inspect}
case "${mode}" in inspect|delete) ;; *) exit 2 ;; esac

grep -q 'SCALE_4XC_COMPLETE lr=1e-3 iter=157000 mode=full' \
    "${log_root}/257m_scale_bf16_4xC_cloud_4gpu_full_rank0.log"
for scale in 1 2; do
    target=$((39250 * scale))
    grep -q "SCALE_DECAY_COMPLETE scale=${scale} lr=1e-3 iter=${target}" \
        "${log_root}/257m_scale_bf16_${scale}xC_decay_cloud_4gpu_rank0.log"
done

[[ -d "${checkpoint_root}" && ! -L "${checkpoint_root}" ]] || exit 3
[[ "$(realpath -e -- "${checkpoint_root}")" == "${checkpoint_root}" ]] || exit 3
for checkpoint in 35325 70650 141300 latest; do
    [[ -d "${checkpoint_root}/${checkpoint}" && ! -L "${checkpoint_root}/${checkpoint}" ]] || exit 3
done

echo "SCALE_CHECKPOINT_ROOT=${checkpoint_root}"
du -sh -- "${checkpoint_root}"
find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -printf 'SCALE_CHECKPOINT_CHILD=%f\n' | sort
df -h /home/jovyan
if [[ "${mode}" == inspect ]]; then
    echo SCALE_CHECKPOINT_INSPECT_COMPLETE
    exit 0
fi

rm -r -- "${checkpoint_root}"
[[ ! -e "${checkpoint_root}" ]] || exit 4
echo SCALE_CHECKPOINT_DELETE_COMPLETE
df -h /home/jovyan
