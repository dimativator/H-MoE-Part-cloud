#!/usr/bin/env bash
set -euo pipefail

readonly run_root=/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/frugal
readonly checkpoint=${run_root}/500M_time_matched_muon_fp8_1xC_20260925/llama500M_frugal_bf16_time_matched_4gpu/ckpts/latest
readonly log=${run_root}/logs/llama500M_frugal_bf16_time_matched_4gpu_rank0.log
readonly mode=${MODE:-inspect}
case "${mode}" in inspect|delete) ;; *) exit 2 ;; esac

grep -q 'TIME_MATCHED_COMPLETE optimizer=frugal iter=91258' "${log}"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || exit 3
[[ "$(realpath -e -- "${checkpoint}")" == "${checkpoint}" ]] || exit 3
[[ -s "${checkpoint}/main.pt" && ! -L "${checkpoint}/main.pt" ]] || exit 3
for rank in 0 1 2 3; do
    [[ -s "${checkpoint}/worker_${rank}.pt" && ! -L "${checkpoint}/worker_${rank}.pt" ]] || exit 3
done
[[ "$(find "${checkpoint}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 5 ]] || exit 3
[[ "$(find "${checkpoint}" -mindepth 1 -maxdepth 1 | wc -l)" -eq 5 ]] || exit 3

echo "FRUGAL_CHECKPOINT=${checkpoint}"
du -sh -- "${checkpoint}"
find "${checkpoint}" -mindepth 1 -maxdepth 1 -printf 'FRUGAL_CHECKPOINT_FILE=%f\n' | sort
df -h /workspace-SR006.nfs2
if [[ "${mode}" == inspect ]]; then
    echo FRUGAL_CHECKPOINT_INSPECT_COMPLETE
    exit 0
fi

rm -r -- "${checkpoint}"
[[ ! -e "${checkpoint}" ]] || exit 4
echo FRUGAL_CHECKPOINT_DELETE_COMPLETE
df -h /workspace-SR006.nfs2
