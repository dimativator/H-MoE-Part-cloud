#!/usr/bin/env bash
set -euo pipefail

readonly run_root=/home/jovyan/dimativator/500m-time-matched-20260925/slim_adam
readonly checkpoint=${run_root}/500M_time_matched_muon_fp8_1xC_20260925/llama500M_slim_adam_bf16_time_matched_4gpu/ckpts/latest
readonly log=${run_root}/logs/llama500M_slim_adam_bf16_time_matched_4gpu_rank0.log
readonly mode=${MODE:-inspect}
case "${mode}" in inspect|delete) ;; *) exit 2 ;; esac

grep -q 'TIME_MATCHED_COMPLETE optimizer=slim_adam iter=90333' "${log}"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || exit 3
[[ "$(realpath -e -- "${checkpoint}")" == "${checkpoint}" ]] || exit 3
[[ -s "${checkpoint}/main.pt" ]] || exit 3
for rank in 0 1 2 3; do
    [[ -s "${checkpoint}/worker_${rank}.pt" ]] || exit 3
done

echo "SLIMADAM_CHECKPOINT=${checkpoint}"
du -sh -- "${checkpoint}"
find "${checkpoint}" -mindepth 1 -maxdepth 1 -printf 'SLIMADAM_CHECKPOINT_FILE=%f\n' | sort
df -h /home/jovyan
if [[ "${mode}" == inspect ]]; then
    echo SLIMADAM_CHECKPOINT_INSPECT_COMPLETE
    exit 0
fi

rm -r -- "${checkpoint}"
[[ ! -e "${checkpoint}" ]] || exit 4
echo SLIMADAM_CHECKPOINT_DELETE_COMPLETE
df -h /home/jovyan
