#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-audit}
case "${MODE}" in
    audit|sync|cleanup) ;;
    *) echo "MODE must be audit, sync, or cleanup" >&2; exit 2 ;;
esac

readonly home_root=/home/jovyan/dimativator/500m-slimadam-bf16-wd-2xc-warmup2000-20260926
readonly nfs3_root=/workspace-SR006.nfs3/dimativator/500m-slimadam-bf16-wd-2xc-warmup2000-20260926
readonly trunk_group=2xChinchilla_500M_slimadam_bf16_wd_sweep_2gpu_cloud
readonly decay_group=1xChinchilla_decay_500M_slimadam_bf16_wd_sweep_2gpu_cloud

shopt -s nullglob
offline_runs=()
checkpoint_dirs=()
for wd in 1e-2 1e-3 1e-4; do
    root=${home_root}
    [[ ${wd} == 1e-4 ]] && root=${nfs3_root}
    trunk=llama500M_slim_adam_bf16_wd${wd}_2xC_warmup2000_2gpu
    decay=llama500M_slim_adam_bf16_wd${wd}_1xC_decay_warmup2000_2gpu
    log=${root}/logs/${trunk}_full_rank0.log
    trunk_metrics=${root}/${trunk_group}/${trunk}/metrics.jsonl
    decay_metrics=${root}/${decay_group}/${decay}/metrics.jsonl
    ckpt_root=${root}/${trunk_group}/${trunk}/ckpts
    ckpt=${ckpt_root}/67911

    grep -Fq "SLIMADAM_2XC_COMPLETE wd=${wd} iter=150914" "${log}"
    grep -Fq "SLIMADAM_1XC_DECAY_COMPLETE wd=${wd} iter=75457" "${log}"
    grep -F '"final-val/loss"' "${trunk_metrics}" | grep -Fq '"iter": 150914'
    grep -F '"final-val/loss"' "${decay_metrics}" | grep -Fq '"iter": 75457'

    [[ ! -L ${ckpt_root} && ! -L ${ckpt} && -d ${ckpt} ]]
    [[ $(find "${ckpt_root}" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ]]
    for file in main.pt worker_0.pt worker_1.pt; do
        [[ -f ${ckpt}/${file} && ! -L ${ckpt}/${file} && -s ${ckpt}/${file} ]]
    done
    checkpoint_dirs+=("${ckpt}")
    echo "CHECKPOINT wd=${wd} path=${ckpt} bytes=$(du -sb "${ckpt}" | cut -f1)"

    mapfile -d '' -t matches < <(find "${root}/wandb" -maxdepth 3 -type d -name 'offline-run-*' -print0)
    mapfile -t log_run_ids < <(
        grep -oE 'offline-run-[0-9_]+-[a-z0-9]+' "${log}" |
        sed 's/.*-//' | awk '!seen[$0]++'
    )
    echo "LOG_OFFLINE_IDS wd=${wd} ids=${log_run_ids[*]:-none}"
    [[ ${#log_run_ids[@]} -eq 2 ]] || { echo "Expected two offline run IDs in ${log}" >&2; exit 3; }
    for run_id in "${log_run_ids[@]}"; do
        matched=0
        for run_dir in "${matches[@]}"; do
            [[ ${run_dir##*-} == "${run_id}" ]] || continue
            [[ -s ${run_dir}/run-${run_id}.wandb ]]
            offline_runs+=("${run_dir}")
            matched=$((matched + 1))
            echo "OFFLINE_RUN wd=${wd} id=${run_id} synced=$([[ -e ${run_dir}/.synced ]] && echo true || echo false) path=${run_dir}"
            find "${run_dir}" -maxdepth 1 -type f -printf 'WANDB_ROOT_FILE=%f\n'
        done
        [[ ${matched} -eq 1 ]] || { echo "Expected one offline directory for ${run_id}; found ${matched}" >&2; exit 3; }
    done
done
[[ ${#offline_runs[@]} -eq 6 && ${#checkpoint_dirs[@]} -eq 3 ]]
echo "AUDIT_OK offline_runs=6 checkpoints=3"

if [[ ${MODE} == sync ]]; then
    [[ -n ${WANDB_API_KEY:-} || -r ${HOME}/.netrc ]]
    command -v wandb >/dev/null
    export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey
    for run_dir in "${offline_runs[@]}"; do
        run_id=${run_dir##*-}
        if [[ -e ${run_dir}/.synced ]]; then
            echo "SYNC_ALREADY_DONE id=${run_id}"
        else
            wandb sync --entity andrey --project fp8-pretrain --mark-synced "${run_dir}"
            [[ -e ${run_dir}/.synced ]]
            echo "SYNC_DONE id=${run_id}"
        fi
        echo "SYNC_LINK=https://wandb-radfan.ru/andrey/fp8-pretrain/runs/${run_id}"
    done
    echo 'SLIMADAM_WD_WANDB_SYNC_COMPLETE count=6'
elif [[ ${MODE} == cleanup ]]; then
    [[ ${CONFIRM_WANDB_VERIFIED:-} == yes ]] || { echo 'Remote W&B verification required' >&2; exit 4; }
    for run_dir in "${offline_runs[@]}"; do
        [[ -e ${run_dir}/.synced ]]
    done
    for ckpt in "${checkpoint_dirs[@]}"; do
        echo "DELETE_CHECKPOINT=${ckpt}"
        rm -r -- "${ckpt}"
        [[ ! -e ${ckpt} ]]
    done
    echo 'SLIMADAM_WD_CHECKPOINT_CLEANUP_COMPLETE count=3'
fi
