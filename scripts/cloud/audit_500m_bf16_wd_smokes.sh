#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WANDB_BASE_URL=https://wandb-radfan.ru
for optimizer in frugal galore frugal_mm apollo; do
    case "${optimizer}" in
        frugal) root=/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006; opt=coord_adamw; lr=1e-3 ;;
        galore) root=/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006; opt=galore_adamw; lr=1e-3 ;;
        frugal_mm) root=/workspace-SR006.nfs2/dimativator/500m-bf16-wd-tuning-20261006; opt=coord_muon; lr=2e-3 ;;
        apollo) root=/workspace-SR006.nfs2/dimativator/500m-bf16-wd-tuning-20261006; opt=apollo_adamw; lr=2e-3 ;;
    esac
    name=smoke_llama500M_${optimizer}_bf16_wd1e-2_1xC_warmup2000_2gpu_20261006
    python scripts/cloud/verify_500m_bf16_wd_tuning.py \
        --metrics "${root}/500M_BF16_WD_tuning_1xC_20261006/${name}/metrics.jsonl" \
        --run-id "smoke-wd500-${optimizer}-1e-2-20261006" --target 2 --opt "${opt}" --wd 1e-2 --lr "${lr}"
done
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
echo WD_ALL_SMOKES_COMPLETION_AUDITED
