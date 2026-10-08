#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WANDB_BASE_URL=https://wandb-radfan.ru
ROOT=/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006
NAME=llama500M_frugal_bf16_wd1e-3_1xC_warmup2000_2gpu_20261006
python scripts/cloud/verify_500m_bf16_wd_tuning.py \
    --metrics "${ROOT}/500M_BF16_WD_tuning_1xC_20261006/${NAME}/metrics.jsonl" \
    --run-id wd500-frugal-1e-3-20261006 --target 75457 --opt coord_adamw --wd 1e-3 --lr 1e-3
touch "${ROOT}/markers/${NAME}.done"
echo "WD_COMPLETION_RECOVERED=${NAME}"
