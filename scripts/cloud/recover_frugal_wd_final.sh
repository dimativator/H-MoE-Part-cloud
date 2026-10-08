#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WANDB_BASE_URL=https://wandb-radfan.ru
ROOT=/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006
NAME=llama500M_frugal_bf16_wd1e-3_1xC_warmup2000_2gpu_20261006
if [[ "${SYNC_LOCAL_BEFORE_VERIFY:-0}" == 1 ]]; then
    python - <<'PY'
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, 'scripts/cloud')
from verify_500m_bf16_wd_tuning import final_loss
root = Path('/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006')
name = 'llama500M_frugal_bf16_wd1e-3_1xC_warmup2000_2gpu_20261006'
final_loss(root / '500M_BF16_WD_tuning_1xC_20261006' / name / 'metrics.jsonl', 75457)
run_id = 'wd500-frugal-1e-3-20261006'
files = list((root / 'wandb').rglob(f'run-{run_id}.wandb'))
assert len(files) == 1, f'Expected exactly one original W&B file, got {len(files)}'
print(f'RECOVER_ORIGINAL_WANDB={files[0]}', flush=True)
subprocess.run(['wandb', 'sync', '--include-online', '--entity', 'andrey',
                '--project', 'fp8-pretrain', str(files[0].parent)], check=True, timeout=900)
PY
fi
python scripts/cloud/verify_500m_bf16_wd_tuning.py \
    --metrics "${ROOT}/500M_BF16_WD_tuning_1xC_20261006/${NAME}/metrics.jsonl" \
    --run-id wd500-frugal-1e-3-20261006 --target 75457 --opt coord_adamw --wd 1e-3 --lr 1e-3
touch "${ROOT}/markers/${NAME}.done"
echo "WD_COMPLETION_RECOVERED=${NAME}"
