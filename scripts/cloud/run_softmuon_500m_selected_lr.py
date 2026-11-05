"""Authorized 500M 1xC follow-up after all eight 257M finals are verified."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MODE'] = 'matched'
os.environ['CLOUD_TRAIN_WORLD_SIZE'] = '4'
os.environ.setdefault('CAMPAIGN_DATE', '20261105')

import run_adaptive_muon as campaign
from run_matched_normuon import preflight
from softmuon_selection import select


def main() -> None:
    preflight()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    if os.environ.get('WANDB_MODE') != 'online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('500M follow-up requires online W&B logging')
    selection = select()
    lr = selection['best_matrix_lr']
    if os.environ['LR'] != lr:
        raise RuntimeError('Submitted LR differs from the winner of all eight finals')
    if campaign.RANK == 0:
        campaign.atomic_json(campaign.ROOT / 'selected_257m_lr.json', selection)
        print('SOFTMUON_SWEEP_SELECTION=' + json.dumps(selection), flush=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], check=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_softmuon_algorithm3.py'], check=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_softmuon.py'], check=True)
    for phase in ('matched_smoke', '1xC'):
        campaign.train('softmuon', lr, 500, phase, 75457, weight_decay='1e-4')


if __name__ == '__main__':
    main()
