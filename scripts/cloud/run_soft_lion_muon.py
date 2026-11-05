"""Authorized 257M/500M SoftLionMuon 1xC trial, with no checkpoint writes."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MODE'] = 'matched'
os.environ['CAMPAIGN_DATE'] = '20261105'

import run_adaptive_muon as campaign
from run_matched_normuon import preflight


def main() -> None:
    preflight()
    if os.environ.get('WANDB_MODE') != 'online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B logging is required')
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('All checkpoints must be disabled')
    size = int(os.environ['MODEL_SIZE'])
    lr = os.environ['LR']
    expected_world = 2 if size == 257 else 4
    if size not in (257, 500) or lr not in campaign.LRS or campaign.TRAIN_WORLD != expected_world:
        raise ValueError('Unauthorized LR, model size, or GPU allocation')
    target, wd = (39250, '1e-1') if size == 257 else (75457, '1e-4')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], check=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_soft_lion_muon_gpu.py'], check=True)
    print('SOFT_LION_MUON_CAMPAIGN=' + json.dumps(dict(size=size, target=target, lr=lr,
          weight_decay=wd, world_size=expected_world, checkpoints=False,
          revision=os.environ['CODE_REVISION'])), flush=True)
    campaign.train('soft_lion_muon', lr, size, 'matched_smoke', target, weight_decay=wd)
    campaign.train('soft_lion_muon', lr, size, '1xC', target, weight_decay=wd)


if __name__ == '__main__':
    main()
