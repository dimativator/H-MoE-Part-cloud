"""One authorized single-GPU cosine-to-zero 257M trial, without checkpoints."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MODE'] = 'matched'
os.environ['CAMPAIGN_DATE'] = '20261112'
os.environ['CLOUD_TRAIN_WORLD_SIZE'] = '1'

import run_adaptive_muon as campaign
from run_matched_normuon import preflight

OPTIMIZERS = ('soft_lion_muon', 'lion_muon', 'muon_reference', 'softmuon')


def main() -> None:
    optimizer, lr = os.environ['OPTIMIZER'], os.environ['LR']
    if optimizer not in OPTIMIZERS or lr not in campaign.LRS:
        raise ValueError('Optimizer or LR outside the authorized grid')
    if campaign.WORLD != 1:
        raise RuntimeError('Exactly one physical GPU/MPI worker per trial is required')
    if os.environ.get('WANDB_MODE') != 'online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B logging is required')
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('Checkpoint saving must be disabled')
    preflight()
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'src'),
                   str(Path(__file__).resolve().parents[2])]
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free',
                    '--format=csv'], check=True)
    if optimizer == 'soft_lion_muon':
        subprocess.run([sys.executable, 'scripts/cloud/test_soft_lion_muon_gpu.py'], check=True)
    print('COSINE_CAMPAIGN=' + json.dumps(dict(optimizer=optimizer, lr=lr,
          scheduler='cos_warmup_zero', warmup_steps=2000, target=39250,
          tokens=5144576000, weight_decay=0.1, auxiliary_lr=0.001,
          world_size=1, checkpoints=False, revision=os.environ['CODE_REVISION'])), flush=True)
    # Exercise a complete Lion/Muon period before the fresh full-budget run.
    campaign.train(optimizer, lr, 257, 'matched_smoke', 39250,
                   scheduler='cos_warmup_zero', smoke_steps_override=6)
    campaign.train(optimizer, lr, 257, 'cosine_1xC', 39250,
                   scheduler='cos_warmup_zero')


if __name__ == '__main__':
    main()
