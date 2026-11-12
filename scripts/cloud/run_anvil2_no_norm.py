"""ANVIL II normalization ablation: 257M LR sweep, then selected-LR 500M."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MODE'] = 'matched'
os.environ['CAMPAIGN_DATE'] = '20261112'
os.environ['ANVIL2_NO_NORMALIZATION'] = '1'

import run_adaptive_muon as campaign
from run_matched_normuon import preflight


LRS = ('1e-2', '5e-3', '2e-3', '1e-3', '5e-4', '1e-4')


def main() -> None:
    stage, lr = os.environ['ANVIL2_STAGE'], os.environ['LR']
    if lr not in LRS or stage not in ('257m', '500m'):
        raise ValueError('Unauthorized stage or LR')
    size, target, world, wd = (257, 39250, 2, '1e-1') if stage == '257m' else (500, 75457, 4, '1e-4')
    if campaign.TRAIN_WORLD != world:
        raise ValueError('Wrong training world size for this stage')
    if os.environ.get('WANDB_MODE') != 'online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B is required')
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('All checkpoints must be disabled')
    preflight()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            stat = os.statvfs(path)
            record = dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize, used_for_writes=False)
        except OSError as exc:
            record = dict(path=path, telemetry_error=str(exc), used_for_writes=False)
        print('NFS2_READ_ONLY=' + json.dumps(record), flush=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], check=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_anvil2_gpu.py'], check=True)
    print('ANVIL2_NO_NORM_CAMPAIGN=' + json.dumps(dict(size=size, target=target, lr=lr,
          weight_decay=wd, world_size=world, normalization=False, checkpoints=False,
          revision=os.environ['CODE_REVISION'])), flush=True)
    campaign.train('anvil2', lr, size, 'matched_smoke', target, weight_decay=wd)
    campaign.train('anvil2', lr, size, '1xC', target, weight_decay=wd)


if __name__ == '__main__':
    main()
