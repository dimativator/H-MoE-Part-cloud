"""One authorized ANVIL II 257M/1xC LR trial on two Cloud.ru GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MODE'] = 'matched'
os.environ['CAMPAIGN_DATE'] = '20261106'

import run_adaptive_muon as campaign
from run_matched_normuon import preflight


LRS = ('1e-2', '5e-3', '2e-3', '1e-3', '5e-4', '1e-4')


def main() -> None:
    preflight()
    if os.environ.get('WANDB_MODE') != 'online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B logging is required')
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('All checkpoints must be disabled')
    lr = os.environ['LR']
    if lr not in LRS or campaign.TRAIN_WORLD != 2:
        raise ValueError('Unauthorized LR or GPU allocation')
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
    print('ANVIL2_CAMPAIGN=' + json.dumps(dict(lr=lr, size=257, target=39250,
          weight_decay=0.1, world_size=2, checkpoints=False,
          auxiliary_lr=float(lr), engage_step=19625,
          revision=os.environ['CODE_REVISION'])), flush=True)
    campaign.train('anvil2', lr, 257, 'matched_smoke', 39250, weight_decay='1e-1')
    campaign.train('anvil2', lr, 257, '1xC', 39250, weight_decay='1e-1')


if __name__ == '__main__':
    main()
