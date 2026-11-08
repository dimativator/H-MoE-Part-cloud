"""Four authorized ANVIL II 500M/1xC weight-decay trials, two GPUs each."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MODE'] = 'matched'
os.environ['CAMPAIGN_DATE'] = '20261108'

import run_adaptive_muon as campaign
from run_matched_normuon import preflight


WDS = ('1e-1', '1e-2', '1e-3', '1e-4')


def main() -> None:
    wd = os.environ['WD']
    if wd not in WDS or campaign.TRAIN_WORLD != 2:
        raise ValueError('Unauthorized weight decay or GPU allocation')
    if os.environ.get('WANDB_MODE') != 'online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B logging is required')
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
    print('ANVIL2_CAMPAIGN=' + json.dumps(dict(lr='5e-3', size=500, target=75457,
          weight_decay=float(wd), world_size=2, checkpoints=False,
          auxiliary_lr=0.005, engage_step=37729,
          revision=os.environ['CODE_REVISION'])), flush=True)
    campaign.train('anvil2', '5e-3', 500, 'matched_smoke', 75457, weight_decay=wd)
    campaign.train('anvil2', '5e-3', 500, '1xC', 75457, weight_decay=wd)


if __name__ == '__main__':
    main()
