"""Two-GPU SoftMuon Algorithm 3 trial with fixed auxiliary LR and no checkpoints."""

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
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'src'))
    lr = os.environ['LR']
    allowed_lrs = (*campaign.LRS, '3e-3', '4e-3', '5e-3', '6e-3')
    if lr not in allowed_lrs or campaign.TRAIN_WORLD != 2:
        raise ValueError('Expected an authorized LR and exactly two GPUs')
    print('SOFTMUON_CAMPAIGN=' + json.dumps(dict(
        code_revision=os.environ['CODE_REVISION'], matrix_lr=float(lr), auxiliary_lr=0.001,
        weight_decay=0.1, auxiliary_weight_decay=0.1, target=39250,
        consumed_tokens=5144576000, checkpoint_saving=False,
        sign_iters=35325, transition_iters=3925, model_parameter_dtype='float32',
        compute_dtype='bfloat16', states='float32', update_scale='original_shape_factor',
        seed=0, data_seed=1337, bootstrap_logs=os.environ['VERIFIED_BOOTSTRAP_LOGS_DIR'])), flush=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free',
                    '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv'], check=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_softmuon_algorithm3.py'], check=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_softmuon.py'], check=True)
    campaign.train('softmuon', lr, 257, 'matched_smoke', 39250, weight_decay='1e-1')
    campaign.train('softmuon', lr, 257, '1xC', 39250, weight_decay='1e-1')


if __name__ == '__main__':
    main()
