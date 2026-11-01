"""Matched Muon/NorMuon LR pair, or the conditional 500M NorMuon follow-up."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess

os.environ['MODE'] = 'matched'
os.environ.setdefault('CAMPAIGN_DATE', '20261101')

import run_adaptive_muon as campaign


def validate_final(record: dict, optimizer: str, lr: str, size: int) -> None:
    target = 39250 if size == 257 else 75457
    expected = dict(optimizer=optimizer, lr=lr, size=size, phase='1xC', iter=target,
                    consumed_tokens=target * 128 * 1024, weight_decay=0.1 if size == 257 else 0.0001,
                    ns_steps=5, checkpoint_saving=False, update_rms=0.2,
                    row_normalization=optimizer == 'normuon_matched', auxiliary_lr=0.001)
    if any(record.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Final configuration mismatch: expected {expected}')
    if not math.isfinite(record['final-val/loss']):
        raise RuntimeError('Nonfinite final loss')


def select_winner(source: Path) -> dict:
    records = []
    for lr in campaign.LRS:
        for optimizer in ('muon_matched', 'normuon_matched'):
            path = source / f'lr{lr}' / 'finals' / f'{optimizer}_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
            record = json.loads(path.read_text())
            validate_final(record, optimizer, lr, 257)
            if optimizer == 'normuon_matched':
                records.append(record)
    winner = min(records, key=lambda r: r['final-val/loss'])
    if winner['final-val/loss'] >= 3.0056934356689453:
        raise RuntimeError('No improvement over original NorMuon; 500M not authorized by gate')
    return winner


def preflight() -> None:
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if revision != os.environ['CODE_REVISION']:
        raise RuntimeError('Unexpected source revision')
    logs = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
    if os.environ.get('VERIFIED_BOOTSTRAP_LOGS_DIR') != logs:
        raise RuntimeError('Bootstrap log directory must be verified on NFS3')
    if not str(campaign.ROOT).startswith('/workspace-SR006.nfs3/dimativator/'):
        raise RuntimeError('All outputs must use owned NFS3 root')
    for path in ('/workspace-SR006.nfs3', logs, '/home/jovyan', '/tmp', '/'):
        free = shutil.disk_usage(path).free
        print('PREFLIGHT_DISK=' + json.dumps(dict(path=path, available_bytes=free)), flush=True)
        if free <= 5_000_000_000:
            raise RuntimeError(f'Storage reserve unavailable: {path}, {free}')
    campaign.ROOT.mkdir(parents=True, exist_ok=True)
    for child in ('cache', 'tmp', 'wandb'):
        (campaign.ROOT / child).mkdir(exist_ok=True)
    cache = campaign.ROOT / 'cache'
    os.environ.update(CLOUD_DISK_GUARD_ROOT=str(campaign.ROOT),
                      CLOUD_DISK_GUARD_PATHS=json.dumps([str(campaign.ROOT), '/home/jovyan', '/tmp', '/', logs]),
                      TMPDIR=str(campaign.ROOT / 'tmp'), XDG_CACHE_HOME=str(cache),
                      HF_HOME=str(cache / 'huggingface'), CUDA_CACHE_PATH=str(cache / 'cuda'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                      WANDB_INIT_TIMEOUT='45', WANDB_HTTP_TIMEOUT='20')
    metadata = json.loads((campaign.DATA / 'packed_metadata.json').read_text())
    if metadata['format'] != 'packed_fineweb_h200_v3' or metadata['iterations'] < 75457:
        raise RuntimeError('Unexpected packed data')
    if metadata['world_size'] != 2 or metadata['batch_size'] != 16:
        raise RuntimeError('Unexpected packed replay configuration')
    os.chdir(repo)


def main() -> None:
    preflight()
    stage = os.environ['MATCHED_STAGE']
    if stage == '257m':
        lr = os.environ['LR']
        if lr not in campaign.LRS:
            raise ValueError('LR outside prepared grid')
        for optimizer in ('muon_matched', 'normuon_matched'):
            campaign.train(optimizer, lr, 257, 'matched_smoke', 39250, weight_decay='1e-1')
        for optimizer in ('muon_matched', 'normuon_matched'):
            campaign.train(optimizer, lr, 257, '1xC', 39250, weight_decay='1e-1')
    elif stage == '500m':
        winner = select_winner(Path(os.environ['MATCHED_SWEEP_ROOT']))
        print('MATCHED_WINNER=' + json.dumps(winner), flush=True)
        campaign.train('normuon_matched', winner['lr'], 500, 'matched_smoke', 75457, weight_decay='1e-4')
        campaign.train('normuon_matched', winner['lr'], 500, '1xC', 75457, weight_decay='1e-4')
    else:
        raise ValueError('Unknown stage')


if __name__ == '__main__':
    main()
