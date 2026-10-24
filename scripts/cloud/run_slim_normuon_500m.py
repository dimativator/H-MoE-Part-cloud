"""500M BF16 2xC, followed by independent 1xC decay from step67911."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ['MODE'] = '500m_slim_normuon'
os.environ.setdefault('CAMPAIGN_DATE', '20261024')

import run_adaptive_muon as campaign


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if revision != os.environ['CODE_REVISION']:
        raise RuntimeError('Source revision does not match submission')
    if not str(campaign.ROOT).startswith('/workspace-SR006.nfs3/'):
        raise RuntimeError('All campaign outputs must use NFS3')
    platform_logs = Path('/home/jovyan/shares/SR006.nfs3/mlsub-logs')
    if os.environ.get('VERIFIED_BOOTSTRAP_LOGS_DIR') != str(platform_logs):
        raise RuntimeError('Gateway bootstrap logs_dir must be verified on NFS3 before submission')
    for path in (Path('/workspace-SR006.nfs3'), Path('/home/jovyan'), Path('/tmp'), platform_logs):
        free = shutil.disk_usage(path).free
        print('PREFLIGHT_DISK=' + json.dumps(dict(path=str(path), available_bytes=free)), flush=True)
        if free <= 5_000_000_000:
            raise RuntimeError(f'Insufficient storage reserve: {path}, {free} bytes')
    campaign.ROOT.mkdir(parents=True, exist_ok=True)
    cache = campaign.ROOT / 'cache'
    temporary = campaign.ROOT / 'tmp'
    for path in (cache, temporary):
        path.mkdir(exist_ok=True)
    os.environ.update(CLOUD_DISK_GUARD_ROOT=str(campaign.ROOT),
                      CLOUD_DISK_GUARD_PATHS=json.dumps([str(campaign.ROOT), '/home/jovyan', '/tmp']),
                      TMPDIR=str(temporary), XDG_CACHE_HOME=str(cache),
                      HF_HOME=str(cache / 'huggingface'), CUDA_CACHE_PATH=str(cache / 'cuda'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'))
    mapping = repo / 'src/optim/memory_efficient/slim_adam/layer_map_llama_paper.json'
    print('SLIM_NORMUON_500M_CONFIG=' + json.dumps(dict(
        code_revision=revision, lr=0.001, weight_decay=0.0001, ns_steps=5,
        matrix_momentum=0.95, matrix_beta2=0.95, auxiliary_adamw_betas=[0.9, 0.95],
        auxiliary_lr='shared', auxiliary_weight_decay=0.0,
        compression_map_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
        trunk_steps=150914, bridge_checkpoint_step=67911, decay_target_step=75457,
        bootstrap_logs_dir=str(platform_logs), output_root=str(campaign.ROOT),
        disk_guard_every_steps=10, disk_reserve_bytes=5_000_000_000)), flush=True)
    campaign.main()


if __name__ == '__main__':
    main()
