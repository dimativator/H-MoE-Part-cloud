"""One approved LR per one-GPU Cloud.ru job; four jobs form the sweep."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


LRS = ('0.01', '0.02', '0.03', '0.045')
ITERATIONS = 39250
TOKENS_PER_STEP = 128 * 1024
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')


def build_command(lr: str, root: Path, date: str) -> tuple[list[str], Path, str]:
    if lr not in LRS:
        raise ValueError(f'LR must be one of {LRS}')
    group = f'mnd_muon_257M_1xC_{date}'
    name = f'mnd_muon_257m_bf16_lr{lr}_wd1e-1_1xC_1gpu'
    metrics = root / group / name / 'metrics.jsonl'
    command = [sys.executable, 'src/main.py', '--experiment-name', name,
               '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb',
               '--datasets-dir', str(DATA), '--streaming', '--workers', '8',
               '--fineweb-replay-world-size', '2', '--fineweb-replay-layout', 'serial',
               '--model', 'llama', '--n-layer', '12', '--n-embd', '1024', '--n-head', '8',
               '--multiple-of', '256', '--sequence-length', '1024', '--dtype', 'bfloat16',
               '--dropout', '0', '--opt', 'mnd_muon', '--lr', lr, '--weight-decay', '0.1',
               '--mnd-muon-beta', '0.9', '--mnd-muon-eps', '1e-9', '--mnd-muon-polar-steps', '5',
               '--beta1', '0.95', '--beta2', '0.95', '--eps', '1e-8', '--grad-clip', '1',
               '--batch-size', '16', '--acc-steps', '8', '--eval-batch-size', '32',
               '--iterations', str(ITERATIONS), '--warmup-steps', '2000', '--scheduler', 'wsd',
               '--wsd-fract-decay', '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
               '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
               '--results-base-folder', str(root), '--metrics-jsonl', str(metrics),
               '--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
               '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', group,
               '--wandb-tags', 'mnd_muon', '257M', '1xC', 'bf16', 'fp32_states', 'lr_tuning', 'pe5']
    return command, metrics, name


def verify_final(metrics: Path) -> dict:
    records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    finals = [r for r in records if r.get('event') == 'validation' and r.get('iter') == ITERATIONS]
    if not finals:
        raise RuntimeError('No final validation at the full token budget')
    record = finals[-1]
    loss = record.get('final-val/loss')
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise RuntimeError('Missing or nonfinite native final validation loss')
    if record.get('consumed_tokens') != ITERATIONS * TOKENS_PER_STEP:
        raise RuntimeError('Final token budget does not match the campaign')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    lr = os.environ['LR']
    date = os.environ.get('CAMPAIGN_DATE', '20261022')
    root = Path(os.environ.get('RESULTS_DIR', f'/workspace-SR006.nfs2/dimativator/mnd-muon-{date}'))
    command, metrics, name = build_command(lr, root, date)
    if args.dry_run:
        print(json.dumps(dict(command=command, metrics=str(metrics), checkpoint_saving=False), indent=2))
        return
    import torch

    if any(int(os.environ.get(key, '1')) != 1 for key in ('OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE')):
        raise RuntimeError('Launch each LR as a separate one-GPU job, without DDP')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each LR job must have exactly one visible GPU')
    os.chdir(Path(__file__).resolve().parents[2])
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    if (metadata.get('format') != 'packed_fineweb_h200_v3' or metadata.get('world_size') != 2
            or metadata.get('batch_size') != 16 or metadata.get('iterations', 0) < ITERATIONS):
        raise RuntimeError('Packed FineWeb metadata does not match the reference campaign')
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 10**9:
        raise RuntimeError('Output filesystem needs at least 1GB free')
    log = root / 'logs' / f'{name}.log'
    claim = root / 'claims' / f'{name}.json'
    claim.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if metrics.exists() or log.exists():
        raise RuntimeError(f'Refusing to overwrite existing run {name}')
    with claim.open('x') as output:
        json.dump(dict(lr=lr, command=command, checkpoint_saving=False, token_budget=ITERATIONS * TOKENS_PER_STEP), output)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_MODE='offline',
               WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey',
               WANDB_DIR=str(root / 'wandb'), WANDB_RUN_ID=f'{name}-{date}',
               TRITON_CACHE_DIR=f'/tmp/mnd-muon-{lr}-{os.getpid()}')
    (root / 'wandb').mkdir(exist_ok=True)
    subprocess.run(['df', '-h', str(root), str(DATA)], check=True)
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.free', '--format=csv'], check=True)
    print('MND_MUON_CONFIG=' + json.dumps(dict(lr=lr, wd=0.1, beta=0.9, polar_steps=5,
          matrix_scale='raw_polar', auxiliary_lr=lr, adam_betas=[0.95, 0.95],
          global_batch=128, checkpoint_saving=False, log=str(log))), flush=True)
    try:
        with log.open('x') as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=24 * 3600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print('\n'.join(log.read_text().splitlines()[-80:]), flush=True)
        raise
    result = verify_final(metrics)
    result.update(lr=float(lr), weight_decay=0.1, checkpoint_saving=False)
    final = root / 'finals' / f'{name}.json'
    final.parent.mkdir(exist_ok=True)
    with final.open('x') as output:
        json.dump(result, output, indent=2)
    print('RUN_VERIFIED=' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
