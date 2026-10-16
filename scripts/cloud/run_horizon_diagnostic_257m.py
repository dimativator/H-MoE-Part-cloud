"""Matched, bounded 257M codec pilots using the existing campaign data path."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from run_horizon_adam_257m import (
    DATA, REPO, ROOT, SCIENCE, TARGET, TOKENS_PER_STEP, atomic_json, data_report, train_args,
)

GROUP = 'horizon_adam_257m_diagnostic_20261016'
PILOT_STEPS = 5000
LR = '5e-4'
JOB_STARTED = time.monotonic()
# Leave room within the aggregate 20 GPU-hour cap for startup and the calibration attempt.
WALL_SECONDS = {'adamw': 6000, 'horizon16': 23400, 'identity8': 36000}
VARIANTS = {
    'adamw': dict(optimizer='adamw', bits=None, codec=None),
    'horizon16': dict(optimizer='horizon_adam', bits=16, codec='learned'),
    'identity8': dict(optimizer='horizon_adam', bits=8, codec='identity'),
}


def pilot_args(name: str, variant: str, metrics: Path, smoke: bool = False) -> list[str]:
    config = VARIANTS[variant]
    args = train_args(name, LR, metrics, smoke=False, resume=None)
    replacements = {
        '--opt': config['optimizer'], '--wandb-group': GROUP,
        '--latest-ckpt-interval': '0', '--permanent-ckpt-interval': '0',
        '--eval-interval': '2' if smoke else '500',
        '--eval-batches': '1' if smoke else '32', '--log-interval': '1' if smoke else '50',
    }
    if config['bits'] is not None:
        replacements['--horizon-adam-bits'] = str(config['bits'])
    for flag, value in replacements.items():
        args[args.index(flag) + 1] = value
    args += ['--model-parameter-dtype', 'float32', '--no-local-save',
             '--early-stop-iteration', '2' if smoke else str(PILOT_STEPS)]
    if config['codec'] is not None:
        args += ['--horizon-adam-codec', config['codec']]
    return args


def verify_metrics(path: Path, target: int) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        for key in ('train/loss', 'val/loss', 'final-val/loss'):
            if key in row and not math.isfinite(row[key]):
                raise RuntimeError(f'Nonfinite {key} at step {row.get("iter")}')
    if max(row['iter'] for row in rows) != target:
        raise RuntimeError(f'Pilot did not stop at exactly {target}')
    final = [row for row in rows if row['iter'] == target and 'val/loss' in row]
    if not final or final[-1]['consumed_tokens'] != target * TOKENS_PER_STEP:
        raise RuntimeError('Missing exact-budget pilot validation')
    return dict(step=target, consumed_tokens=target * TOKENS_PER_STEP,
                val_loss=final[-1]['val/loss'], validation=final[-1])


def preflight() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(ROOT)
    # Pilots retain JSONL, stdout and offline W&B, but do not write new checkpoints.
    if disk.free < 2_000_000_000:
        raise RuntimeError(f'Less than 2 GB available for pilot logs: {disk.free}')
    result = dict(timestamp=time.time(), data=data_report(), free_bytes=disk.free,
                  required_free_bytes=2_000_000_000, checkpoint_policy='disabled_for_short_pilots')
    print('HORIZON_DIAGNOSTIC_PREFLIGHT=' + json.dumps(result), flush=True)
    return result


def train(name: str, variant: str, directory: Path, smoke: bool) -> dict:
    prefix = 'smoke' if smoke else 'train'
    metrics = directory / ('smoke.jsonl' if smoke else 'metrics.jsonl')
    log = directory / f'{prefix}.log'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1']
    command += pilot_args(name, variant, metrics, smoke)
    atomic_json(directory / f'{prefix}_command.json', dict(command=command))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_MODE='offline',
               WANDB_DIR=str(ROOT / 'wandb'), WANDB_RUN_ID=name,
               TRITON_CACHE_DIR=f'/tmp/horizon-diagnostic-{os.getpid()}')
    (ROOT / 'wandb').mkdir(parents=True, exist_ok=True)
    # Independent process group lets a wall-time timeout terminate torchrun and its workers.
    import signal
    with log.open('x') as output:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            remaining = max(1, WALL_SECONDS[variant] - (time.monotonic() - JOB_STARTED))
            process.wait(timeout=min(900, remaining) if smoke else remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError('Pilot exceeded its wall-time cap')
    if process.returncode:
        from collections import deque
        with log.open() as source:
            print(''.join(deque(source, maxlen=80)), flush=True)
        raise RuntimeError(f'Training exited with code {process.returncode}')
    return verify_metrics(metrics, 2 if smoke else PILOT_STEPS)


def campaign() -> None:
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Exactly one MPI worker required')
    variant = os.environ['VARIANT']
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant}')
    attempt = int(os.environ.get('ATTEMPT', '0'))
    if attempt < 0:
        raise ValueError('Negative ATTEMPT')
    name = f'horizon_diag257m_{variant}_attempt{attempt}'
    directory = ROOT / GROUP / name
    directory.mkdir(parents=True, exist_ok=False)
    science = {**SCIENCE, **VARIANTS[variant], 'lr': float(LR), 'pilot_steps': PILOT_STEPS,
               'scheduler_iterations': TARGET, 'checkpoint_policy': 'disabled'}
    manifest = dict(name=name, group=GROUP, lr=LR, variant=variant, attempt=attempt,
                    science=science, timestamp=time.time(), resume_from=None)
    atomic_json(directory / 'campaign.json', manifest)
    status = directory / 'status.json'
    try:
        atomic_json(status, dict(phase='preflight', timestamp=time.time()))
        preflight()
        subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv'], check=True)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU required')
        atomic_json(status, dict(phase='model_smoke', timestamp=time.time()))
        atomic_json(directory / 'smoke.json', train(name + '_smoke', variant, directory, True))
        atomic_json(status, dict(phase='training', timestamp=time.time()))
        result = train(name, variant, directory, False)
        atomic_json(directory / 'done.json', {**manifest, **result})
        atomic_json(status, dict(phase='complete', timestamp=time.time()))
        print('HORIZON_DIAGNOSTIC_DONE=' + json.dumps(result), flush=True)
    except Exception as error:
        failure = dict(timestamp=time.time(), message=str(error), traceback=traceback.format_exc())
        atomic_json(directory / 'failure.json', failure)
        atomic_json(status, dict(phase='failed', **failure))
        raise


if __name__ == '__main__':
    os.chdir(REPO)
    mode = os.environ.get('MODE', 'run')
    if mode == 'run':
        campaign()
    elif mode == 'preflight':
        preflight()
    elif mode == 'inspect':
        from inspect_horizon_diagnostic_257m import main
        main()
    else:
        raise ValueError(mode)
