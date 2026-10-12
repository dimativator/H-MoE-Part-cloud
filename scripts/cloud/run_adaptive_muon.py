"""Cloud LR sweep and 500M trunk/decay entrypoint for Muon² and NorMuon."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(os.environ.get('RESULTS_DIR', '/workspace-SR006.nfs3/dimativator/adaptive-muon-20261011'))
SOURCE_ROOT = ROOT
RECOVERY = os.environ.get('RECOVERY_RESULTS_DIR')
if RECOVERY:
    ROOT = Path(RECOVERY)
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
LRS = ('1e-4', '5e-4', '1e-3', '2e-3')
OPTS = ('muon2', 'normuon')
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
WORLD = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
MODE = os.environ.get('MODE', 'smoke')


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def wait_json(path: Path, timeout: int = 180) -> dict:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(1)
    return json.loads(path.read_text())


def final_metric(path: Path, target: int) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    finals = [r for r in records if r.get('event') == 'validation' and r.get('iter') == target]
    if not finals:
        raise RuntimeError(f'No final validation at {target}: {path}')
    result = finals[-1]
    loss = result.get('final-val/loss', result.get('val/loss'))
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise RuntimeError(f'Nonfinite final loss: {result}')
    if result['consumed_tokens'] != target * 128 * 1024:
        raise RuntimeError(f'Token budget mismatch: {result}')
    return {**result, 'val_loss': loss}


def train(opt: str, lr: str, size: int, phase: str, target: int, resume: Path | None = None) -> None:
    group = f'adaptive_muon_{size}M_{phase}_20261011'
    name = f'{opt}_{size}m_bf16_lr{lr}_wd1e-1_{phase}_2gpu'
    if RECOVERY:
        name += '_storage_retry1'
    decay_resume = resume is not None and phase == '1xC_decay'
    run_dir = ROOT / group / name
    metrics = run_dir / 'metrics.jsonl'
    done = ROOT / 'finals' / f'{name}.json'
    ready = ROOT / 'rendezvous' / f'{name}.json'
    if RANK == 0:
        if metrics.exists() or done.exists() or ready.exists():
            raise RuntimeError(f'Refusing duplicate or overwrite: {run_dir}')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('8.8.8.8', 80))
            address = connection.getsockname()[0]
        atomic_json(ready, {'master_addr': os.environ.get('MASTER_ADDR', address), 'master_port': 29500})
    rendezvous = wait_json(ready)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_MODE=os.environ.get('WANDB_MODE', 'offline'),
               WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey',
               WANDB_DIR=str(ROOT / 'wandb'), WANDB_RUN_ID=f'adapt-{name}-20261011',
               TRITON_CACHE_DIR=f'/tmp/adaptive-muon-{opt}-{RANK}-{os.getpid()}')
    if WORLD > 1:
        if WORLD != 2:
            raise RuntimeError('Exactly two MPI ranks required')
        env.update(RANK=str(RANK), WORLD_SIZE='2',
                   LOCAL_RANK=os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'),
                   MASTER_ADDR=rendezvous['master_addr'], MASTER_PORT=str(rendezvous['master_port']))
        launch = [sys.executable]
    else:
        launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    layers, width, heads = (12, 1024, 8) if size == 257 else (18, 1280, 20)
    args = ['src/main.py', '--distributed-backend', 'nccl', '--experiment-name', name,
            '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb', '--datasets-dir', str(DATA),
            '--fineweb-replay-world-size', '1', '--fineweb-replay-layout', 'concat',
            '--sequence-length', '1024', '--streaming', '--workers', '8',
            '--model', 'llama', '--n-layer', str(layers), '--n-embd', str(width),
            '--n-head', str(heads), '--multiple-of', '256', '--dtype', 'bfloat16', '--dropout', '0',
            '--opt', opt, '--lr', lr, '--weight-decay', '1e-1', '--momentum', '0.95',
            '--muon_beta2', '0.95', '--muon_ns_steps', '5', '--muon2-ns-steps', '5',
            '--beta1', '0.9', '--beta2', '0.95', '--eps', '1e-8', '--grad-clip', '1',
            '--batch-size', '16', '--acc-steps', '8', '--eval-batch-size', '32',
            '--iterations', str(target), '--scheduler', 'wsd', '--warmup-steps', '0' if decay_resume else '2000',
            '--wsd-fract-decay', '1.0' if decay_resume else '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
            '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0',
            '--results-base-folder', str(ROOT), '--metrics-jsonl', str(metrics),
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', group,
            '--wandb-tags', opt, f'{size}M', phase, 'bf16', 'fp32_states', 'lr_tuning', 'ns5']
    if MODE == 'smoke':
        args += ['--early-stop-iteration', '2', '--eval-interval', '2', '--eval-batches', '1', '--log-interval', '1', '--no-local-save']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50']
        if phase == '2xC' and resume is None:
            args += ['--inter-ckpts', '67911']
        else:
            args += ['--no-local-save']
        if resume:
            for filename in ('main.pt', 'worker_0.pt', 'worker_1.pt'):
                if not (resume / filename).is_file():
                    raise RuntimeError(f'Missing checkpoint file: {resume / filename}')
            args += ['--resume-from', str(resume)]
            if decay_resume:
                args += ['--decay-from-checkpoint']
    log = ROOT / 'logs' / f'{name}_rank{RANK}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN_START optimizer={opt} lr={lr} size={size} phase={phase} rank={RANK} log={log}', flush=True)
    try:
        with log.open('x') as output:
            subprocess.run(launch + args, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        print('\n'.join(log.read_text().splitlines()[-80:]), flush=True)
        raise
    if RANK == 0:
        if MODE == 'smoke':
            result = final_metric(metrics, 2)
            result.update(smoke=True, optimizer=opt, lr=lr, size=size)
            print('\n'.join(log.read_text().splitlines()[-25:]), flush=True)
        else:
            result = final_metric(metrics, target)
            result.update(optimizer=opt, lr=lr, size=size, phase=phase,
                          wandb_url=f'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/{env["WANDB_RUN_ID"]}')
        atomic_json(done, result)
        print('RUN_VERIFIED=' + json.dumps(result, sort_keys=True), flush=True)
    wait_json(done)


def best_lr(opt: str) -> str:
    finals = []
    for lr in LRS:
        path = SOURCE_ROOT / 'finals' / f'{opt}_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
        record = wait_json(path, timeout=1)
        if record.get('iter') != 39250:
            raise RuntimeError('Sweep not complete')
        finals.append(record)
    return min(finals, key=lambda r: r['val_loss'])['lr']


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[2])
    (ROOT / 'wandb').mkdir(parents=True, exist_ok=True)
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    assert metadata['format'] == 'packed_fineweb_h200_v3'
    assert metadata['iterations'] >= 150914
    assert metadata['world_size'] == 2 and metadata['batch_size'] == 16
    subprocess.run(['df', '-h', str(ROOT), str(DATA)], check=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], check=True)
    if MODE == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_adaptive_muon.py', '-v'],
                       env={**os.environ, 'PYTHONPATH': 'src:.'}, check=True)
        for opt in OPTS:
            train(opt, '1e-3', 257, 'smoke', 39250)
    elif MODE == 'sweep':
        lr = os.environ['LR']
        assert lr in LRS
        for opt in OPTS:
            train(opt, lr, 257, '1xC', 39250)
    elif MODE == '500m':
        opt = os.environ['OPTIMIZER']
        assert opt in OPTS
        # Require all eight sweep completions before either 500M launch.
        winners = {optimizer: best_lr(optimizer) for optimizer in OPTS}
        lr = winners[opt]
        if RANK == 0:
            atomic_json(ROOT / f'best_lr_{opt}.json', {'optimizer': opt, 'lr': lr})
        source = SOURCE_ROOT / 'adaptive_muon_500M_2xC_20261011' / f'{opt}_500m_bf16_lr{lr}_wd1e-1_2xC_2gpu' / 'ckpts' / '67911'
        if RECOVERY:
            import shutil
            if shutil.disk_usage(ROOT).free < 2 * 10**9:
                raise RuntimeError('Recovery output filesystem needs at least2GB free')
        train(opt, lr, 500, '2xC', 150914, resume=source if RECOVERY else None)
        train(opt, lr, 500, '1xC_decay', 75457, resume=source)
    else:
        raise ValueError(MODE)


if __name__ == '__main__':
    main()
