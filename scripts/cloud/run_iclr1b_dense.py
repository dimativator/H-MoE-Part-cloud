"""Five authorized 1B dense BF16 trials, matching 500M settings, no checkpoints."""

import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('RESULTS_DIR', '/workspace-SR006.nfs3/dimativator/iclr1b-dense-20261106'))
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
MODEL_PARAMETERS = 1_003_974_144
TARGET_STEPS = 153_194
TRIALS = {
    'muon_fp32': ('muon', 'bf16_fp32states', '1e-3', '1e-4', TARGET_STEPS, 2),
    'muon_fp8': ('muon', 'w16a16g32_fp8states', '1e-3', '1e-4', TARGET_STEPS, 2),
    'adamw_fp8': ('triton_coat_adamw', 'w16a16g32_fp8states', '1e-3', '1e-4', TARGET_STEPS, 2),
    'slimadam_fp32': ('slim_adam', 'bf16_fp32states', '5e-4', '1e-4', TARGET_STEPS, 2),
    'slimadam_fp8': ('slim_adam', 'w16a16g32_fp8states', '5e-4', '1e-4', TARGET_STEPS, 2),
}


def emit(marker: str, value: dict) -> None:
    print(marker + '=' + json.dumps(value), flush=True)


def disk_snapshot() -> dict:
    paths = ('/workspace-SR006.nfs3', LOGS, '/home/jovyan', '/tmp', '/')
    values = {}
    for path in paths:
        stat = os.statvfs(path)
        values[path] = stat.f_bavail * stat.f_frsize
    result = dict(timestamp=time.time(), logs_dir=LOGS, available_bytes=values,
                  code_revision=os.environ['CODE_REVISION'], ready=all(v > 5_100_000_000 for v in values.values()))
    emit('SPACE_SNAPSHOT', result)
    if not result['ready']:
        emit('CLOUD_STORAGE_EMERGENCY', dict(message='CRITICALLY LOW CLOUD.RU STORAGE; STOPPING THIS TRACKED RUN', **result))
        raise RuntimeError('Storage reserve unavailable; no launch or continuation')
    return result


def preflight() -> None:
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if revision != os.environ['CODE_REVISION']:
        raise RuntimeError('Unexpected source revision')
    if os.environ.get('VERIFIED_BOOTSTRAP_LOGS_DIR') != LOGS:
        raise RuntimeError('NFS3 bootstrap logs must be verified on gateway')
    allowed_root = Path('/workspace-SR006.nfs3/dimativator/iclr1b-dense-20261106')
    if ROOT != allowed_root and allowed_root not in ROOT.parents:
        raise RuntimeError('Unexpected output root')
    for path in (ROOT, *ROOT.parents):
        if path.is_symlink():
            raise RuntimeError('Symlink output path is not allowed')
    disk_snapshot()
    for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            stat = os.statvfs(path)
            record = dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize, used_for_writes=False)
        except OSError as exc:
            record = dict(path=path, telemetry_error=str(exc), used_for_writes=False)
        emit('NFS2_READ_ONLY', record)
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    expected = dict(format='packed_fineweb_h200_v3', world_size=2, batch_size=16, sequence_length=1024)
    if any(metadata[k] != v for k, v in expected.items()) or metadata['iterations'] < TARGET_STEPS:
        raise RuntimeError('Unexpected packed FineWeb configuration')
    emit('PACKED_DATA', {k: metadata[k] for k in (*expected, 'iterations')})
    ROOT.mkdir(parents=True, exist_ok=True)
    for child in ('logs', 'wandb', 'finals', 'rendezvous'):
        (ROOT / child).mkdir(exist_ok=True)
    cache = Path('/tmp') / ('iclr1b-' + os.environ.get('TRIAL', 'inspect') + f'-rank{RANK}')
    cache.mkdir(exist_ok=True)
    os.environ.update(CLOUD_DISK_GUARD_ROOT=str(ROOT),
        CLOUD_DISK_GUARD_PATHS=json.dumps([str(ROOT), '/home/jovyan', '/tmp', '/', LOGS]),
        TMPDIR=str(cache), TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
        WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey',
        WANDB_DIR=str(ROOT / 'wandb'), WANDB_INIT_TIMEOUT='60', WANDB_HTTP_TIMEOUT='20',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', PYTORCH_ALLOC_CONF='expandable_segments:True')
    os.chdir(REPO)


def arguments(trial: str, smoke: bool = False) -> list[str]:
    opt, precision, lr, wd, target, world = TRIALS[trial]
    name = trial + ('_smoke' if smoke else '')
    args = ['scripts/cloud/train_iclr1b_audit.py', '--experiment-name', name, '--distributed-backend', 'nccl',
        '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb', '--datasets-dir', str(DATA),
        '--fineweb-replay-world-size', '1', '--fineweb-replay-layout', 'concat',
        '--eval-cache-dir', '/home/jovyan/evals_cache', '--sequence-length', '1024', '--streaming', '--workers', '8',
        '--model', 'llama', '--n-layer', '30', '--n-embd', '1536', '--n-head', '24', '--multiple-of', '256',
        '--dtype', 'bfloat16', '--model-parameter-dtype', 'float32', '--opt', opt, '--lr', lr,
        '--weight-decay', wd, '--beta1', '0.9', '--beta2', '0.99',
        '--grad-clip', '1', '--density', '0.25', '--update_gap', '50', '--coord_choice', 'columns',
        '--inactive_update_rule', 'sign_sgd', '--batch-size', str(32 // world), '--acc-steps', str(4 * world),
        '--eval-batch-size', '32', '--iterations', str(target), '--scheduler', 'wsd', '--warmup-steps', '2000',
        '--wsd-fract-decay', '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
        '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
        '--results-base-folder', str(ROOT), '--metrics-jsonl', str(ROOT / name / 'metrics.jsonl')]
    if opt == 'muon':
        args += ['--lite-ns-steps', '6', '--lite-muon-theta', '0.95']
    if precision in ('fp8_full', 'fp8_mixed'):
        args += ['--fp8', '--fp8-fabit', 'E4M3', '--fp8-fwbit', 'E4M3', '--fp8-babit', 'E5M2', '--fp8-bwbit', 'E5M2', '--fp8-group-size', '16']
    if precision in ('fp8_full', 'w16a16g32_fp8states'):
        args += ['--fp8-optim', '--fp8-qgroup-size', '128', '--fp8-first-order-bit', 'E4M3', '--fp8-second-order-bit', 'E4M3', '--fp8-expansion', 'expand']
    if smoke:
        args += ['--early-stop-iteration', '2', '--eval-interval', '2', '--eval-batches', '1', '--log-interval', '1']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
            '--downstream-eval-enabled', '--downstream-eval-interval', '2000', '--downstream-task-group', 'basic_v2',
            '--lm-eval-enabled', '--lm-eval-interval', '2000', '--lm-eval-datasets', 'wikitext103',
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', 'ICLR1B_dense_1xC_20261106',
            '--wandb-tags', '1B', precision, 'wd1e-4', '1xC', 'independent', 'no_checkpoints']
    return args


def latest_metrics(path: Path) -> dict:
    result = {}
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            for line in stream.read().splitlines():
                try:
                    value = json.loads(line)
                    result[value['event']] = value
                except (ValueError, KeyError):
                    continue
    return result


def train(trial: str, smoke: bool) -> None:
    opt, precision, lr, wd, target, world = TRIALS[trial]
    name = trial + ('_smoke' if smoke else '')
    metrics = ROOT / name / 'metrics.jsonl'
    ready = ROOT / 'rendezvous' / (name + '.json')
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) != world:
        raise RuntimeError('MPI allocation mismatch')
    if RANK == 0:
        if metrics.exists() or ready.exists():
            raise RuntimeError('Refusing duplicate run')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('8.8.8.8', 80))
            address = connection.getsockname()[0]
        temporary = ready.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(master_addr=os.environ.get('MASTER_ADDR', address), master_port=29500)))
        temporary.replace(ready)
    started = time.monotonic()
    while not ready.exists():
        if time.monotonic() - started > 180:
            raise RuntimeError('Rendezvous unavailable')
        time.sleep(1)
    rendezvous = json.loads(ready.read_text())
    env = {**os.environ, 'RANK': str(RANK), 'WORLD_SIZE': str(world),
        'LOCAL_RANK': os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'),
        'MASTER_ADDR': rendezvous['master_addr'], 'MASTER_PORT': str(rendezvous['master_port']),
        'WANDB_MODE': 'disabled' if smoke else 'online', 'WANDB_RUN_ID': 'iclr1b-' + trial + '-20261106', 'WANDB_RESUME': 'never'}
    (ROOT / name).mkdir(exist_ok=True)
    env['CLOUD_DISK_GUARD_ROOT'] = str(ROOT / name)
    env['CLOUD_EXPECTED_PARAMETERS'] = str(MODEL_PARAMETERS)
    env['CLOUD_EXPECTED_STATE_FORMAT'] = 'fp8' if 'fp8states' in precision else 'fp32'
    command = [sys.executable, *arguments(trial, smoke)]
    log = ROOT / 'logs' / f'{name}_rank{RANK}.log'
    emit('RUN_CONFIG', dict(trial=trial, smoke=smoke, rank=RANK, world=world, command=command, checkpoints=False))
    with log.open('x') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        last_disk = last_report = 0.0
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - last_disk >= 30:
                    last_disk = now
                    snapshot = disk_snapshot()
                    if RANK == 0:
                        status = ROOT / ('launcher_disk_' + trial + '.json')
                        temporary = status.with_suffix('.tmp')
                        temporary.write_text(json.dumps(snapshot))
                        temporary.replace(status)
                if RANK == 0 and now - last_report >= 60:
                    last_report = now
                    emit('LIVE_STATUS', dict(trial=trial, smoke=smoke, timestamp=time.time(), metrics=latest_metrics(metrics)))
                time.sleep(1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if process.returncode:
                print('\n'.join(log.read_text().splitlines()[-100:]), flush=True)
    if RANK == 0:
        final = latest_metrics(metrics).get('validation', {})
        expected = 2 if smoke else target
        loss = final.get('final-val/loss', final.get('val/loss'))
        if final.get('iter') != expected or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise RuntimeError('Exact finite final validation missing')
        if final['consumed_tokens'] != expected * 128 * 1024:
            raise RuntimeError('Final token budget mismatch')
        result = dict(trial=trial, optimizer=opt, precision=precision, lr=lr, wd=wd,
            world=world, target=expected, checkpoint_saving=False, smoke=smoke,
            status='smoke_verified' if smoke else 'local_complete_pending_online_verification', **final)
        if not smoke:
            result['wandb_url'] = 'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + env['WANDB_RUN_ID']
        (ROOT / 'finals' / (name + '.json')).write_text(json.dumps(result))
        emit('SMOKE_VERIFIED' if smoke else 'RUN_LOCAL_COMPLETE', result)
    if 'OPTIMIZER_STATE_AUDIT=' not in log.read_text():
        raise RuntimeError('First-step optimizer-state audit missing')
    if any((ROOT / name).glob('**/*.pt')):
        raise RuntimeError('Unexpected checkpoint artifact')


def stop_on_signal(signum: int, _frame: object) -> None:
    # Raise through train() so its finally block stops the exact child group.
    raise SystemExit(128 + signum)


def main() -> None:
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    preflight()
    if os.environ.get('CAMPAIGN_STAGE') == 'preflight':
        import base64
        import urllib.request
        token = base64.b64encode(('api:' + os.environ['WANDB_API_KEY']).encode()).decode()
        request = urllib.request.Request('https://wandb-radfan.ru/graphql',
            data=json.dumps(dict(query='{ viewer { username } }')).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + token})
        with urllib.request.urlopen(request, timeout=20) as response:
            response_data = json.load(response)
        if not (response_data.get('data') or {}).get('viewer') or response_data.get('errors'):
            raise RuntimeError('Online W&B unavailable')
        emit('WANDB_READY', dict(ready=True))
        import torch
        sys.path.insert(0, str(REPO / 'src'))
        from optim.optimization import get_optimizer
        from models.llama import Llama
        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=50304, sequence_length=1024, n_layer=30, n_embd=1536,
            n_head=24, multiple_of=256, dropout=0.0, rmsnorm_eps=1e-5, init_std=0.02,
            fp8=False, qkv_clipping=False)
        with torch.device('meta'):
            model = Llama(config)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != MODEL_PARAMETERS:
            raise RuntimeError('Unexpected parameter count')
        if TARGET_STEPS != int(20 * parameters // (128 * 1024)):
            raise RuntimeError('Unexpected Chinchilla endpoint')
        emit('RUNTIME', dict(torch=torch.__version__, factory_import=True, parameters=parameters,
            target_steps=TARGET_STEPS, target_tokens=TARGET_STEPS * 128 * 1024))
        return
    if os.environ.get('CAMPAIGN_STAGE') == 'inspect':
        for path in sorted(ROOT.glob('finals/*.json')):
            emit('FINAL', json.loads(path.read_text()))
        for trial in TRIALS:
            emit('PROGRESS', dict(trial=trial, metrics=latest_metrics(ROOT / trial / 'metrics.jsonl')))
        for path in sorted(ROOT.glob('launcher_disk_*.json')):
            emit('WORKER_DISK', json.loads(path.read_text()))
        emit('CHECKPOINT_AUDIT', dict(paths=[str(p) for p in ROOT.glob('**/*.pt')]))
        return
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1' or os.environ.get('WANDB_MODE') != 'online':
        raise RuntimeError('No checkpoints and online W&B required')
    trial = os.environ['TRIAL']
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv'], check=True)
    train(trial, True)
    train(trial, False)


if __name__ == '__main__':
    main()
