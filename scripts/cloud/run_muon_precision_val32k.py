"""Table-matched 500M Muon precision runs with fixed 32k validation."""
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
sys.path.insert(0, str(REPO))
ROOT = Path('/home/jovyan/dimativator/muon-precision-val32k-20261128')
DATA = Path(os.environ.get('DATASETS_DIR', '/workspace-SR006.nfs3/dimativator/fineweb-h200-packed'))
LOGS = '/home/jovyan/mlsub-logs'
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
MONITOR = None
LAST_SPACE = None
TRIALS = ('fp8-act', 'fp8-full', 'fp8-states')


def emit(marker: str, value: dict) -> None:
    print(marker + '=' + json.dumps(value, allow_nan=False), flush=True)


def space() -> dict:
    global LAST_SPACE
    paths = (LOGS, '/home/jovyan', '/tmp', '/', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3')
    values = {}
    for path in paths:
        stat = os.statvfs(path)
        values[path] = stat.f_bavail * stat.f_frsize
    record = dict(timestamp=time.time(), logs_dir=LOGS, available_bytes=values,
                  code_revision=os.environ['CODE_REVISION'], ready=all(v > 5_000_000_000 for v in values.values()))
    try:
        stat = os.statvfs('/workspace-SR006.nfs2')
        record['nfs2'] = dict(available_bytes=stat.f_bavail * stat.f_frsize, bootstrap_logs_on_same_filesystem=os.stat(LOGS).st_dev == os.stat('/workspace-SR006.nfs2').st_dev)
    except OSError as exc:
        record['nfs2'] = dict(telemetry_error=str(exc))
    LAST_SPACE = record
    emit('SPACE_SNAPSHOT', record)
    if not record['ready']:
        emit('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU: ОСТАНАВЛИВАЮ ЭКСПЕРИМЕНТ',
             {p: v for p, v in values.items() if v <= 5_000_000_000})
        raise RuntimeError('Storage reserve unavailable; stop tracked child')
    return record


def setup() -> dict:
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if revision != os.environ['CODE_REVISION'] or os.environ.get('VERIFIED_BOOTSTRAP_LOGS_DIR') != LOGS:
        raise RuntimeError('Source or bootstrap-log policy mismatch')
    if any(os.stat(p).st_dev == os.stat("/workspace-SR006.nfs3").st_dev for p in ("/home/jovyan", LOGS, "/tmp", "/")):
        raise RuntimeError("A writing filesystem resolves to NFS3")
    space()
    if not Path('/workspace-SR006.nfs3').resolve().is_mount():
        raise RuntimeError('NFS3 mount missing')
    for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            stat = os.statvfs(path)
            record = dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize)
        except OSError as exc:
            record = dict(path=path, telemetry_error=str(exc))
        emit('NFS2_TELEMETRY', dict(timestamp=time.time(), **record))
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    expected = dict(format='packed_fineweb_h200_v3', world_size=2, batch_size=16, sequence_length=1024)
    if any(metadata[k] != v for k, v in expected.items()):
        raise RuntimeError('Unexpected packed data configuration')
    emit('PACKED_DATA', {k: metadata[k] for k in (*expected, 'iterations')})
    for path in (ROOT, *ROOT.parents):
        if path.is_symlink():
            raise RuntimeError('Output symlink forbidden')
    for child in ('logs', 'finals', 'rendezvous', 'wandb', 'evals_cache'):
        (ROOT / child).mkdir(parents=True, exist_ok=True)
    cache = Path('/tmp') / ('lion-valci-' + os.environ.get('TRIAL', 'inspect') + f'-rank{RANK}')
    cache.mkdir(exist_ok=True)
    os.environ.update(LIONMUON_VALCI='1', CLOUD_DISK_GUARD_ROOT=str(ROOT),
        CLOUD_DISK_GUARD_PATHS=json.dumps([str(ROOT), '/home/jovyan', '/tmp', '/', LOGS, '/workspace-SR006.nfs2', '/workspace-SR006.nfs3']),
        TMPDIR=str(cache), TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), WANDB_DIR=str(ROOT / 'wandb'),
        WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey', WANDB_INIT_TIMEOUT='60',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', PYTORCH_ALLOC_CONF='expandable_segments:True')
    os.chdir(REPO)
    return metadata


def records(name: str) -> list[dict]:
    path = ROOT / name / 'metrics.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def arguments(name: str, optimizer: str, target: int, decay: int, benchmark: bool) -> list[str]:
    args = ['src/main.py', '--experiment-name', name, '--distributed-backend', 'nccl',
        '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb', '--datasets-dir', str(DATA),
        '--fineweb-replay-world-size', '1', '--fineweb-replay-layout', 'concat',
        '--sequence-length', '1024', '--streaming', '--workers', '8',
        '--eval-cache-dir', str(ROOT / 'evals_cache'),
        '--model', 'llama', '--n-layer', '18', '--n-embd', '1280', '--n-head', '20', '--multiple-of', '256',
        '--dtype', 'bfloat16', '--model-parameter-dtype', 'float32', '--dropout', '0',
        '--opt', optimizer, '--lr', '1e-3', '--weight-decay', '1e-4',
        '--beta1', '0.9', '--beta2', '0.99', '--eps', '1e-7', '--grad-clip', '1',
        '--batch-size', '8', '--acc-steps', '16', '--eval-batch-size', '32',
        '--iterations', str(target), '--scheduler', 'wsd', '--warmup-steps', '2000' if name == 'fp8-states' else '7000',
        '--wsd-fract-decay', repr((decay + 0.5) / target), '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
        '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
        '--results-base-folder', str(ROOT), '--metrics-jsonl', str(ROOT / name / 'metrics.jsonl')]
    if benchmark:
        args += ['--early-stop-iteration', '240', '--eval-interval', '1000000', '--eval-batches', '1', '--log-interval', '0']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', 'Muon_precision_val32k_500M_20261128',
            '--wandb-tags', '500M', 'Muon', 'val32768', '1xC', 'no_checkpoints', name]
    if name in ('fp8-act', 'fp8-full'):
        args += ['--fp8', '--fp8-fabit', 'E4M3', '--fp8-fwbit', 'E4M3',
                 '--fp8-babit', 'E5M2', '--fp8-bwbit', 'E5M2', '--fp8-group-size', '16']
    if name in ('fp8-full', 'fp8-states'):
        args += ['--fp8-optim', '--fp8-qgroup-size', '128', '--fp8-first-order-bit', 'E4M3',
                 '--fp8-second-order-bit', 'E4M3', '--fp8-expansion', 'expand']
    return args


def run(name: str, optimizer: str, target: int, decay: int, benchmark: bool) -> None:
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) != 4:
        raise RuntimeError('Exactly four MPI workers required')
    ready = ROOT / 'rendezvous' / (name + '.json')
    if RANK == 0:
        if records(name) or ready.exists():
            raise RuntimeError('Duplicate run refused: ' + name)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('8.8.8.8', 80)); address = connection.getsockname()[0]
        temporary = ready.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(master_addr=os.environ.get('MASTER_ADDR', address), master_port=29500)))
        temporary.replace(ready)
    started = time.monotonic()
    while not ready.exists():
        if time.monotonic() - started > 180:
            raise RuntimeError('Rendezvous unavailable')
        time.sleep(1)
    rendezvous = json.loads(ready.read_text())
    env = {**os.environ, 'RANK': str(RANK), 'WORLD_SIZE': '4',
        'LOCAL_RANK': os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'),
        'MASTER_ADDR': rendezvous['master_addr'], 'MASTER_PORT': str(rendezvous['master_port']),
        'WANDB_MODE': 'disabled' if benchmark else 'online',
        'WANDB_RUN_ID': 'muon32k-' + name + '-20261128', 'WANDB_RESUME': 'never',
        'EXPECTED_GLOBAL_BATCH': '128', 'CLOUD_DISK_GUARD_ROOT': str(ROOT / name)}
    command = [sys.executable, *arguments(name, optimizer, target, decay, benchmark)]
    emit('RUN_CONFIG', dict(name=name, rank=RANK, command=command, checkpoints=False, global_batch=128))
    (ROOT / name).mkdir(exist_ok=True)
    log = ROOT / 'logs' / f'{name}_rank{RANK}.log'
    with log.open('x') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        emit('CHILD_PROCESS', dict(name=name, rank=RANK, pid=process.pid, process_group=process.pid))
        last_disk = last_report = 0.0
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - last_disk >= 30:
                    last_disk = now; snapshot = space()
                    temporary = ROOT / ('launcher_disk_' + name + '_rank' + str(RANK) + '.tmp')
                    temporary.write_text(json.dumps(snapshot)); temporary.replace(temporary.with_suffix('.json'))
                if RANK == 0 and now - last_report >= 60:
                    last_report = now; rows = records(name)
                    payload = dict(status='running', name=name, timestamp=time.time(), latest=rows[-2:],
                        space=LAST_SPACE, code_revision=os.environ['CODE_REVISION'], pid=process.pid, rank=RANK,
                        rank_storage={str(i): json.loads(p.read_text()) for i in range(4)
                            if (p := ROOT / ('launcher_disk_' + name + '_rank' + str(i) + '.json')).exists()})
                    if MONITOR is not None:
                        MONITOR.summary['worker_status'] = payload
                    emit('LIVE_STATUS', payload)
                time.sleep(1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            if process.returncode:
                print('\n'.join(log.read_text().splitlines()[-80:]), flush=True)
    checkpoint_paths = [str(p) for p in (ROOT / name).rglob('*') if p.suffix in ('.pt', '.pth')]
    if checkpoint_paths:
        raise RuntimeError('Unexpected checkpoint artifacts: ' + repr(checkpoint_paths))
    (ROOT / name / ('completed_rank' + str(RANK) + '.json')).write_text(json.dumps(
        dict(rank=RANK, timestamp=time.time(), code_revision=os.environ['CODE_REVISION'], checkpoint_paths=[])))
    if RANK == 0 and not benchmark:
        final = [r for r in records(name) if r['event'] == 'validation'][-1]
        loss = final.get('final-val/loss')
        if final['iter'] != target or final['consumed_tokens'] != target * 131072 or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise RuntimeError('Exact finite final validation missing')
        result = dict(name=name, status='complete', target=target, decay_steps=decay, checkpoint_saving=False,
                      wandb_url='https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + env['WANDB_RUN_ID'], **final)
        (ROOT / 'finals' / (name + '.json')).write_text(json.dumps(result)); emit('RUN_COMPLETE', result)



if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    trial = os.environ['TRIAL']
    if trial not in TRIALS or os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise ValueError('Invalid role or checkpoint setting')
    metadata = setup()
    if metadata['iterations'] < 75457:
        raise ValueError('Insufficient training data')
    import hashlib
    validation = DATA.parent / 'fineweb-validation-32768-20261125/validation.uint16'
    if hashlib.sha256(validation.read_bytes()).hexdigest() != 'bc23cac94158ff98af16de2b31bf113752b0e5de1891786a27e9d279ef98c232':
        raise ValueError('Validation digest mismatch')
    run(trial, 'muon', 75457, 7546, False)
