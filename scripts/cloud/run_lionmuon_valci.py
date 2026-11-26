"""Nine valci 500M LionMuon ablations, four GPUs and no checkpoints."""
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
ROOT = Path('/home/jovyan/dimativator/lionmuon-valci-500m-20261126')
DATA = Path(os.environ.get('DATASETS_DIR', '/workspace-SR006.nfs3/dimativator/fineweb-h200-packed'))
LOGS = '/home/jovyan/mlsub-logs'
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
MONITOR = None
LAST_SPACE = None
TRIALS = ('p5', 'linear', 'early', 'muon')


def emit(marker: str, value: dict) -> None:
    print(marker + '=' + json.dumps(value, allow_nan=False), flush=True)


def space() -> dict:
    global LAST_SPACE
    paths = (LOGS, '/home/jovyan', '/tmp', '/')
    values = {}
    for path in paths:
        stat = os.statvfs(path)
        values[path] = stat.f_bavail * stat.f_frsize
    record = dict(timestamp=time.time(), logs_dir=LOGS, available_bytes=values,
                  code_revision=os.environ['CODE_REVISION'], ready=all(v > 5_100_000_000 for v in values.values()))
    try:
        stat = os.statvfs('/workspace-SR006.nfs2')
        record['nfs2'] = dict(available_bytes=stat.f_bavail * stat.f_frsize, bootstrap_logs_on_same_filesystem=os.stat(LOGS).st_dev == os.stat('/workspace-SR006.nfs2').st_dev)
    except OSError as exc:
        record['nfs2'] = dict(telemetry_error=str(exc))
    LAST_SPACE = record
    emit('SPACE_SNAPSHOT', record)
    if not record['ready']:
        emit('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU: ОСТАНАВЛИВАЮ ЭКСПЕРИМЕНТ',
             {p: v for p, v in values.items() if v <= 5_100_000_000})
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
        CLOUD_DISK_GUARD_PATHS=json.dumps([str(ROOT), '/home/jovyan', '/tmp', '/', LOGS]),
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
        '--opt', optimizer, '--lion-ablation', {'p5':'original', 'linear':'probability_down', 'early':'stagger_early', 'muon':'original'}[name], '--lion-compensation-gamma', '0', '--lion-lr-ratio', '0.01', '--lion-beta1', '0.95', '--lion-beta2', '0.99',
        '--lion-neuron-beta', '0.95', '--lion-probability-seed', '20261117', '--lr', '1e-3', '--reference-aux-lr', '1e-3', '--weight-decay', '1e-4',
        '--beta1', '0.9', '--beta2', '0.99', '--eps', '1e-7', '--grad-clip', '1',
        '--batch-size', '8', '--acc-steps', '16', '--eval-batch-size', '32',
        '--iterations', str(target), '--scheduler', 'wsd', '--warmup-steps', '2000',
        '--wsd-fract-decay', repr((decay + 0.5) / target), '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
        '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
        '--results-base-folder', str(ROOT), '--metrics-jsonl', str(ROOT / name / 'metrics.jsonl')]
    if benchmark:
        args += ['--early-stop-iteration', '240', '--eval-interval', '1000000', '--eval-batches', '1', '--log-interval', '0']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', 'LionMuon_valci_500M_20261126',
            '--wandb-tags', '500M', 'bf16', 'valci_ablation', '1xC', 'no_checkpoints', name]
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
        'WANDB_RUN_ID': 'lion-valci-500m-' + name + '-20261126', 'WANDB_RESUME': 'never',
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


def main() -> None:
    global MONITOR
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(128 + signum))
    metadata = setup(); stage = os.environ['CAMPAIGN_STAGE']
    if RANK == 0:
        import wandb
        MONITOR = wandb.init(entity='andrey', project='fp8-pretrain',
            id='lion-valci-500m-' + os.environ.get('TRIAL', stage) + '-monitor-20261126',
            group='LionMuon_valci_500M_20261126', job_type='monitor',
            dir=str(ROOT / 'wandb'), resume='never')
    if stage == 'preflight':
        import hashlib
        raw = (DATA.parent / 'fineweb-validation-32768-20261125/validation.uint16').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == 'bc23cac94158ff98af16de2b31bf113752b0e5de1891786a27e9d279ef98c232'
        assert raw[:1024*1025*2] == (DATA / metadata['validation']['file']).read_bytes()
        emit('DATA_INVENTORY', dict(files=[dict(name=p.name, bytes=p.stat().st_size) for p in DATA.iterdir() if p.is_file()], metadata=metadata))
        import torch
        from third_party.lion_muon.lion_muon import LionMuon
        from third_party.lion_muon.muon import Muon
        emit('RUNTIME', dict(torch=torch.__version__, optimizers=[LionMuon.__name__, Muon.__name__]))
        import base64
        import urllib.request
        auth = base64.b64encode(('api:' + os.environ['WANDB_API_KEY']).encode()).decode()
        request = urllib.request.Request('https://wandb-radfan.ru/graphql', data=json.dumps(dict(query='{ viewer { username } }')).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + auth})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        if not (data.get('data') or {}).get('viewer') or data.get('errors'):
            raise RuntimeError('Online W&B unavailable')
        subprocess.run([sys.executable, 'tests/test_lion_muon_cadence.py'], check=True,
                       env={**os.environ, 'PYTHONPATH': str(REPO) + ':' + str(REPO / 'src'), 'TORCH_COMPILE_DISABLE': '1'})
        subprocess.run([sys.executable, 'scripts/cloud/run_factorized_lionmuon.py', '--check-data'], check=True)
        emit('WANDB_READY', dict(ready=True)); return
    if stage == 'inspect':
        for name in TRIALS:
            rows = records(name)
            emit('CURVE', dict(name=name, validation=[r for r in rows if r['event'] == 'validation'],
                              latest=rows[-3:]))
        for path in sorted(ROOT.glob('finals/*.json')):
            emit('FINAL', json.loads(path.read_text()))
        for path in sorted(ROOT.glob('launcher_disk_*.json')):
            emit('WORKER_DISK', json.loads(path.read_text()))
        emit('CHECKPOINT_AUDIT', dict(paths=[str(p) for p in ROOT.glob('**/*.pt')]))
        return
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('Checkpoints must be disabled')
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], check=True)
    trial = os.environ['TRIAL']
    if trial not in TRIALS or metadata['iterations'] < 75457:
        raise RuntimeError('Invalid trial or insufficient dataset horizon')
    run(trial, 'muon_reference' if trial == 'muon' else 'lion_muon', 75457, 7546, False)


if __name__ == '__main__':
    try:
        main()
        if MONITOR is not None:
            MONITOR.summary['worker_status'] = dict(status='complete', timestamp=time.time(),
                space=space(), code_revision=os.environ['CODE_REVISION'])
            MONITOR.finish()
    except BaseException as exc:
        if MONITOR is not None:
            MONITOR.summary['worker_status'] = dict(status='failed', error=str(exc), timestamp=time.time(),
                space=LAST_SPACE, code_revision=os.environ['CODE_REVISION'])
            MONITOR.finish(exit_code=1)
        raise
