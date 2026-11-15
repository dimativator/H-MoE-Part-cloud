"""500M LionMuon p=5: paired timing and independent extended WSD runs."""
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
ROOT = Path('/workspace-SR006.nfs3/dimativator/lionmuon-time-target-500m-20261115')
DATA = Path(os.environ.get('DATASETS_DIR', '/workspace-SR006.nfs3/dimativator/fineweb-h200-packed'))
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
BASELINES = {'1xc': 75457, '2xc': 150914}


def emit(marker: str, value: dict) -> None:
    print(marker + '=' + json.dumps(value, allow_nan=False), flush=True)


def space() -> dict:
    paths = ('/workspace-SR006.nfs3', LOGS, '/home/jovyan', '/tmp', '/')
    values = {}
    for path in paths:
        stat = os.statvfs(path)
        values[path] = stat.f_bavail * stat.f_frsize
    record = dict(timestamp=time.time(), logs_dir=LOGS, available_bytes=values,
                  code_revision=os.environ['CODE_REVISION'], ready=all(v > 5_100_000_000 for v in values.values()))
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
    space()
    if not Path('/workspace-SR006.nfs3').resolve().is_mount():
        raise RuntimeError('NFS3 mount missing')
    for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            stat = os.statvfs(path)
            record = dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize)
        except OSError as exc:
            record = dict(path=path, telemetry_error=str(exc))
        emit('NFS2_READ_ONLY', dict(timestamp=time.time(), used_for_writes=False, **record))
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
    cache = Path('/tmp') / ('lion-ttt-' + os.environ.get('TRIAL', 'inspect') + f'-rank{RANK}')
    cache.mkdir(exist_ok=True)
    os.environ.update(CLOUD_DISK_GUARD_ROOT=str(ROOT),
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
        '--opt', optimizer, '--lr', '1e-3', '--reference-aux-lr', '1e-3', '--weight-decay', '1e-4',
        '--beta1', '0.9', '--beta2', '0.99', '--eps', '1e-8', '--grad-clip', '1',
        '--batch-size', '8', '--acc-steps', '16', '--eval-batch-size', '32',
        '--iterations', str(target), '--scheduler', 'wsd', '--warmup-steps', '2000',
        '--wsd-fract-decay', repr((decay + 0.5) / target), '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
        '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
        '--results-base-folder', str(ROOT), '--metrics-jsonl', str(ROOT / name / 'metrics.jsonl')]
    if benchmark:
        args += ['--early-stop-iteration', '240', '--eval-interval', '1000000', '--eval-batches', '1', '--log-interval', '0']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', 'LionMuon_time_target_500M_20261115',
            '--wandb-tags', '500M', 'bf16', 'p5', 'time_target', '1.05x', 'no_checkpoints', name]
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
        'WANDB_RUN_ID': 'lion-ttt-500m-' + name + '-20261115', 'WANDB_RESUME': 'never',
        'LION_TTT_BENCHMARK': '1' if benchmark else '0'}
    command = [sys.executable, *arguments(name, optimizer, target, decay, benchmark)]
    emit('RUN_CONFIG', dict(name=name, rank=RANK, command=command, checkpoints=False, global_batch=128))
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
                    if RANK == 0:
                        temporary = ROOT / ('launcher_disk_' + name + '.tmp')
                        temporary.write_text(json.dumps(snapshot)); temporary.replace(temporary.with_suffix('.json'))
                if RANK == 0 and now - last_report >= 60:
                    last_report = now; rows = records(name)
                    emit('LIVE_STATUS', dict(name=name, timestamp=time.time(), latest=rows[-2:]))
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
    if RANK == 0 and not benchmark:
        final = [r for r in records(name) if r['event'] == 'validation'][-1]
        loss = final.get('final-val/loss')
        if final['iter'] != target or final['consumed_tokens'] != target * 131072 or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise RuntimeError('Exact finite final validation missing')
        result = dict(name=name, status='complete', target=target, decay_steps=decay, checkpoint_saving=False,
                      wandb_url='https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + env['WANDB_RUN_ID'], **final)
        (ROOT / 'finals' / (name + '.json')).write_text(json.dumps(result)); emit('RUN_COMPLETE', result)


def plan() -> dict:
    timing = {}
    for name in ('muon_reference', 'lion_muon'):
        rows = [r for r in records('bench_' + name) if r['event'] == 'ttt_benchmark']
        if len(rows) != 10 or [r['iter'] for r in rows] != list(range(60, 241, 20)):
            raise RuntimeError('Incomplete paired timing: ' + name)
        timing[name] = dict(seconds_per_step=sum(r['seconds'] for r in rows) / 200,
                            block_seconds_per_step=[r['seconds_per_step'] for r in rows], steps=200, warmup=40)
    muon, lion = (timing[n]['seconds_per_step'] for n in ('muon_reference', 'lion_muon'))
    targets = {}
    for name, baseline in BASELINES.items():
        target = math.floor(baseline * muon / (1.05 * lion))
        if target <= baseline:
            raise RuntimeError('Measured speedup does not permit extending stable phase')
        targets[name] = dict(baseline_steps=baseline, target_steps=target, decay_steps={'1xc': 7546, '2xc': 15091}[name],
            extra_stable_steps=target - baseline, consumed_tokens=target * 131072,
            muon_seconds=baseline * muon, lion_seconds=target * lion,
            predicted_speedup=baseline * muon / (target * lion))
    return dict(code_revision=os.environ['CODE_REVISION'], timing=timing, targets=targets,
                method='200 measured steps after 40 warmup, 20-step blocks, maximum elapsed across 4 ranks',
                training_time_excludes='evaluation, setup, compilation warmup')


def main() -> None:
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(128 + signum))
    metadata = setup(); stage = os.environ['CAMPAIGN_STAGE']
    if stage == 'preflight':
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
        emit('WANDB_READY', dict(ready=True)); return
    if stage == 'inspect':
        for name in ('1xc', '2xc', 'bench_muon_reference', 'bench_lion_muon'):
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
    if stage == 'benchmark':
        for optimizer in ('muon_reference', 'lion_muon'):
            run('bench_' + optimizer, optimizer, 75457, 7545, True)
        if RANK == 0:
            value = plan()
            value['packed_data_iterations'] = metadata['iterations']
            value['data_extension_required'] = any(metadata['iterations'] < t['target_steps'] for t in value['targets'].values())
            (ROOT / 'timing_plan.json').write_text(json.dumps(value, indent=2)); emit('TIMING_PLAN', value)
        return
    trial = os.environ['TRIAL']
    value = json.loads(Path(os.environ.get('LION_TTT_PLAN_PATH', str(ROOT / 'timing_plan.json'))).read_text())
    if value['code_revision'] != os.environ['CODE_REVISION']:
        raise RuntimeError('Timing source revision mismatch')
    target = value['targets'][trial]
    if metadata['iterations'] < target['target_steps']:
        raise RuntimeError('Packed dataset does not cover extended horizon')
    run(trial, 'lion_muon', target['target_steps'], target['decay_steps'], False)


if __name__ == '__main__':
    main()
