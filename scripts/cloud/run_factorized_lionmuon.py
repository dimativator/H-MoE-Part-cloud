"""Four independent 257M rank trials, with log-only storage and worker watchdogs."""
import hashlib
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
ROOT = Path('/workspace-SR006.nfs3/dimativator/lionmuon-factorized-257m-20261116/batch128')
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
NFS2 = '/workspace-SR006.nfs2'
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
STAGE = os.environ.get('FACTOR_STAGE', 'preflight')
FACTOR = int(os.environ.get('FACTOR_RANK', '0'))
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def emit(marker: str, value: dict) -> None:
    print(marker + '=' + json.dumps(value, allow_nan=False), flush=True)


def space() -> dict:
    values = {}
    for path in ('/workspace-SR006.nfs3', LOGS, '/home/jovyan', '/tmp', '/'):
        stat = os.statvfs(path)
        values[path] = stat.f_bavail * stat.f_frsize
    # The gateway policy now writes bootstrap logs to NFS3, never NFS2.
    # Still inspect NFS2 explicitly and record unavailable telemetry honestly.
    try:
        stat = os.statvfs(NFS2)
        nfs2 = dict(path=NFS2, available_bytes=stat.f_bavail * stat.f_frsize, used_for_writes=False)
    except OSError as exc:
        nfs2 = dict(path=NFS2, telemetry_error=str(exc), used_for_writes=False)
    record = dict(timestamp=time.time(), available_bytes=values, logs_dir=LOGS, nfs2=nfs2,
                  code_revision=os.environ['CODE_REVISION'], reserve_bytes=5_000_000_000,
                  ready=all(v > 5_100_000_000 for v in values.values()))
    emit('SPACE_SNAPSHOT', record)
    if not record['ready']:
        emit('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU', record)
        raise RuntimeError('Disk reserve unavailable; no automatic restart')
    return record


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def arguments(name: str) -> list[str]:
    return [sys.executable, 'src/main.py', '--experiment-name', name,
        '--distributed-backend', 'nccl', '--seed', '0', '--data-seed', '1337',
        '--dataset', 'fineweb', '--datasets-dir', str(DATA), '--sequence-length', '1024',
        '--streaming', '--workers', '8', '--eval-cache-dir', str(ROOT / 'evals_cache'),
        '--model', 'llama', '--n-layer', '12', '--n-embd', '1024', '--n-head', '8',
        '--multiple-of', '256', '--dtype', 'bfloat16', '--model-parameter-dtype', 'float32',
        '--dropout', '0', '--opt', 'lion_muon', '--lion-momentum-rank', str(FACTOR),
        '--lion-momentum-damping', '1e-8', '--lr', '1e-3', '--reference-aux-lr', '1e-3',
        '--weight-decay', '0.1', '--beta1', '0.9', '--beta2', '0.99', '--eps', '1e-7',
        '--grad-clip', '1', '--batch-size', '16', '--acc-steps', '8', '--eval-batch-size', '32',
        '--iterations', '39250', '--scheduler', 'wsd', '--warmup-steps', '3925',
        '--wsd-fract-decay', '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
        '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
        '--results-base-folder', str(ROOT), '--metrics-jsonl', str(ROOT / name / 'metrics.jsonl'),
        '--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
        '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', 'LionMuon_factorized_257M_20261116',
        '--wandb-tags', '257M', 'factorized_momentum', 'rank' + str(FACTOR), 'wd0.1', 'no_checkpoints']


def check_data() -> None:
    import numpy as np
    import torch
    from data import fineweb_packed as packed
    meta = json.loads((DATA / 'packed_metadata.json').read_text())
    expected = dict(format='packed_fineweb_h200_v3', world_size=2, batch_size=16,
        sequence_length=1024, manifest_fingerprint=packed.EXPECTED_MANIFEST_FINGERPRINT,
        split_plan_fingerprint=packed.EXPECTED_SPLIT_PLAN_FINGERPRINT,
        validation_blocks_sha256=packed.EXPECTED_VAL_BLOCKS_SHA256)
    if any(meta.get(k) != v for k, v in expected.items()) or meta['iterations'] < 39250:
        raise RuntimeError('Unexpected data identity or insufficient training data')
    for source in meta['ranks']:
        if source['blocks'] < 39250 * 4 * 16:
            raise RuntimeError('Insufficient per-rank data')
        for segment in source['segments']:
            path = DATA / segment['file']
            if path.stat().st_size != segment['bytes']:
                raise RuntimeError('Packed segment size mismatch')
        segment = source['segments'][0]
        tokens = np.fromfile(DATA / segment['file'], dtype='<u2', count=16*1025)
        if tokens.size != 16*1025 or tokens.max() >= 50304:
            raise RuntimeError('Invalid training tokens')
    val = meta['validation']
    if packed._sha256_file(DATA / val['file']) != val['sha256']:
        raise RuntimeError('Validation hash mismatch')
    emit('DATA_VERIFIED', dict(**expected, iterations=meta['iterations'], torch=torch.__version__))
    subprocess.run([sys.executable, 'tests/test_factorized_lion_momentum.py'], check=True,
                   env={**os.environ, 'PYTHONPATH': str(REPO), 'TORCH_COMPILE_DISABLE': '1'})


def main() -> None:
    if '--check-data' in sys.argv:
        check_data()
        return
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(128+s))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(128+s))
    if STAGE not in ('preflight', 'train'):
        raise ValueError('Unknown stage')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() != os.environ['CODE_REVISION']:
        raise RuntimeError('Code revision mismatch')
    if os.environ.get('VERIFIED_BOOTSTRAP_LOGS_DIR') != LOGS or os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('Bootstrap policy or checkpoint setting mismatch')
    initial_space = space()
    for child in ('logs', 'rendezvous', 'wandb', 'evals_cache'):
        (ROOT / child).mkdir(parents=True, exist_ok=True)
    name = 'preflight' if STAGE == 'preflight' else 'rank' + str(FACTOR)
    output = ROOT / name
    output.mkdir(exist_ok=True)
    cache = Path('/tmp') / ('factor-lion-' + name + '-rank' + str(RANK))
    cache.mkdir(exist_ok=True)
    os.environ.update(CLOUD_DISK_GUARD_ROOT=str(ROOT), CLOUD_DISK_GUARD_PATHS=json.dumps(
        [str(ROOT), LOGS, '/home/jovyan', '/tmp', '/']), TMPDIR=str(cache),
        TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), WANDB_DIR=str(ROOT / 'wandb'),
        WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', PYTORCH_ALLOC_CONF='expandable_segments:True')
    os.chdir(REPO)
    env = dict(os.environ)
    if STAGE == 'train':
        import torch
        if FACTOR not in (32, 64, 128, 256) or int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) != 2:
            raise RuntimeError('Exactly two MPI workers and authorized rank required')
        emit('GPU_ALLOCATION', dict(rank=RANK, visible_devices=torch.cuda.device_count(),
             device=torch.cuda.get_device_name(int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')))))
        ready = ROOT / 'rendezvous' / (name + '.json')
        if RANK == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(('8.8.8.8', 80))
                address = sock.getsockname()[0]
            with ready.open('x') as stream:
                json.dump(dict(master_addr=address, master_port=29500), stream)
        start = time.monotonic()
        while not ready.exists():
            if time.monotonic()-start > 180:
                raise RuntimeError('Rendezvous missing')
            time.sleep(1)
        rendezvous = json.loads(ready.read_text())
        env.update(EXPECTED_GLOBAL_BATCH='128', RANK=str(RANK), WORLD_SIZE='2', LOCAL_RANK=os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'),
            MASTER_ADDR=rendezvous['master_addr'], MASTER_PORT=str(rendezvous['master_port']),
            WANDB_RUN_ID='factor-lion-257m-b128-' + name + '-20261116', WANDB_RESUME='never')
        command = arguments(name)
    else:
        command = [sys.executable, __file__, '--check-data']
    run = None
    if RANK == 0:
        import wandb
        run = wandb.init(entity='andrey', project='fp8-pretrain',
            id='factor-lion-257m-b128-' + name + '-monitor-20261116',
            name='Factorized LionMuon ' + name + ' worker telemetry',
            group='LionMuon_factorized_257M_20261116', job_type='monitor',
            dir=str(ROOT / 'wandb'), resume='never')
    log = ROOT / 'logs' / (name + '_rank' + str(RANK) + '.log')
    process = None
    try:
        emit('RUN_CONFIG', dict(rank=RANK, factor_rank=FACTOR, command=command, checkpoints=False))
        with log.open('x') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            emit('CHILD_PROCESS', dict(rank=RANK, pid=process.pid, process_group=process.pid))
            last = 0
            while process.poll() is None:
                if time.monotonic()-last >= 30:
                    last = time.monotonic()
                    current = space()
                    payload = dict(status='running', stage=STAGE, factor_rank=FACTOR,
                        code_revision=os.environ['CODE_REVISION'], timestamp=time.time(), space=current,
                        pid=process.pid, latest=records(output / 'metrics.jsonl')[-2:])
                    (output / ('worker' + str(RANK) + '_disk.json')).write_text(json.dumps(payload))
                    if run is not None:
                        run.summary['worker_status'] = payload
                        emit('LIVE_STATUS', payload)
                time.sleep(1)
            if process.returncode:
                raise RuntimeError('Child failed with exit code ' + str(process.returncode))
        if STAGE == 'train' and RANK == 0:
            finals = [r for r in records(output / 'metrics.jsonl') if 'final-val/loss' in r]
            if (not finals or finals[-1]['iter'] != 39250 or finals[-1]['consumed_tokens'] != 5144576000
                or not math.isfinite(finals[-1]['final-val/loss'])):
                raise RuntimeError('Expected finite final validation absent')
        if list(output.rglob('*.pt')) or list(output.rglob('*.pth')):
            raise RuntimeError('Unexpected checkpoint artifact')
        result = dict(status='complete', stage=STAGE, factor_rank=FACTOR, timestamp=time.time(),
            code_revision=os.environ['CODE_REVISION'], space=space(), checkpoint_saving=False,
            latest=records(output / 'metrics.jsonl')[-2:])
        if run is not None:
            run.summary['worker_status'] = result
        (output / ('completed_rank' + str(RANK) + '.json')).write_text(json.dumps(result))
        emit('COMPLETE', result)
    except BaseException as exc:
        if run is not None:
            run.summary['worker_status'] = dict(status='failed', error=str(exc), timestamp=time.time(),
                code_revision=os.environ['CODE_REVISION'], stage=STAGE, factor_rank=FACTOR)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if log.exists():
            print('\n'.join(log.read_text().splitlines()[-60:]), flush=True)
        if run is not None:
            run.finish(exit_code=0 if process is not None and process.returncode == 0 else 1)


if __name__ == '__main__':
    main()
