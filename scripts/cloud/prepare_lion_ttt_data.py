"""Prepare a private deterministic FineWeb tail for the measured LionMuon horizon."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import run_lionmuon_time_target as common

sys.path[:0] = [str(common.REPO / 'src'), str(common.REPO)]
ROOT = common.ROOT
SOURCE = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
DESTINATION = ROOT / 'extended_data'
PLAN = ROOT / 'single_gpu_timing/timing_plan.json'
TRAIN_REVISION = '3e7604f1210d42eb1c83f34af24b9f542fee81c8'
BASE_STEPS = 157100
BLOCKS_PER_STEP_PER_SOURCE = 64
BLOCK_BYTES = 1025 * 2
RESERVE = 5_000_000_000
HEADROOM = 200_000_000
CHUNK_BLOCKS = 8192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    guard(len(encoded.encode()))
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded)
    temporary.replace(path)


def guard(planned_bytes: int = 0) -> dict:
    snapshot = common.space()
    free = snapshot['available_bytes']['/workspace-SR006.nfs3']
    if free <= RESERVE + HEADROOM + planned_bytes:
        common.emit('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU: ЗАПИСЬ ХВОСТА ОСТАНОВЛЕНА',
                    dict(available_bytes=free, planned_bytes=planned_bytes, reserve_bytes=RESERVE))
        raise RuntimeError('Insufficient NFS3 reserve for planned dataset write')
    return snapshot


def required_bytes(target: int) -> int:
    return max(0, target - BASE_STEPS) * BLOCKS_PER_STEP_PER_SOURCE * BLOCK_BYTES * 2


def load_plan() -> tuple[dict, int]:
    if os.environ.get('TRAIN_REVISION') != TRAIN_REVISION:
        raise ValueError('Explicit frozen training revision required')
    plan = json.loads(PLAN.read_text())
    if plan.get('code_revision') != TRAIN_REVISION or plan.get('measurement_gpus') != 1 or plan.get('global_batch') != 128:
        raise ValueError('Timing plan identity does not match the authorized single-GPU benchmark')
    muon = float(plan['timing']['muon_reference']['seconds_per_step'])
    lion = float(plan['timing']['lion_muon']['seconds_per_step'])
    if not all(math.isfinite(v) and v > 0 for v in (muon, lion)):
        raise ValueError('Invalid measured step time')
    for name, baseline in common.BASELINES.items():
        row = plan['targets'][name]
        if row['baseline_steps'] != baseline or row['target_steps'] != math.floor(baseline * muon / (1.05 * lion)):
            raise ValueError('Target differs from the measured 1.05x time budget')
    return plan, max(int(row['target_steps']) for row in plan['targets'].values())


def validate_source(metadata: dict) -> None:
    expected = dict(format='packed_fineweb_h200_v3', iterations=BASE_STEPS,
        world_size=2, batch_size=16, sequence_length=1024, block_tokens=1025,
        microsteps_per_iteration=4, blocks_per_rank=BASE_STEPS * 64)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Unexpected source packed dataset dimensions')
    from data import fineweb_packed as packed
    for key, value in dict(manifest_fingerprint=packed.EXPECTED_MANIFEST_FINGERPRINT,
        split_plan_fingerprint=packed.EXPECTED_SPLIT_PLAN_FINGERPRINT,
        validation_blocks_sha256=packed.EXPECTED_VAL_BLOCKS_SHA256).items():
        if metadata[key] != value:
            raise ValueError('Unexpected source fingerprint: ' + key)
    if len(metadata['ranks']) != 2:
        raise ValueError('Exactly two source ranks required')
    for rank, source in enumerate(metadata['ranks']):
        if source['rank'] != rank or source['blocks'] != BASE_STEPS * 64 or len(source['segments']) != 3:
            raise ValueError('Unexpected rank or source segment coverage')
        if sum(segment['blocks'] for segment in source['segments']) != source['blocks']:
            raise ValueError('Source block total mismatch')
        if sum(segment['bytes'] for segment in source['segments']) != source['bytes']:
            raise ValueError('Source byte total mismatch')
        for segment in source['segments']:
            if segment['bytes'] != segment['blocks'] * BLOCK_BYTES or (SOURCE / segment['file']).stat().st_size != segment['bytes']:
                raise ValueError('Source segment size mismatch')


def validate_state(payload: dict, rank: int, metadata: dict) -> dict:
    if payload['blocks_written'] != (BASE_STEPS - 150914) * 64:
        raise ValueError('Third-segment state is not at its final block')
    state = payload['stream_state']
    expected = dict(rank=rank, world_size=2, worker_id=0, num_data_workers=1,
        split='train', emitted_block_count=BASE_STEPS * 64,
        manifest_fingerprint=metadata['manifest_fingerprint'],
        split_plan_fingerprint=metadata['split_plan_fingerprint'])
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError('Source stream state identity or position mismatch')
    if set(state['committed_cursor']) != {'assigned_row_group_index', 'row_in_group'}:
        raise ValueError('Unexpected stream cursor schema')
    if any(not isinstance(v, int) or v < 0 for v in state['committed_cursor'].values()):
        raise ValueError('Invalid stream cursor')
    if any(not isinstance(token, int) or not 0 <= token < 65536 for token in state['token_buffer']):
        raise ValueError('Invalid stream token buffer')
    return state


def private_metadata(source: dict, target: int, tails: list[dict]) -> dict:
    result = copy.deepcopy(source)
    for rank, entry in enumerate(result['ranks']):
        for segment in entry['segments']:
            segment['file'] = str(SOURCE / segment['file'])
        entry['segments'].append(copy.deepcopy(tails[rank]))
        entry['blocks'] = target * 64
        entry['bytes'] = entry['blocks'] * BLOCK_BYTES
    result['validation']['file'] = str(SOURCE / result['validation']['file'])
    result.update(iterations=target, blocks_per_rank=target * 64,
        lion_ttt_data_provenance=dict(train_revision=TRAIN_REVISION, code_revision=os.environ['CODE_REVISION'],
            source_steps=BASE_STEPS, source_metadata_sha256=sha256_file(SOURCE / 'packed_metadata.json')))
    return result


def safe_destination() -> None:
    for path in (DESTINATION, *DESTINATION.parents):
        if path.is_symlink():
            raise RuntimeError('Private dataset output must not traverse symlinks')
    DESTINATION.mkdir(exist_ok=True)


def prepare_rank(rank: int, target: int) -> None:
    common.setup()
    safe_destination()
    plan, checked_target = load_plan()
    if checked_target != target or target <= BASE_STEPS:
        raise ValueError('Requested tail does not match the timing plan')
    metadata = json.loads((SOURCE / 'packed_metadata.json').read_text())
    validate_source(metadata)
    payload = json.loads((SOURCE / f'train_rank{rank}.third.state.json').read_text())
    state = validate_state(payload, rank, metadata)
    blocks = (target - BASE_STEPS) * 64
    final = DESTINATION / f'train_rank{rank}.lion_ttt.uint16'
    receipt = final.with_suffix('.json')
    if final.exists() or receipt.exists():
        if not final.is_file() or not receipt.is_file():
            raise RuntimeError('Incomplete tail finalization requires inspection')
        saved = json.loads(receipt.read_text())
        if saved['blocks'] != blocks or final.stat().st_size != blocks * BLOCK_BYTES or sha256_file(final) != saved['sha256']:
            raise RuntimeError('Existing tail does not match the measured target')
        common.emit('TAIL_ALREADY_COMPLETE', saved)
        return
    part = final.with_suffix(final.suffix + '.part')
    progress = DESTINATION / f'tokenization_rank{rank}.progress.json'
    if part.exists() or progress.exists():
        raise RuntimeError('Partial tokenization exists; inspect before an explicit resume')
    from scripts.cloud.prepare_fineweb_h200_packed import build_snapshot, load_manifest, make_stream
    import numpy as np
    manifest = load_manifest(common.REPO / 'scripts/cloud/fineweb_h200_manifest.json.gz.b64')
    if not manifest.dataset_root.startswith('https://huggingface.co/'):
        raise RuntimeError('Expected remote HF source manifest')
    snapshot = build_snapshot(manifest)
    stream = make_stream(manifest, snapshot.plan, rank)
    digest = hashlib.sha256()
    written = 0
    try:
        stream.load_state_dict(state)
        guard(min(CHUNK_BLOCKS, blocks) * BLOCK_BYTES)
        with part.open('xb') as output:
            while written < blocks:
                count = min(CHUNK_BLOCKS, blocks - written)
                generated = [next(stream) for _ in range(count)]
                array = np.asarray(generated, dtype=np.int64)
                if array.shape != (count, 1025) or np.any(array < 0) or np.any(array >= 65536):
                    raise ValueError('Unexpected GPT-2 token block')
                data = array.astype('<u2').tobytes()
                guard(len(data))
                output.write(data); output.flush(); os.fsync(output.fileno())
                digest.update(data); written += count
                atomic(progress, dict(rank=rank, blocks_written=written, target_blocks=blocks,
                    stream_state=stream.state_dict(), train_revision=TRAIN_REVISION,
                    source_state_sha256=sha256_file(SOURCE / f'train_rank{rank}.third.state.json')))
                common.emit('TAIL_PROGRESS', dict(rank=rank, blocks_written=written, target_blocks=blocks))
    finally:
        stream.close()
    if part.stat().st_size != blocks * BLOCK_BYTES:
        raise RuntimeError('Tail size mismatch')
    value = dict(rank=rank, file=final.name, blocks=blocks, bytes=blocks * BLOCK_BYTES,
        sha256=digest.hexdigest(), train_revision=TRAIN_REVISION)
    if sha256_file(part) != value['sha256']:
        raise RuntimeError('Persisted tail hash mismatch')
    part.replace(final)
    atomic(receipt, value)
    common.emit('TAIL_COMPLETE', value)


def export(run: Any, status: str, target: int | None, **fields: Any) -> None:
    value = dict(status=status, timestamp=time.time(), train_revision=TRAIN_REVISION,
        code_revision=os.environ['CODE_REVISION'], dataset_dir=str(DESTINATION if target and target > BASE_STEPS else SOURCE),
        target_steps=target, space=common.space(), **fields)
    atomic(ROOT / 'data_prepare_status.json', value)
    run.summary['data_prepare_status'] = value
    common.emit('DATA_PREPARE_STATUS', value)


def execute_rank(run: Any, rank: int, target: int) -> None:
    path = ROOT / 'logs' / f'data_prepare_rank{rank}.log'
    if path.exists():
        raise RuntimeError('Existing tokenization log requires inspection before another attempt')
    argv = [sys.executable, str(Path(__file__).resolve()), '--rank', str(rank), '--target', str(target)]
    with path.open('x') as output:
        child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        common.emit('DATA_PREPARE_CHILD', dict(rank=rank, pid=child.pid, process_group=child.pid))
        last = 0.0
        try:
            while child.poll() is None:
                if time.monotonic() - last >= 30:
                    last = time.monotonic()
                    export(run, 'preparing', target, current_rank=rank, child_pid=child.pid)
                time.sleep(1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        if child.returncode:
            raise RuntimeError(f'Dataset rank {rank} failed: ' + path.read_text()[-6000:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=(0, 1))
    parser.add_argument('--target', type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(128 + signum))
    if args.rank is not None:
        if args.target is None:
            parser.error('--target is required with --rank')
        prepare_rank(args.rank, args.target); return
    common.setup()
    if common.DATA != SOURCE or os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1':
        raise RuntimeError('Expected shared source and disabled training checkpoints')
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) != 1:
        raise RuntimeError('One CPU launcher process required')
    import wandb
    run = wandb.init(entity='andrey', project='fp8-pretrain', id='lion-ttt-single-data_prepare-20261115',
        name='LionMuon deterministic FineWeb tail', dir=str(ROOT / 'wandb'), resume='never',
        settings=wandb.Settings(init_timeout=60))
    target = None
    try:
        plan, target = load_plan()
        if target <= BASE_STEPS:
            export(run, 'complete', target, extension_required=False); return
        metadata_path = SOURCE / 'packed_metadata.json'
        original = metadata_path.read_bytes()
        metadata = json.loads(original); validate_source(metadata)
        guard(required_bytes(target))
        safe_destination()
        export(run, 'starting', target, expected_extra_bytes=required_bytes(target))
        for rank in (0, 1):
            execute_rank(run, rank, target)
        if metadata_path.read_bytes() != original:
            raise RuntimeError('Shared source metadata changed during preparation')
        tails = []
        for rank in (0, 1):
            path = DESTINATION / f'train_rank{rank}.lion_ttt.uint16'
            value = json.loads(path.with_suffix('.json').read_text())
            if sha256_file(path) != value['sha256']:
                raise RuntimeError('Final tail hash mismatch')
            tails.append(value)
        output = private_metadata(metadata, target, tails)
        atomic(DESTINATION / 'packed_metadata.json', output)
        export(run, 'complete', target, extension_required=True, expected_extra_bytes=required_bytes(target),
            metadata_sha256=sha256_file(DESTINATION / 'packed_metadata.json'), tails=tails,
            source_metadata_sha256=hashlib.sha256(original).hexdigest())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        try:
            export(run, 'failed', target, error=str(exc))
        except (OSError, RuntimeError):
            run.summary['data_prepare_status'] = dict(status='failed', timestamp=time.time(),
                train_revision=TRAIN_REVISION, target_steps=target, error=str(exc), space_unavailable=True)
        raise
    finally:
        run.finish()


if __name__ == '__main__':
    main()
