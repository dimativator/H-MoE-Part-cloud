"""CPU-only snapshot of this HorizonAdam campaign, including loss histories."""
from collections import deque
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from run_horizon_adam16_257m import DATA, GROUP, ROOT, TARGET, storage_report


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return dict(read_error=str(error), path=str(path))


def metric_history(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    by_step = {}
    errors = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get('iter'), int):
                    raise ValueError('Missing integer iter')
                for key in ('train/loss', 'val/loss', 'final-val/loss'):
                    if key in record and (not isinstance(record[key], (int, float)) or not math.isfinite(record[key])):
                        raise ValueError(f'Nonfinite {key}')
                by_step[(record.get('event'), record['iter'])] = record
            except (json.JSONDecodeError, ValueError) as error:
                errors.append(f'{path}:{line_number}: {error}')
    records = sorted(by_step.values(), key=lambda r: (r['iter'], r.get('timestamp', 0)))
    # Expected combined count is <900. Bound output if a pathological restart logs more.
    if len(records) > 1000:
        errors.append(f'History truncated from {len(records)} to latest 1000 records')
        records = records[-1000:]
    return records, errors


def log_tail(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(errors='replace') as source:
        return [line.rstrip()[:2000] for line in deque(source, maxlen=25)]


def main() -> None:
    campaign_dir = ROOT / GROUP
    progress = []
    histories = []
    errors = []
    for run_dir in sorted(campaign_dir.glob('horizon_adam_257m_b16_h32_lr*_wd1e-1_s0_attempt*')):
        if not run_dir.is_dir():
            continue
        manifest = read_json(run_dir / 'campaign.json')
        if manifest is None or 'lr' not in manifest:
            errors.append(f'Missing/unreadable campaign manifest: {run_dir}')
            continue
        history, history_errors = metric_history(run_dir / 'metrics.jsonl')
        errors.extend(history_errors)
        status = read_json(run_dir / 'status.json') or {}
        done = read_json(run_dir / 'done.json')
        failure = read_json(run_dir / 'failure.json')
        latest = history[-1] if history else None
        checkpoint = run_dir / 'ckpts' / 'latest'
        checkpoint_files = {}
        for filename in ('main.pt', 'worker_0.pt'):
            path = checkpoint / filename
            if path.exists():
                stat = path.stat()
                checkpoint_files[filename] = dict(bytes=stat.st_size, modified=stat.st_mtime)
        common = dict(name=manifest['name'], lr=manifest['lr'], attempt=manifest['attempt'],
                      status=status.get('phase', 'unknown'), resume_from=manifest.get('resume_from'))
        histories.append(dict(**common, history=history))
        progress.append(dict(**common, path=str(run_dir), status_detail=status, latest_metric=latest,
                             completed_steps=max((r['iter'] for r in history), default=0), target_steps=TARGET,
                             metric_age_seconds=time.time() - latest['timestamp'] if latest else None,
                             done=done, failure=failure, checkpoint_path=str(checkpoint),
                             checkpoint_files=checkpoint_files, log_tail=log_tail(run_dir / 'train.log')))
    metadata = read_json(DATA / 'packed_metadata.json')
    storage = storage_report(require_space=False) if ROOT.exists() else dict(path=str(ROOT), exists=False)
    snapshot = dict(timestamp=time.time(), group=GROUP, results_dir=str(ROOT), storage=storage,
                    data_path=str(DATA), data_metadata=metadata, runs=progress, errors=errors)
    # MPI logging inserts prefixes into long stdout lines and Cloud logs may
    # return lines in reverse order. Indexed short chunks survive both cases.
    archive = dict(progress=snapshot, history=dict(group=GROUP, runs=histories))
    compressed = gzip.compress(json.dumps(archive, sort_keys=True, allow_nan=False).encode())
    digest = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start:start + 900] for start in range(0, len(encoded), 900)]
    snapshot_id = str(time.time_ns())
    for part, data in enumerate(chunks):
        print('HORIZON_SNAPSHOT_CHUNK=' + json.dumps(dict(snapshot_id=snapshot_id, part=part,
              total=len(chunks), data=data, sha256=digest)), flush=True)
    compact = dict(timestamp=snapshot['timestamp'], group=GROUP, results_dir=str(ROOT),
                   storage=storage, errors=errors,
                   runs=[{key: run[key] for key in ('name', 'lr', 'attempt', 'status',
                         'completed_steps', 'target_steps', 'metric_age_seconds')}
                         for run in progress])
    print('HORIZON_PROGRESS_JSON=' + json.dumps(compact, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
