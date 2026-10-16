"""CPU-only, indexed snapshot of the matched codec diagnostic pilots."""
import base64
import gzip
import hashlib
import json
import time

from inspect_horizon_adam_257m import log_tail, metric_history, read_json
from run_horizon_diagnostic_257m import GROUP, PILOT_STEPS, ROOT


def main() -> None:
    progress, histories, errors = [], [], []
    for directory in sorted((ROOT / GROUP).glob('horizon_diag257m_*_attempt*')):
        manifest = read_json(directory / 'campaign.json')
        if not manifest or 'variant' not in manifest:
            errors.append(f'Unreadable manifest: {directory}')
            continue
        history, problems = metric_history(directory / 'metrics.jsonl')
        errors.extend(problems)
        status = read_json(directory / 'status.json') or {}
        latest = history[-1] if history else None
        common = dict(name=manifest['name'], lr=manifest['lr'], variant=manifest['variant'],
                      attempt=manifest['attempt'], status=status.get('phase', 'unknown'))
        histories.append(dict(**common, history=history))
        progress.append(dict(**common, path=str(directory), status_detail=status,
                             latest_metric=latest, completed_steps=max((r['iter'] for r in history), default=0),
                             metric_age_seconds=time.time() - latest['timestamp'] if latest else None,
                             target_steps=PILOT_STEPS, smoke=read_json(directory / 'smoke.json'),
                             done=read_json(directory / 'done.json'), failure=read_json(directory / 'failure.json'),
                             log_tail=log_tail(directory / 'train.log')))
    snapshot = dict(timestamp=time.time(), group=GROUP, runs=progress, errors=errors)
    archive = dict(progress=snapshot, history=dict(group=GROUP, runs=histories))
    compressed = gzip.compress(json.dumps(archive, sort_keys=True, allow_nan=False).encode())
    digest = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start:start + 900] for start in range(0, len(encoded), 900)]
    snapshot_id = str(time.time_ns())
    for part, data in enumerate(chunks):
        print('HORIZON_SNAPSHOT_CHUNK=' + json.dumps(dict(snapshot_id=snapshot_id, part=part,
              total=len(chunks), data=data, sha256=digest)), flush=True)
    print('HORIZON_DIAGNOSTIC_PROGRESS=' + json.dumps(snapshot), flush=True)


if __name__ == '__main__':
    main()
