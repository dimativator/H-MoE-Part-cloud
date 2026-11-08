"""Gateway submission receipts for the four authorized 500M WD trials."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import netrc
from pathlib import Path
import re
import subprocess
import time

from anvil2_controller import BRANCH, LOGS, REPO, atomic_json, check_policy, command

ROOT = '/workspace-SR006.nfs3/dimativator/anvil2-500m-wd-20261108'
GATEWAY = Path('/home/dimativator/anvil2-500m-wd-20261108')
WDS = ('1e-1', '1e-2', '1e-3', '1e-4')


def submit(revision: str, wd: str | None = None) -> dict:
    check_policy()
    if wd is not None and wd not in WDS:
        raise ValueError('Unauthorized weight decay')
    label = 'preflight' if wd is None else 'wd' + wd
    receipt = GATEWAY / (label + '.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if not prior.get('job_id') or prior['code_revision'] != revision:
            raise RuntimeError('Existing or ambiguous submission requires reconciliation: ' + label)
        return prior
    if wd is not None:
        proof = json.loads((GATEWAY / 'preflight_verified.json').read_text())
        if proof['code_revision'] != revision or not 0 <= time.time() - proof['timestamp'] <= 600:
            raise RuntimeError('Storage/source preflight is stale')
        if any(v <= 5_500_000_000 for v in proof['available_bytes'].values()):
            raise RuntimeError('Insufficient disk reserve for four small logs/caches')
    inventory = command(['mlsub', 'list', '--active'])
    note = 'anvil2_500m_' + label + '_20261108'
    if inventory['returncode'] or note in inventory['stdout']:
        raise RuntimeError('Cannot rule out an existing submission: ' + label)
    auth = netrc.netrc().authenticators('wandb-radfan.ru')
    if not auth:
        raise RuntimeError('Online W&B credentials unavailable')
    key = auth[2]
    env = dict(CODE_REVISION=revision, RESULTS_DIR=ROOT if wd is None else ROOT + '/wd' + wd,
               WANDB_MODE='online', WANDB_API_KEY=key, VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS,
               CLOUD_DISABLE_CHECKPOINTS='1')
    if wd is None:
        env.update(ANVIL2_PREFLIGHT='1', CHECK_PACKED_DATA='1', CHECK_WANDB='1')
    else:
        env.update(WD=wd, LR='5e-3', CLOUD_TRAIN_WORLD_SIZE='2')
    args = ['mlsub', 'run', '--repo', REPO, '--branch', BRANCH, '--entry',
            'scripts/cloud/inspect_anvil2.py' if wd is None else 'scripts/cloud/run_anvil2_500m_wd.py',
            '--gpus', 'cpu' if wd is None else '2', '--image', 'efficient', '--no-pip', '--note', note]
    for name, value in env.items():
        args += ['--env', name + '=' + value]
    try:
        dry = command(args + ['--dry-run'])
    except subprocess.TimeoutExpired:
        raise RuntimeError('Cloud dry-run timed out before submission') from None
    if dry['returncode']:
        raise RuntimeError('Cloud dry-run failed: ' + dry['stderr'].replace(key, '[REDACTED]'))
    record = dict(state='submitting', timestamp=time.time(), wd=wd, code_revision=revision,
                  results_dir=env['RESULTS_DIR'], gpus=0 if wd is None else 2)
    with receipt.open('x') as stream:
        json.dump(record, stream)
    try:
        result = command(args, timeout=660)
        record.update({k: v.replace(key, '[REDACTED]') if isinstance(v, str) else v
                       for k, v in result.items()})
        match = re.search(r'lm-mpi-job-[0-9a-f-]+', record['stdout'])
        record['state'] = 'submitted' if match else 'ambiguous_submission'
        if match:
            record['job_id'] = match.group()
    except subprocess.TimeoutExpired:
        record['state'] = 'ambiguous_submission'
    atomic_json(receipt, record)
    if not record.get('job_id'):
        raise RuntimeError('Submission not confirmed: ' + label)
    return record


def verify_preflight(revision: str) -> dict:
    receipt = submit(revision)
    result = command(['mlsub', 'status', receipt['job_id']])
    status = json.loads(result['stdout'])
    if result['returncode'] or status.get('status', '').lower() != 'completed' or status.get('error_code') != 0:
        return dict(status=status, ready=False)
    result = command(['mlsub', 'logs', receipt['job_id'], '--tail', '350'])
    if result['returncode']:
        raise RuntimeError('Cannot read preflight logs')
    log = re.sub(r'\[\d+,\d+\]<(?:stdout|stderr)>:', '', result['stdout'])
    (GATEWAY / 'preflight.log').write_text(log)
    events = {}
    for line in log.splitlines():
        for marker in ('SPACE_SNAPSHOT=', 'PACKED_DATA=', 'WANDB_READY='):
            if marker in line:
                events[marker[:-1]] = json.loads(line.split(marker, 1)[1])
    if 'ANVIL2_REGRESSION_OK' not in log or not events.get('WANDB_READY', {}).get('ready'):
        raise RuntimeError('Preflight regressions or online W&B check failed')
    proof = events['SPACE_SNAPSHOT']
    if proof['logs_dir'] != LOGS or any(v <= 5_500_000_000 for v in proof['available_bytes'].values()):
        raise RuntimeError('Preflight disk reserve unavailable')
    data = events['PACKED_DATA']
    expected = dict(format='packed_fineweb_h200_v3', world_size=2, batch_size=16, sequence_length=1024)
    if any(data[k] != v for k, v in expected.items()) or data['iterations'] < 75457:
        raise RuntimeError('Packed data mismatch')
    proof.update(code_revision=revision, ready=True, packed_data=data,
                 nfs2_write_usage=False, regression_passed=True)
    atomic_json(GATEWAY / 'preflight_verified.json', proof)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'verify', 'launch'))
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    GATEWAY.mkdir(parents=True, exist_ok=True)
    with (GATEWAY / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            value = submit(args.revision)
        elif args.action == 'verify':
            value = verify_preflight(args.revision)
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                value = list(pool.map(lambda wd: submit(args.revision, wd), WDS))
            atomic_json(GATEWAY / 'submissions.json', dict(timestamp=time.time(), trials=value))
        print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
