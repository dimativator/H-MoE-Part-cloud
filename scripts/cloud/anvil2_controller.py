"""Gateway-only submission and inspection for six authorized ANVIL II trials.

Credentials are read from the gateway netrc and never persisted in receipts.
An ambiguous submission is recorded and never retried automatically.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import netrc
from pathlib import Path
import re
import subprocess
import time
import urllib.request

BRANCH = 'codex/soft-lion-muon-cloud-r1-20261105'
REPO = 'https://github.com/dimativator/H-MoE-Part-cloud.git'
ROOT = '/workspace-SR006.nfs3/dimativator/anvil2-257m-lr-20261106'
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
GATEWAY = Path('/home/dimativator/anvil2-257m-lr-20261106')
LRS = ('1e-2', '5e-3', '2e-3', '1e-3', '5e-4', '1e-4')
CONFIG_KEYS = ('opt', 'lr', 'weight_decay', 'iterations', 'seed', 'data_seed', 'dtype',
               'model_parameter_dtype', 'batch_size', 'acc_steps', 'sequence_length',
               'no_local_save', 'latest_ckpt_interval', 'permanent_ckpt_interval', 'inter_ckpts',
               'n_layer', 'n_embd', 'n_head', 'scheduler', 'warmup_steps', 'wsd_fract_decay',
               'wsd_final_lr_scale', 'decay_type', 'beta1', 'beta2', 'anvil2_engage_fraction',
               'anvil2_momentum_warmup_steps', 'anvil2_momentum_cooldown_steps')


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(args: list[str], timeout: int = 40) -> dict:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def check_policy() -> None:
    if json.loads(Path('/etc/mlsub/policy.json').read_text())['logs_dir'] != LOGS:
        raise RuntimeError('Bootstrap logging must use verified NFS3; NFS2 is not writable')


def submit(revision: str, lr: str | None = None) -> dict:
    check_policy()
    if lr is not None and lr not in LRS:
        raise ValueError('Unauthorized LR')
    label = 'preflight' if lr is None else 'lr' + lr
    receipt = GATEWAY / (label + '.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if not prior.get('job_id') or prior['code_revision'] != revision:
            raise RuntimeError('Existing or ambiguous submission requires reconciliation: ' + label)
        return prior
    if lr is not None:
        proof = json.loads((GATEWAY / 'preflight_verified.json').read_text())
        if proof['code_revision'] != revision or not 0 <= time.time() - proof['timestamp'] <= 600:
            raise RuntimeError('Storage/source preflight is stale')
        # Reserve room for six small logs/caches as well as the mandatory 5GB.
        if any(v <= 5_500_000_000 for v in proof['available_bytes'].values()):
            raise RuntimeError('Insufficient disk reserve for parallel submissions')
    inventory = command(['mlsub', 'list', '--active'])
    note = 'anvil2_257m_' + label + '_20261106'
    if inventory['returncode'] or note in inventory['stdout']:
        raise RuntimeError('Cannot rule out an existing submission: ' + label)
    auth = netrc.netrc().authenticators('wandb-radfan.ru')
    if not auth:
        raise RuntimeError('Online W&B credentials unavailable')
    key = auth[2]
    env = dict(CODE_REVISION=revision, RESULTS_DIR=ROOT if lr is None else ROOT + '/lr' + lr,
               WANDB_MODE='online', WANDB_API_KEY=key, VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS,
               CLOUD_DISABLE_CHECKPOINTS='1')
    if lr is None:
        env.update(ANVIL2_PREFLIGHT='1', CHECK_PACKED_DATA='1', CHECK_WANDB='1')
    else:
        env.update(LR=lr, CLOUD_TRAIN_WORLD_SIZE='2')
    args = ['mlsub', 'run', '--repo', REPO, '--branch', BRANCH, '--entry',
            'scripts/cloud/inspect_anvil2.py' if lr is None else 'scripts/cloud/run_anvil2.py',
            '--gpus', 'cpu' if lr is None else '2', '--image', 'efficient', '--no-pip', '--note', note]
    for name, value in env.items():
        args += ['--env', name + '=' + value]
    try:
        dry = command(args + ['--dry-run'])
    except subprocess.TimeoutExpired:
        # TimeoutExpired contains argv, including the W&B credential.
        raise RuntimeError('Cloud dry-run timed out before submission') from None
    if dry['returncode']:
        raise RuntimeError('Cloud dry-run failed: ' + dry['stderr'].replace(key, '[REDACTED]'))
    record = dict(state='submitting', timestamp=time.time(), lr=lr, code_revision=revision,
                  results_dir=env['RESULTS_DIR'], gpus=0 if lr is None else 2)
    with receipt.open('x') as stream:
        json.dump(record, stream)
    try:
        result = command(args, timeout=660)
        record.update({k: v.replace(key, '[REDACTED]') if isinstance(v, str) else v
                       for k, v in result.items()})
        match = re.search(r'lm-mpi-job-[0-9a-f-]+', record['stdout'])
        if match:
            record.update(state='submitted', job_id=match.group())
        else:
            record['state'] = 'ambiguous_submission'
    except subprocess.TimeoutExpired:
        record['state'] = 'ambiguous_submission'
    atomic_json(receipt, record)
    if not record.get('job_id'):
        raise RuntimeError('Submission not confirmed: ' + label)
    return record


def verify_preflight(revision: str) -> dict:
    receipt = submit(revision)
    status = json.loads(command(['mlsub', 'status', receipt['job_id']])['stdout'])
    if status.get('status', '').lower() != 'completed' or status.get('error_code') != 0:
        return dict(status=status, ready=False)
    log = command(['mlsub', 'logs', receipt['job_id'], '--tail', '300'])['stdout']
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
    if any(data[k] != v for k, v in dict(format='packed_fineweb_h200_v3', world_size=2,
                                        batch_size=16, sequence_length=1024).items()) or data['iterations'] < 39250:
        raise RuntimeError('Packed data mismatch')
    proof.update(code_revision=revision, ready=True, packed_data=data,
                 nfs2_write_usage=False, regression_passed=True)
    atomic_json(GATEWAY / 'preflight_verified.json', proof)
    return proof


def collect(lr: str) -> dict:
    receipt = json.loads((GATEWAY / ('lr' + lr + '.json')).read_text())
    record = dict(lr=lr, job_id=receipt.get('job_id'))
    if not record['job_id']:
        return record
    try:
        record['status'] = json.loads(command(['mlsub', 'status', record['job_id']])['stdout'])
        log = command(['mlsub', 'logs', record['job_id'], '--tail', '100'])['stdout']
        (GATEWAY / ('lr' + lr + '.log')).write_text(log)
        record['events'] = {}
        for line in log.splitlines():
            for marker in ('LIVE_STATUS=', 'LAUNCHER_DISK_GUARD=', 'RUN_VERIFIED='):
                if marker in line:
                    event = json.loads(line.split(marker, 1)[1])
                    if marker != 'LIVE_STATUS=' or event.get('phase') == '1xC':
                        record['events'][marker[:-1]] = event
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        record['scheduler_error'] = type(exc).__name__
    run_id = f'adapt-anvil2_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu-20261106'
    record['wandb_url'] = 'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + run_id
    try:
        auth = netrc.netrc().authenticators('wandb-radfan.ru')
        token = base64.b64encode(('api:' + auth[2]).encode()).decode()
        query = 'query { project(name:"fp8-pretrain", entityName:"andrey") { run(name:"' + run_id + '") { state config summaryMetrics } } }'
        request = urllib.request.Request('https://wandb-radfan.ru/graphql',
                    data=json.dumps(dict(query=query)).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + token})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        run = ((data.get('data') or {}).get('project') or {}).get('run')
        if run:
            config = json.loads(run['config'])
            record['wandb'] = dict(state=run['state'], summary=json.loads(run['summaryMetrics']),
                config={k: config.get(k, {}).get('value') for k in CONFIG_KEYS})
    except Exception as exc:
        record['wandb_error'] = type(exc).__name__
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'verify', 'launch', 'snapshot'))
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    GATEWAY.mkdir(parents=True, exist_ok=True)
    with (GATEWAY / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            value = submit(args.revision)
        elif args.action == 'verify':
            value = verify_preflight(args.revision)
        elif args.action == 'launch':
            with ThreadPoolExecutor(max_workers=6) as pool:
                value = list(pool.map(lambda lr: submit(args.revision, lr), LRS))
            atomic_json(GATEWAY / 'submissions.json', dict(timestamp=time.time(), trials=value))
        else:
            with ThreadPoolExecutor(max_workers=6) as pool:
                trials = list(pool.map(collect, LRS))
            value = dict(timestamp=time.time(), trials=trials)
            atomic_json(GATEWAY / 'snapshot.json', value)
        print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
