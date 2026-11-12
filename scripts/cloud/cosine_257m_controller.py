"""Gateway-only, receipt-protected 16-trial cosine comparison. No follow-up runs."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import netrc
from pathlib import Path
import re
import subprocess
import time
import urllib.request

BRANCH = 'codex/cosine-257m-four-optimizers-20261112'
REPO = 'https://github.com/dimativator/H-MoE-Part-cloud.git'
ROOT = '/workspace-SR006.nfs3/dimativator/cosine-257m-four-optimizers-20261112'
GATEWAY = Path('/home/dimativator/cosine-257m-four-optimizers-20261112')
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
LRS = ('2e-3', '1e-3', '5e-4', '1e-4')
OPTS = ('soft_lion_muon', 'lion_muon', 'muon_reference', 'softmuon')
GRID = tuple((opt, lr) for opt in OPTS for lr in LRS)
TERMINAL = ('completed', 'failed', 'cancelled', 'canceled', 'killed', 'stopped')


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(args: list[str], timeout: int = 45) -> dict:
    try:
        r = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired:
        return dict(error='timeout')


def check_policy() -> None:
    if json.loads(Path('/etc/mlsub/policy.json').read_text())['logs_dir'] != LOGS:
        raise RuntimeError('Bootstrap logs must use verified NFS3, never NFS2')


def credential() -> str:
    auth = netrc.netrc().authenticators('wandb-radfan.ru')
    if not auth:
        raise RuntimeError('W&B credentials unavailable')
    return auth[2]


def run_name(opt: str, lr: str) -> str:
    return f'adapt-{opt}_257m_bf16_lr{lr}_wd1e-1_cosine_1xC_1gpu-20261112'


def label_for(opt: str, lr: str) -> str:
    return f'{opt}_lr{lr}'


def inventory() -> str:
    result = command(['mlsub', 'list', '--active'], timeout=60)
    if result.get('returncode') != 0:
        raise ConnectionError('Active scheduler inventory unavailable')
    return result['stdout']


def submit(label: str, revision: str, *, opt: str | None = None, lr: str | None = None,
           inspection: bool = False) -> dict:
    check_policy()
    path = GATEWAY / (label + '.json')
    if path.exists():
        previous = json.loads(path.read_text())
        if not previous.get('job_id'):
            raise RuntimeError('Ambiguous receipt requires reconciliation: ' + label)
        return previous
    preflight = opt is None and not inspection
    if opt is not None:
        if (opt, lr) not in GRID:
            raise ValueError('Outside authorized optimizer/LR grid')
        proof = json.loads((GATEWAY / 'preflight_verified.json').read_text())
        if proof['code_revision'] != revision or not 0 <= time.time() - proof['timestamp'] <= 600:
            raise RuntimeError('Fresh source-matched storage preflight required')
        if any(x <= 5_500_000_000 for x in proof['available_bytes'].values()):
            raise RuntimeError('Insufficient storage reserve before launch')
    key = credential() if not inspection else None
    env = dict(CODE_REVISION=revision, RESULTS_DIR=ROOT,
               VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS, CLOUD_DISABLE_CHECKPOINTS='1')
    if not inspection:
        env.update(WANDB_MODE='online', WANDB_API_KEY=key)
    if preflight:
        env.update(COSINE_PREFLIGHT='1', CHECK_PACKED_DATA='1', CHECK_WANDB='1')
    if opt is not None:
        env.update(OPTIMIZER=opt, LR=lr, CLOUD_TRAIN_WORLD_SIZE='1',
                   RESULTS_DIR=f'{ROOT}/{opt}/lr{lr}')
    args = ['mlsub', 'run', '--repo', REPO, '--branch', BRANCH,
            '--entry', 'scripts/cloud/inspect_cosine_257m.py' if opt is None else 'scripts/cloud/run_cosine_257m.py',
            '--gpus', 'cpu' if opt is None else '1', '--image', 'efficient', '--no-pip',
            '--note', 'cosine257m_' + label + '_20261112']
    for name, value in env.items():
        args += ['--env', name + '=' + value]
    dry = command(args + ['--dry-run'], timeout=60)
    if dry.get('returncode') != 0:
        raise RuntimeError('Dry-run failed before submission: ' + label)
    record = dict(label=label, optimizer=opt, lr=lr, code_revision=revision,
                  results_dir=env['RESULTS_DIR'], timestamp=time.time(), state='submitting')
    with path.open('x') as stream:
        json.dump(record, stream)
    result = command(args, timeout=660)
    if key:
        result = {k: v.replace(key, '[REDACTED]') if isinstance(v, str) else v for k, v in result.items()}
    record.update(result)
    match = re.search(r'lm-mpi-job-[0-9a-f-]+', result.get('stdout', ''))
    record['state'] = 'submitted' if match else 'ambiguous_submission'
    if match:
        record['job_id'] = match.group()
    atomic_json(path, record)
    if not match:
        raise RuntimeError('Submission not confirmed; never blindly retry: ' + label)
    return record


def decode_log_json(payload: str) -> dict:
    # MPI can insert a new stream prefix every 4 KiB, even inside scalars.
    return json.loads(re.sub(r'\[\d+,\d+\]<(?:stdout|stderr)>:', '', payload))


def observe_inspection(label: str) -> dict:
    receipt = json.loads((GATEWAY / (label + '.json')).read_text())
    raw = command(['mlsub', 'status', receipt['job_id']])
    if raw.get('returncode') != 0:
        return dict(ready=False, status_error=raw)
    status = json.loads(raw['stdout'])
    state = status.get('status', '').lower()
    if state in TERMINAL and (state != 'completed' or status.get('error_code') != 0):
        raise RuntimeError('CPU inspection failed: ' + str(status))
    if state != 'completed':
        return dict(ready=False, status=status)
    raw = command(['mlsub', 'logs', receipt['job_id'], '--tail', '8000'], timeout=60)
    if raw.get('returncode') != 0:
        return dict(ready=False, log_error=raw.get('error', 'unavailable'))
    text = raw.get('stdout', '')
    (GATEWAY / (label + '.log')).write_text(text)
    data = dict(ready=True, timestamp=time.time(), status=status, job_id=receipt['job_id'],
                workers=[], progress=[], finals=[], curves=[], nfs2=[],
                regression_passed='COSINE_REGRESSION_OK' in text)
    multiple = dict(WORKER_DISK='workers', PROGRESS='progress', FINAL='finals',
                    CURVE='curves', NFS2_READ_ONLY='nfs2')
    single = dict(SPACE_SNAPSHOT='space', PACKED_DATA='packed_data',
                  WANDB_READY='wandb_ready', CHECKPOINT_AUDIT='checkpoints')
    for line in text.splitlines():
        for marker in (*multiple, *single):
            if marker + '=' not in line:
                continue
            value = decode_log_json(line.split(marker + '=', 1)[1])
            if marker in multiple:
                data[multiple[marker]].append(value)
            else:
                data[single[marker]] = value
    if 'space' not in data:
        return dict(ready=False, status=status, reason='Waiting for complete log export')
    atomic_json(GATEWAY / (label + '_observed.json'), data)
    atomic_json(GATEWAY / 'latest_inspection.json', data)
    return data


def verify_preflight(revision: str) -> dict:
    data = observe_inspection('preflight')
    if not data['ready']:
        return data
    if not data['regression_passed'] or not data.get('wandb_ready', {}).get('ready'):
        raise RuntimeError('Preflight source/scheduler/numerical/W&B tests did not pass')
    metadata = data.get('packed_data', {})
    expected = dict(format='packed_fineweb_h200_v3', world_size=2,
                    batch_size=16, sequence_length=1024)
    if any(metadata.get(k) != v for k, v in expected.items()) or metadata.get('iterations', 0) < 39250:
        raise RuntimeError('Dataset mismatch')
    proof = data['space']
    if proof['logs_dir'] != LOGS or any(v <= 5_500_000_000 for v in proof['available_bytes'].values()):
        raise RuntimeError('Storage reserve unavailable before launch')
    proof.update(code_revision=revision, regression_passed=True, nfs2_write_usage=False,
                 packed_data=metadata)
    atomic_json(GATEWAY / 'preflight_verified.json', proof)
    return dict(ready=True, proof=proof)


def wandb_batch() -> dict:
    aliases = {f'r{i}': (opt, lr) for i, (opt, lr) in enumerate(GRID)}
    fields = ' '.join(f'{alias}: run(name:"{run_name(opt, lr)}") {{ name state config summaryMetrics }}'
                      for alias, (opt, lr) in aliases.items())
    query = 'query { project(name:"fp8-pretrain", entityName:"andrey") { ' + fields + ' } }'
    token = base64.b64encode(('api:' + credential()).encode()).decode()
    request = urllib.request.Request('https://wandb-radfan.ru/graphql',
        data=json.dumps(dict(query=query)).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + token})
    with urllib.request.urlopen(request, timeout=40) as response:
        data = json.load(response)
    if data.get('errors'):
        raise RuntimeError('Batched W&B query failed')
    result = {}
    for alias, pair in aliases.items():
        run = (data.get('data', {}).get('project') or {}).get(alias)
        if run:
            run['summary'] = json.loads(run.pop('summaryMetrics'))
            run['config'] = {k: v.get('value') for k, v in json.loads(run['config']).items()}
            run['url'] = 'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + run_name(*pair)
            result[pair] = run
    return result


def snapshot() -> dict:
    records = []
    for opt, lr in GRID:
        path = GATEWAY / (label_for(opt, lr) + '.json')
        if path.exists():
            r = json.loads(path.read_text())
            records.append(dict(optimizer=opt, lr=lr, job_id=r.get('job_id')))
    def status(record: dict) -> dict:
        if not record['job_id']:
            return {**record, 'status_error': 'Ambiguous submission'}
        raw = command(['mlsub', 'status', record['job_id']])
        try:
            return {**record, 'status': json.loads(raw['stdout'])}
        except (KeyError, ValueError):
            return {**record, 'status_error': raw.get('error', 'unavailable')}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(status, records))
    try:
        runs = wandb_batch()
        for record in records:
            record['wandb'] = runs.get((record['optimizer'], record['lr']), {})
        error = None
    except Exception as exc:
        error = type(exc).__name__
    result = dict(timestamp=time.time(), trials=records, wandb_error=error)
    atomic_json(GATEWAY / 'latest_snapshot.json', result)
    return result


def validate_final(record: dict) -> float:
    status, run = record.get('status', {}), record.get('wandb', {})
    if status.get('status', '').lower() != 'completed' or status.get('error_code') != 0 or run.get('state') != 'finished':
        raise ValueError('Scheduler/W&B have not confirmed successful completion')
    expected = dict(opt=record['optimizer'], lr=float(record['lr']), weight_decay=0.1,
        iterations=39250, scheduler='cos_warmup_zero', warmup_steps=2000,
        seed=0, data_seed=1337, dtype='bfloat16', model_parameter_dtype='float32',
        batch_size=32, acc_steps=4, sequence_length=1024, n_layer=12, n_embd=1024, n_head=8,
        no_local_save=True, latest_ckpt_interval=0, permanent_ckpt_interval=0, inter_ckpts=[],
        fineweb_replay_world_size=2, fineweb_replay_layout='concat')
    if record['optimizer'] in ('lion_muon', 'muon_reference'):
        expected['reference_aux_lr'] = 0.001
    elif record['optimizer'] == 'softmuon':
        expected.update(softmuon_aux_lr=0.001, softmuon_sign_fraction=0.9,
                        softmuon_eps=1e-4, softmuon_newton_iters=10)
    else:
        expected.update(soft_lion_muon_aux_lr=0.001, soft_lion_muon_period=5,
            soft_lion_muon_beta1=0.95, soft_lion_muon_beta2=0.99,
            soft_lion_muon_lr_ratio=0.01, soft_lion_muon_start_fraction=0.9,
            soft_lion_muon_eps=1e-4, soft_lion_muon_ns_steps=6, soft_lion_muon_soft_ns_steps=30)
    mismatches = {k: (run.get('config', {}).get(k), v) for k, v in expected.items()
                  if run.get('config', {}).get(k) != v}
    if mismatches:
        raise RuntimeError('Configuration mismatch: ' + str(mismatches))
    summary = run.get('summary', {})
    loss = summary.get('final-val/loss')
    if summary.get('iter') != 39250 or summary.get('consumed_tokens') != 5144576000:
        raise RuntimeError('Incomplete final budget')
    if not isinstance(loss, (float, int)) or not math.isfinite(loss):
        raise RuntimeError('Missing/nonfinite final loss')
    return float(loss)


def launch_all(revision: str) -> list[dict]:
    check_policy()
    active = inventory()
    for opt, lr in GRID:
        if 'cosine257m_' + label_for(opt, lr) + '_20261112' in active and not (GATEWAY / (label_for(opt, lr) + '.json')).exists():
            raise RuntimeError('Existing matching job without receipt, reconcile before submitting')
    def launch(pair: tuple[str, str]) -> dict:
        return submit(label_for(*pair), revision, opt=pair[0], lr=pair[1])
    with ThreadPoolExecutor(max_workers=16) as pool:
        receipts = list(pool.map(launch, GRID))
    atomic_json(GATEWAY / 'submissions.json', dict(timestamp=time.time(), trials=receipts))
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'verify', 'launch', 'snapshot', 'inspect', 'inspection', 'run'))
    parser.add_argument('--revision', required=True)
    parser.add_argument('--label')
    args = parser.parse_args()
    GATEWAY.mkdir(parents=True, exist_ok=True)
    if args.action == 'snapshot':
        result = snapshot()
    elif args.action == 'inspection':
        result = observe_inspection(args.label)
    else:
        with (GATEWAY / 'controller.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action in ('preflight', 'inspect'):
                check_policy()
                active = inventory()
                if any('entry:scripts/cloud/inspect_cosine_257m.py' in line for line in active.splitlines()):
                    raise RuntimeError('A CPU inspector is already active; reuse its receipt')
                result = submit('preflight' if args.action == 'preflight' else args.label,
                                args.revision, inspection=args.action == 'inspect')
            elif args.action == 'verify':
                result = verify_preflight(args.revision)
            else:
                result = launch_all(args.revision)
                if args.action == 'run':
                    # Monitoring inspections use the same receipt lock. Hold it only
                    # for submissions, never for the multi-hour observation loop.
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    while True:
                        data = snapshot()
                        states = [r.get('status', {}).get('status', '').lower() for r in data['trials']]
                        all_terminal = len(states) == 16 and all(s in TERMINAL for s in states)
                        winners, failures = {}, []
                        for opt in OPTS:
                            losses = {}
                            for r in data['trials']:
                                if r['optimizer'] != opt:
                                    continue
                                try:
                                    losses[r['lr']] = validate_final(r)
                                except (ValueError, RuntimeError) as exc:
                                    if r.get('status', {}).get('status', '').lower() in TERMINAL:
                                        failures.append(dict(optimizer=opt, lr=r['lr'], reason=str(exc)))
                            if len(losses) == 4:
                                lr = min(LRS, key=lambda x: (losses[x], float(x)))
                                winners[opt] = dict(lr=lr, final_val_loss=losses[lr], all_losses=losses)
                        complete = len(winners) == 4
                        state = dict(timestamp=time.time(), code_revision=args.revision,
                            stage='complete' if complete else ('terminal_with_failures' if all_terminal else 'running'),
                            winners=winners, failures=failures)
                        atomic_json(GATEWAY / 'controller_state.json', state)
                        failed_jobs = any(s in TERMINAL and s != 'completed' for s in states)
                        if complete or (all_terminal and failed_jobs):
                            atomic_json(GATEWAY / 'final_results.json', dict(**state, snapshot=data))
                            result = state
                            break
                        time.sleep(300)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        GATEWAY.mkdir(parents=True, exist_ok=True)
        atomic_json(GATEWAY / 'controller_failure.json',
                    dict(timestamp=time.time(), error=type(exc).__name__, message=str(exc)))
        raise
