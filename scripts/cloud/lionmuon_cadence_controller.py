"""Receipt-locked submissions and read-only W&B observations for this campaign."""
import argparse
import base64
import fcntl
import json
import netrc
from pathlib import Path
import re
import subprocess
import time
import urllib.request

BASE = Path('/home/dimativator/lionmuon-cadence-500m-20261123')
REPO = 'https://github.com/dimativator/H-MoE-Part-cloud.git'
BRANCH = 'codex/lionmuon-cadence-500m-20261123'
LOGS = '/home/jovyan/mlsub-logs'
ROOT = '/workspace-SR006.nfs3/dimativator/lionmuon-cadence-500m-20261123'


TRIALS = ('regular', 'constant_random', 'muon_ema', 'linear_zero', 'half_cosine', 'stagger_uniform', 'stagger_early', 'stagger_middle', 'stagger_late')


def key() -> str:
    return netrc.netrc().authenticators('wandb-radfan.ru')[2]


def query(text: str) -> dict:
    req = urllib.request.Request('https://wandb-radfan.ru/graphql',
        data=json.dumps(dict(query=text)).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Basic ' + base64.b64encode(('api:' + key()).encode()).decode()})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.load(response)
    if data.get('errors'):
        raise RuntimeError(str(data['errors']))
    return data['data']['project']


def snapshot(preflight_only=False) -> dict:
    names = ['lion-cadence-500m-preflight-monitor-20261123', 'lion-norm-500m-probability_down-20261117']
    for rank in TRIALS:
        names += [f'lion-cadence-500m-{rank}-20261123', f'lion-cadence-500m-{rank}-monitor-20261123']
    if preflight_only:
        names = names[:1]
    fields = [f'r{i}:run(name:{json.dumps(name)}) {{ state config summaryMetrics history(samples: 10000) }}' for i, name in enumerate(names)]
    data = query('query { project(name:"fp8-pretrain",entityName:"andrey") { ' + ' '.join(fields) + ' } }')
    result = dict(timestamp=time.time(), runs={}, receipts={})
    for i, name in enumerate(names):
        value = data.get('r'+str(i))
        if value:
            result['runs'][name] = dict(state=value['state'], summary=json.loads(value['summaryMetrics']),
                config=json.loads(value['config']), history=[json.loads(x) for x in value['history']])
    for path in BASE.glob('receipt_*.json'):
        result['receipts'][path.stem] = json.loads(path.read_text())

    return result


def submit(role: str, revision: str) -> dict:
    if role not in ('preflight', *TRIALS):
        raise ValueError('Unknown trial')
    receipt = BASE / ('receipt_'+role+'.json')
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if value.get('job_id'):
            return value
        raise RuntimeError('Ambiguous prior submission, no automatic retry')
    if json.loads(Path('/etc/mlsub/policy.json').read_text())['logs_dir'] != LOGS:
        raise RuntimeError('Bootstrap logs policy changed')
    if role == 'preflight':
        authorization = json.loads((BASE / 'bootstrap_authorization.json').read_text())
        if authorization.get('code_revision') != revision or authorization.get('user_instruction') != 'just run. without checkpoints':
            raise RuntimeError('Explicit campaign bootstrap authorization required')
        proof = None
    else:
        current = snapshot(preflight_only=True)
        preflight = current['runs'].get('lion-cadence-500m-preflight-monitor-20261123', {})
        status = preflight.get('summary', {}).get('worker_status', {})
        if preflight.get('state') != 'finished' or status.get('status') != 'complete' or status.get('code_revision') != revision:
            raise RuntimeError('Successful same-revision preflight required')
        proof = status['space']
    if not proof and role != 'preflight':
        raise RuntimeError('Missing storage proof')
    if proof:
        required_paths = {'/workspace-SR006.nfs3', LOGS, '/home/jovyan', '/tmp', '/'}
        if not required_paths.issubset(proof['available_bytes']) or 'nfs2' not in proof:
            raise RuntimeError('Incomplete storage telemetry')
        if not 0 <= time.time()-proof['timestamp'] < 600 or any(v <= 5_500_000_000 for v in proof['available_bytes'].values()):
            raise RuntimeError('Fresh sufficient worker storage proof required')
    remote = subprocess.run(['git','ls-remote',REPO,'refs/heads/'+BRANCH],capture_output=True,text=True,timeout=30)
    if remote.returncode or not remote.stdout.startswith(revision+'\t'):
        raise RuntimeError('Published code revision mismatch')
    env = dict(CODE_REVISION=revision, CAMPAIGN_STAGE='preflight' if role=='preflight' else 'train',
        VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS, CLOUD_DISABLE_CHECKPOINTS='1',
        WANDB_API_KEY=key(), WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_MODE='online')
    if role != 'preflight':
        env['TRIAL'] = role
    command = ['mlsub','run','--repo',REPO,'--branch',BRANCH,'--entry','scripts/cloud/run_lionmuon_cadence.py',
        '--gpus','cpu' if role=='preflight' else '4','--image','efficient','--no-pip',
        '--note','lion_norm_500m_'+role+'_20261123']
    for name, value in env.items():
        command += ['--env',name+'='+value]
    dry = subprocess.run(command+['--dry-run'],capture_output=True,text=True,timeout=60)
    if dry.returncode:
        raise RuntimeError('Dry-run rejected: '+dry.stderr.replace(key(),'[REDACTED]'))
    record = dict(state='submitting', role=role, timestamp=time.time(), code_revision=revision,
        gpus=0 if role=='preflight' else 4, results_root=ROOT, storage_proof=proof,
        wandb_run_id='lion-cadence-500m-'+('preflight-monitor' if role=='preflight' else role)+'-20261123')
    with receipt.open('x') as stream:
        json.dump(record,stream)
    try:
        result = subprocess.run(command,capture_output=True,text=True,timeout=660)
        record.update(returncode=result.returncode, stdout=result.stdout.replace(key(),'[REDACTED]'),
                      stderr=result.stderr.replace(key(),'[REDACTED]'))
        match = re.search(r'lm-mpi-job-[0-9a-f-]+', record['stdout'])
        record['state'] = 'submitted' if match else 'ambiguous_submission'
        if match:
            record['job_id'] = match.group()
    except subprocess.TimeoutExpired:
        record['state'] = 'ambiguous_submission'
    receipt.write_text(json.dumps(record,indent=2))
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('snapshot','preflight',*TRIALS))
    p.add_argument('--revision')
    a=p.parse_args()
    BASE.mkdir(exist_ok=True)
    with (BASE/(a.action+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=snapshot() if a.action=='snapshot' else submit(a.action,a.revision)
        if a.action=='snapshot':
            (BASE/'snapshot.json').write_text(json.dumps(value))
            value=dict(timestamp=value['timestamp'],runs={n:dict(state=v['state'],
                iter=v['summary'].get('iter'),loss=v['summary'].get('val/loss'),
                worker_status=v['summary'].get('worker_status')) for n,v in value['runs'].items()})
        print(json.dumps(value),flush=True)
