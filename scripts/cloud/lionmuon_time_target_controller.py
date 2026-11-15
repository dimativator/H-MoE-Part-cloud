"""Idempotent Cloud submission and read-only monitoring, credentials stay on gateway."""
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

BRANCH = 'codex/lionmuon-time-target-500m-20261115'
REPO = 'https://github.com/dimativator/H-MoE-Part-cloud.git'
ROOT = '/workspace-SR006.nfs3/dimativator/lionmuon-time-target-500m-20261115'
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
GATEWAY = Path('/home/dimativator/lionmuon-time-target-500m-20261115')
TRIALS = ['1xc', '2xc']
CONFIG_KEYS = ('opt','lr','weight_decay','iterations','seed','data_seed','dtype','model_parameter_dtype',
    'batch_size','acc_steps','sequence_length','no_local_save','latest_ckpt_interval','permanent_ckpt_interval',
    'inter_ckpts','n_layer','n_embd','n_head','scheduler','warmup_steps','wsd_fract_decay','wsd_final_lr_scale',
    'decay_type','beta1','beta2','fp8','fp8_optim','density','update_gap','non_proj_opt','apollo_proj','apollo_scale_type')


def atomic(path: Path, value: dict) -> None:
    temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(value,indent=2)); temporary.replace(path)


def command(args: list[str], timeout: int=40) -> dict:
    response=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    return dict(returncode=response.returncode,stdout=response.stdout,stderr=response.stderr)


def policy() -> None:
    if json.loads(Path('/etc/mlsub/policy.json').read_text())['logs_dir'] != LOGS:
        raise RuntimeError('NFS3 bootstrap logs required; NFS2 must not be used')


def submit(revision: str, stage: str, trial: str | None=None, label_override: str | None=None) -> dict:
    policy()
    advertised=command(['git','ls-remote',REPO,'refs/heads/'+BRANCH],30)
    if advertised['returncode'] or not advertised['stdout'].startswith(revision+'\t'):
        raise RuntimeError('Requested revision does not match published branch')
    label=label_override or trial or stage
    if stage=='inspect': label += '_' + time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    receipt=GATEWAY/(label+'.json')
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if not prior.get('job_id') or prior['code_revision']!=revision: raise RuntimeError('Reconcile existing submission '+label)
        return prior
    if stage in ('train', 'benchmark'):
        proof=json.loads((GATEWAY/'preflight_verified.json').read_text())
        if proof['code_revision']!=revision or not 0 <= time.time()-proof['timestamp'] <= 600:
            raise RuntimeError('Fresh preflight required')
        if not proof['wandb_ready'] or any(v <= 5_500_000_000 for v in proof['available_bytes'].values()):
            raise RuntimeError('Online W&B or disk reserve unavailable')
    inventory=command(['mlsub','list','--active'])
    note='lionmuon_ttt_'+label+'_20261115'
    if inventory['returncode'] or note in inventory['stdout']: raise RuntimeError('Duplicate or unavailable inventory')
    key=netrc.netrc().authenticators('wandb-radfan.ru')[2]
    env=dict(CODE_REVISION=revision,RESULTS_DIR=ROOT,WANDB_MODE='online',WANDB_API_KEY=key,
        VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS,CLOUD_DISABLE_CHECKPOINTS='1',CAMPAIGN_STAGE=stage)
    if trial: env['TRIAL']=trial
    if stage == 'train':
        benchmark=json.loads((GATEWAY/'benchmark.json').read_text())
        status=json.loads(command(['mlsub','status',benchmark['job_id']])['stdout'])
        if status.get('status') != 'completed' or status.get('error_code') != 0:
            raise RuntimeError('Paired timing must complete before training')
        log=command(['mlsub','logs',benchmark['job_id'],'--tail','400'])['stdout']
        if 'TIMING_PLAN=' not in log:
            raise RuntimeError('Verified timing plan missing')
    gpus=4 if trial or stage=='benchmark' else 0
    args=['mlsub','run','--repo',REPO,'--branch',BRANCH,'--entry','scripts/cloud/run_lionmuon_time_target.py',
        '--gpus',str(gpus) if gpus else 'cpu','--image','efficient','--no-pip','--note',note]
    for name,value in env.items(): args += ['--env',name+'='+value]
    try: dry=command(args+['--dry-run'],60)
    except subprocess.TimeoutExpired: raise RuntimeError('Dry run timed out, no submission') from None
    if dry['returncode']: raise RuntimeError('Dry run failed: '+dry['stderr'].replace(key,'[REDACTED]'))
    record=dict(state='submitting',timestamp=time.time(),trial=trial,stage=stage,code_revision=revision,results_dir=ROOT,gpus=gpus,note=note)
    with receipt.open('x') as stream: json.dump(record,stream)
    try:
        result=command(args,660)
        record.update({k:v.replace(key,'[REDACTED]') if isinstance(v,str) else v for k,v in result.items()})
        match=re.search('lm-mpi-job-[0-9a-f-]+',record['stdout'])
        record['state']='submitted' if match else 'ambiguous_submission'
        if match: record['job_id']=match.group()
    except subprocess.TimeoutExpired: record['state']='ambiguous_submission'
    atomic(receipt,record)
    if not record.get('job_id'): raise RuntimeError('Submission unconfirmed; never auto-retry '+label)
    return record


def verify(revision: str) -> dict:
    receipt=json.loads((GATEWAY/'preflight.json').read_text())
    state=json.loads(command(['mlsub','status',receipt['job_id']])['stdout'])
    if state.get('status')!='completed' or state.get('error_code')!=0: return dict(ready=False,status=state)
    log=command(['mlsub','logs',receipt['job_id'],'--tail','300'])['stdout']
    (GATEWAY/'preflight.log').write_text(log)
    events={}
    for line in log.splitlines():
        for marker in ('SPACE_SNAPSHOT=','PACKED_DATA=','WANDB_READY=','RUNTIME='):
            if marker in line: events[marker[:-1]]=json.loads(re.sub(r'\[\d+,\d+\]<(?:stdout|stderr)>:', '', line.split(marker,1)[1]))
    proof=events['SPACE_SNAPSHOT']
    proof.update(packed_data=events['PACKED_DATA'],wandb_ready=events['WANDB_READY']['ready'],runtime=events['RUNTIME'],nfs2_write_usage=False)
    if proof['code_revision']!=revision or not proof['wandb_ready']: raise RuntimeError('Preflight failed')
    atomic(GATEWAY/'preflight_verified.json',proof)
    return proof


def snapshot() -> dict:
    def collect(trial: str) -> dict:
        path=GATEWAY/(trial+'.json')
        if not path.exists(): return dict(trial=trial,status='not_submitted')
        receipt=json.loads(path.read_text()); record=dict(trial=trial,job_id=receipt.get('job_id'),gpus=receipt['gpus'])
        if record['job_id']:
            try: record['status']=json.loads(command(['mlsub','status',record['job_id']],30)['stdout'])
            except (ValueError,OSError,subprocess.TimeoutExpired) as exc: record['scheduler_error']=type(exc).__name__
        record['wandb_url']='https://wandb-radfan.ru/andrey/fp8-pretrain/runs/lion-ttt-500m-'+trial+'-20261115'
        return record
    with ThreadPoolExecutor(max_workers=8) as pool: records=list(pool.map(collect,TRIALS))
    try:
        key=netrc.netrc().authenticators('wandb-radfan.ru')[2]
        token=base64.b64encode(('api:'+key).encode()).decode()
        fields=['r'+str(i)+':run(name:"lion-ttt-500m-'+t+'-20261115") { state config summaryMetrics }' for i,t in enumerate(TRIALS)]
        query='query { project(name:"fp8-pretrain",entityName:"andrey") { '+' '.join(fields)+' } }'
        request=urllib.request.Request('https://wandb-radfan.ru/graphql',data=json.dumps(dict(query=query)).encode(),
            headers={'Content-Type':'application/json','Authorization':'Basic '+token})
        with urllib.request.urlopen(request,timeout=25) as response: data=json.load(response)
        project=(data.get('data') or {}).get('project') or {}
        for i,record in enumerate(records):
            run=project.get('r'+str(i))
            if run:
                config=json.loads(run['config']); record['wandb']=dict(state=run['state'],summary=json.loads(run['summaryMetrics']),
                    config={k:config.get(k,{}).get('value') for k in CONFIG_KEYS})
    except (OSError,ValueError,KeyError,TypeError) as exc:
        for record in records: record['wandb_error']=type(exc).__name__
    value=dict(timestamp=time.time(),trials=records)
    atomic(GATEWAY/'snapshot.json',value)
    atomic(GATEWAY/('snapshot_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'),value)
    return value


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('preflight','verify','benchmark','launch','snapshot','inspect'))
    parser.add_argument('--revision',required=True); args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.revision): raise ValueError('Invalid revision')
    GATEWAY.mkdir(exist_ok=True)
    with (GATEWAY/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='preflight':
            pending=GATEWAY/'preflight.json'
            if pending.exists():
                previous=json.loads(pending.read_text())
                if not previous.get('job_id'):
                    raise RuntimeError('Reconcile pending preflight before another submission')
                state=json.loads(command(['mlsub','status',previous['job_id']])['stdout']).get('status')
                if state not in ('completed','failed','cancelled','canceled','killed'):
                    print(json.dumps(previous),flush=True); return
            label='preflight_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
            value=submit(args.revision,'preflight',label_override=label)
            atomic(GATEWAY/'preflight.json',value)
        elif args.action=='verify': value=verify(args.revision)
        elif args.action=='benchmark': value=submit(args.revision,'benchmark')
        elif args.action=='inspect': value=submit(args.revision,'inspect')
        elif args.action=='snapshot': value=snapshot()
        else:
            with ThreadPoolExecutor(max_workers=8) as pool: value=list(pool.map(lambda t:submit(args.revision,'train',t),TRIALS))
            atomic(GATEWAY/'submissions.json',dict(timestamp=time.time(),trials=value))
        print(json.dumps(value),flush=True)


if __name__=='__main__': main()
