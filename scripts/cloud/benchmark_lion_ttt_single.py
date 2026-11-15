"""Find maximum fitting microbatch, then time both optimizers at effective batch128."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_lionmuon_time_target as common
sys.path[:0] = [str(common.REPO / 'src'), str(common.REPO)]

ROOT = common.ROOT / 'single_gpu_timing'
OPTIMIZERS = ('muon_reference','lion_muon')


def atomic(path: Path, value: dict) -> None:
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def export(run, status: str, **fields: object) -> None:
    value=dict(status=status,timestamp=time.time(),code_revision=os.environ['CODE_REVISION'],
               measurement_gpus=1,global_batch=128,**fields)
    atomic(ROOT/'status.json',value);run.summary['benchmark_status']=value
    common.emit('SINGLE_GPU_STATUS',value)


def command(name: str, optimizer: str, batch: int, timing: bool) -> list[str]:
    args=common.arguments(name,optimizer,75457,7546,True)
    del args[args.index('--distributed-backend'):args.index('--distributed-backend')+2]
    for flag,value in (('--batch-size',batch),('--acc-steps',128//batch),('--fineweb-replay-world-size',2),
                       ('--early-stop-iteration',240 if timing else 6)):
        args[args.index(flag)+1]=str(value)
    args[0]='scripts/cloud/lion_ttt_single_train.py'
    args[args.index('--metrics-jsonl')+1]=str(ROOT/name/'metrics.jsonl')
    return [sys.executable,*args]


def execute(name: str, optimizer: str, batch: int, timing: bool) -> dict:
    path=ROOT/'logs'/(name+'.log');metrics=ROOT/name/'metrics.jsonl'
    if path.exists() or metrics.exists():
        raise RuntimeError('Refusing duplicate benchmark trial: '+name)
    env={**os.environ,'WANDB_MODE':'disabled','LION_TTT_BENCHMARK':'1' if timing else '0'}
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    argv=command(name,optimizer,batch,timing)
    with path.open('x') as stream:
        process=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        common.emit('BENCH_CHILD',dict(name=name,pid=process.pid,command=argv))
        last=0
        try:
            while process.poll() is None:
                if time.monotonic()-last>=30:
                    last=time.monotonic();space=common.space();atomic(ROOT/'disk_status.json',space)
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    text=path.read_text()
    result=dict(name=name,optimizer=optimizer,microbatch=batch,accumulation=128//batch,returncode=process.returncode)
    if process.returncode:
        if 'CUDA out of memory' in text or 'torch.OutOfMemoryError' in text:
            return dict(**result,status='oom',error_tail=text[-2500:])
        raise RuntimeError('Non-OOM trial failure: '+name+'\n'+text[-6000:])
    if 'EARLY_STOP_ITERATION='+str(240 if timing else 6) not in text:
        raise RuntimeError('Missing exact completion marker: '+name)
    rows=[json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    if timing:
        rows=[r for r in rows if r['event']=='ttt_benchmark']
        if len(rows)!=10 or [r['iter'] for r in rows]!=list(range(60,241,20)):
            raise RuntimeError('Incomplete timing period coverage')
        result.update(seconds_per_step=sum(r['seconds'] for r in rows)/200,blocks=rows,
                      peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
    return dict(**result,status='complete')


def main() -> None:
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s))
    signal.signal(signal.SIGINT,lambda s,f:sys.exit(128+s))
    common.setup()
    for child in ('logs','wandb'):(ROOT/child).mkdir(parents=True,exist_ok=True)
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS')!='1':raise RuntimeError('Checkpoints must be disabled')
    import wandb
    stage=os.environ.get('SINGLE_STAGE','benchmark')
    run=wandb.init(entity='andrey',project='fp8-pretrain',id=os.environ.get('SINGLE_WANDB_RUN_ID','lion-ttt-single-'+stage+'-20261115'),
        name='LionMuon single GPU batch128 '+stage,dir=str(ROOT/'wandb'),resume='never',
        settings=wandb.Settings(init_timeout=60))
    try:
        space=common.space()
        if stage=='preflight':
            import torch
            from third_party.lion_muon.lion_muon import LionMuon
            from third_party.lion_muon.muon import Muon
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_lion_ttt_single.py','-v'],check=True)
            from data import fineweb_packed as packed
            import hashlib
            meta=json.loads((common.DATA/'packed_metadata.json').read_text())
            proofs=[]
            proof_path=ROOT/'data_proof.json'
            prior=json.loads(proof_path.read_text()) if proof_path.exists() else {}
            for source in meta['ranks']:
                segment=source['segments'][0];path=common.DATA/segment['file'];stat=path.stat()
                previous=next((p for p in prior.get('sources',[]) if p['rank']==source['rank']),{})
                if not (previous.get('sha256')==segment['sha256'] and previous.get('mtime_ns')==stat.st_mtime_ns and previous.get('bytes')==stat.st_size):
                    if packed._sha256_file(path)!=segment['sha256']:raise RuntimeError('Source segment hash mismatch')
                with path.open('rb') as stream:prefix=stream.read(15360*1025*2)
                if len(prefix)!=15360*1025*2:raise RuntimeError('Timing prefix too short')
                proofs.append(dict(rank=source['rank'],path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    sha256=segment['sha256'],timing_prefix_sha256=hashlib.sha256(prefix).hexdigest(),timing_prefix_blocks=15360))
            atomic(proof_path,dict(code_revision=os.environ['CODE_REVISION'],sources=proofs))
            space=common.space()
            export(run,'preflight_complete',space=space,data_proof=proofs,torch=torch.__version__,optimizers=[LionMuon.__name__,Muon.__name__]);return
        import torch
        if int(os.environ.get('OMPI_COMM_WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:
            raise RuntimeError('Exactly one allocated GPU required')
        device=dict(name=torch.cuda.get_device_name(0),total_memory=torch.cuda.get_device_properties(0).total_memory)
        probes=[];timing={}
        for opt in OPTIMIZERS:
            selected=None
            for batch in (128,64,32,16,8,4,2,1):
                probe=execute(f'probe_{opt}_b{batch}',opt,batch,False);probes.append(probe)
                export(run,'probing',probes=probes,device=device)
                if probe['status']=='complete':selected=batch;break
            if selected is None:raise RuntimeError('No fitting microbatch: '+opt)
            result=execute('timing_'+opt,opt,selected,True)
            if result['status']!='complete':raise RuntimeError('Chosen batch did not sustain full timing: '+opt)
            timing[opt]=result;export(run,'timing',probes=probes,timing=timing,device=device)
        muon=timing['muon_reference']['seconds_per_step'];lion=timing['lion_muon']['seconds_per_step']
        targets={}
        for name,n in common.BASELINES.items():
            target=math.floor(n*muon/(1.05*lion))
            targets[name]=dict(baseline_steps=n,target_steps=target,extra_stable_steps=target-n,
                decay_steps={'1xc':7546,'2xc':15091}[name],consumed_tokens=target*131072,
                muon_seconds=n*muon,lion_seconds=target*lion,predicted_speedup=n*muon/(target*lion))
        plan=dict(code_revision=os.environ['CODE_REVISION'],measurement_gpus=1,training_gpus_per_run=4,
            global_batch=128,device=device,timing=timing,targets=targets,probes=probes,
            feasible=all(r['extra_stable_steps']>0 for r in targets.values()),
            data_extension_required=targets['2xc']['target_steps']>157100,
            method='Maximum fitting divisor microbatch per optimizer;6 probe steps including p5 Muon step;40 warmup +200 timed steps in20-step periods; one GPU; no distributed collectives.')
        atomic(ROOT/'timing_plan.json',plan)
        space=common.space();run.summary['timing_plan']=plan
        export(run,'complete',timing_plan=plan,space=space)
        common.emit('TIMING_PLAN',plan)
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
        export(run,'failed',error=str(exc));raise
    finally:run.finish()


if __name__=='__main__':main()
