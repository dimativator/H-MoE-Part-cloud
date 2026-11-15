"""Diagnostic one-GPU batch32 timing, without gradient accumulation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import benchmark_lion_ttt_single as original
import run_lionmuon_time_target as common

ROOT=common.ROOT/'batch32_timing'
DATA_REVISION='3e7604f1210d42eb1c83f34af24b9f542fee81c8'


def command(optimizer: str)->list[str]:
    args=original.command('batch32_'+optimizer,optimizer,32,True)
    args[args.index('--acc-steps')+1]='1'
    args[args.index('--metrics-jsonl')+1]=str(ROOT/optimizer/'metrics.jsonl')
    return args


def export(run,status:str,**fields:object)->None:
    value=dict(status=status,timestamp=time.time(),code_revision=os.environ['CODE_REVISION'],measurement_gpus=1,
               microbatch=32,accumulation=1,global_batch=32,diagnostic_only=True,**fields)
    (ROOT/'status.json').write_text(json.dumps(value,indent=2));run.summary['batch32_benchmark']=value
    common.emit('BATCH32_STATUS',value)


def measure(optimizer:str,run,timing:dict)->dict:
    log=ROOT/(optimizer+'.log');metrics=ROOT/optimizer/'metrics.jsonl'
    if log.exists() or metrics.exists():raise RuntimeError('Duplicate diagnostic refused')
    env={**os.environ,'WANDB_MODE':'disabled','LION_TTT_BENCHMARK':'1',
         'LION_TTT_EFFECTIVE_BATCH':'32','DATA_PROOF_REVISION':DATA_REVISION}
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT','LION_TTT_PHASE_PROFILE'):env.pop(key,None)
    with log.open('x') as output:
        child=subprocess.Popen(command(optimizer),env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        last=0.0
        try:
            while child.poll() is None:
                if time.monotonic()-last>=30:
                    last=time.monotonic();export(run,'running',current_optimizer=optimizer,child_pid=child.pid,timing=timing,space=common.space())
                time.sleep(1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if child.returncode:raise RuntimeError('Batch32 diagnostic failed: '+log.read_text()[-5000:])
    if 'EARLY_STOP_ITERATION=240' not in log.read_text():raise RuntimeError('Exact completion marker missing')
    rows=[json.loads(line) for line in metrics.read_text().splitlines()]
    rows=[r for r in rows if r['event']=='ttt_benchmark']
    if [r['iter'] for r in rows]!=list(range(60,241,20)) or any(r['microbatch']!=32 or r['accumulation']!=1 or r['world_size']!=1 for r in rows):
        raise RuntimeError('Wrong batch32 protocol or incomplete timing window')
    return dict(microbatch=32,accumulation=1,global_batch=32,seconds_per_step=sum(r['seconds'] for r in rows)/200,blocks=rows,
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))


def main()->None:
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));signal.signal(signal.SIGINT,lambda s,f:sys.exit(128+s))
    common.setup();ROOT.mkdir(exist_ok=True)
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS')!='1':raise RuntimeError('Checkpoints must be disabled')
    import wandb
    stage=os.environ.get('BATCH32_STAGE','benchmark')
    run=wandb.init(entity='andrey',project='fp8-pretrain',id='lion-ttt-batch32-'+stage+'-20261115',
        name='LionMuon single GPU effective batch32 '+stage,dir=str(common.ROOT/'wandb'),resume='never',settings=wandb.Settings(init_timeout=60))
    try:
        if stage=='preflight':
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_lion_ttt_batch32.py','-v'],check=True)
            export(run,'preflight_complete',space=common.space());return
        import torch
        if torch.cuda.device_count()!=1 or int(os.environ.get('OMPI_COMM_WORLD_SIZE','1'))!=1:raise RuntimeError('Exactly one GPU required')
        timing={}
        for optimizer in ('muon_reference','lion_muon'):
            timing[optimizer]=measure(optimizer,run,timing)
            export(run,'running',timing=timing,space=common.space())
        result=dict(timing=timing,speedup=timing['muon_reference']['seconds_per_step']/timing['lion_muon']['seconds_per_step'],
            warmup_steps=40,measured_steps=200,method='Complete training step; singleGPU;batch32;accumulation1;20stepperiods; no phase instrumentation.',
            original_training_global_batch=128)
        export(run,'complete',**result,space=common.space())
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:export(run,'failed',error=str(exc));raise
    finally:run.finish()


if __name__=='__main__':main()
