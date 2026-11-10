"""Independent one-GPU memory attribution campaign; logs and metadata only."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
BACKEND='megatron' if (ROOT/'stage4').exists() else 'efficient'
OUT=Path('/home/jovyan/memory-attribution-20261110')/BACKEND
CHILD=None
METHODS=['Muon','AdamW','SOAP','GaLore','APOLLO','SlimAdam']

def guard():
    records=[]
    for path in ['/', '/tmp','/home/jovyan','/workspace-SR006.nfs3','/home/jovyan/shares/SR006.nfs3/mlsub-logs','/workspace-SR006.nfs2']:
        try:
            stat=os.statvfs(path);record=dict(path=path,available_bytes=stat.f_bavail*stat.f_frsize,writable='nfs2' not in path)
        except OSError as error:
            record=dict(path=path,error=str(error),available_bytes=0,writable='nfs2' not in path)
        records.append(record)
    bad=[r for r in records if r['writable'] and r['available_bytes']<=5_500_000_000]
    value=dict(timestamp=time.time(),records=records,status='stop' if bad else 'ok',reserve=5_000_000_000)
    print('ATTR_STORAGE='+json.dumps(value),flush=True)
    if bad:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid,signal.SIGTERM)
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL)
        raise RuntimeError('CRITICAL CLOUD.RU STORAGE: TRACKED CAMPAIGN STOPPED; NO CHECKPOINTS')
    if OUT.exists():
        with (OUT/'disk_guard.jsonl').open('a') as handle:handle.write(json.dumps(value)+'\n')

def grid():
    return [dict(method=m,states=s,mb=b,repeat=r,allocator=a,steps=15,trace=a=='native' and r==0)
            for m,s,b,(a,r) in itertools.product(METHODS,['fp32','fp8'],[1,32],[('native',0),('native',1),('native',2),('expandable',0)])]

def command(t,path):
    if BACKEND=='efficient':
        sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
        from benchmark_peak_memory import command_for
        cmd=command_for(SimpleNamespace(steps=t['steps']),'500M','Slim Adam' if t['method']=='SlimAdam' else t['method'],
                        'bf16' if t['states']=='fp32' else 'fp8_states',t['mb'])
        cmd=[sys.executable,'-m','torch.distributed.run']+cmd[1:]
        cmd[cmd.index('src/main.py')]='scripts/memory/attribution_train.py'
        cmd[cmd.index('--datasets-dir')+1]='/workspace-SR006.nfs3/dimativator/fineweb-h200-packed'
        cmd[cmd.index('--eval-cache-dir')+1]='/home/jovyan/evals_cache'
        cmd+=['--seed','1234','--activation-checkpointing','none','--mem-profile',str(path/'profile.json'),'--mem-profile-steps',str(t['steps'])]
    else:
        sys.path.insert(0,str(ROOT/'scripts'))
        from benchmark_rebuttal_memory import command as build
        names=dict(Muon='muon',AdamW='adam',SOAP='soap',GaLore='galore',APOLLO='apollo',SlimAdam='slim_adam')
        cmd=build(dict(method=names[t['method']],states=t['states'],mb=t['mb'],dp=1,steps=t['steps'],ac='none'))
    return cmd

def environment(t,path):
    env=os.environ.copy()
    env.update(WANDB_MODE='disabled',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
               ATTRIBUTION_DIR=str(path),ATTRIBUTION_TRACE=str(int(t['trace'])),
               PYTORCH_CUDA_ALLOC_CONF='backend:native,expandable_segments:'+('True' if t['allocator']=='expandable' else 'False'),
               TRITON_CACHE_DIR='/tmp/attribution-triton',TORCHINDUCTOR_CACHE_DIR='/tmp/attribution-inductor',CUDA_CACHE_PATH='/tmp/attribution-cuda')
    env.pop('PYTORCH_ALLOC_CONF',None)
    if BACKEND=='megatron':
        env.update(PYTHONNOUSERSITE='1',REBUTTAL_MEMORY='1',REBUTTAL_TRACE='0',REBUTTAL_SHARD='0',REBUTTAL_AC='none',REBUTTAL_OFFLOAD='0',REBUTTAL_FUSED_FALLBACK='0',REBUTTAL_RUN_DIR=str(path),STAGE4_FP8_BATCHED='1')
        env['PYTHONPATH']=':'.join(str(p) for p in [ROOT/'third_party/Megatron-LM',ROOT/'third_party/emerging-optimizers',ROOT])
    return env

def run(t):
    global CHILD
    guard()
    name=f"{t['method']}_{t['states']}_mb{t['mb']}_{t['allocator']}_r{t['repeat']}"
    path=OUT/name
    if (path/'result.json').exists():
        result=json.loads((path/'result.json').read_text())
        if result['status'] in ['success','oom']:return result
        raise RuntimeError('Existing failed result requires diagnosis before retry: '+str(path))
    if path.exists():
        archive=OUT/'interrupted';archive.mkdir(exist_ok=True);path.rename(archive/(name+'_'+str(time.time_ns())))
    path.mkdir()
    cmd=command(t,path);env=environment(t,path)
    config=dict(backend=BACKEND,trial=t,command=cmd,revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),allocator=env['PYTORCH_CUDA_ALLOC_CONF'])
    (path/'config.json').write_text(json.dumps(config,indent=2))
    print('ATTR_START='+json.dumps(config),flush=True)
    with (path/'train.log').open('w') as log:
        CHILD=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while CHILD.poll() is None:
            try:CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:guard()
    code=CHILD.returncode;CHILD=None
    events=[json.loads(line) for line in (path/'telemetry.jsonl').read_text().splitlines()] if (path/'telemetry.jsonl').exists() else []
    log=(path/'train.log').read_text(errors='replace')
    finite=len(events)==t['steps'] and all(e['losses'] and all(math.isfinite(v) for v in e['losses'].values()) for e in events)
    traces=[e['attribution'] for e in events if 'attribution' in e]
    valid_trace=(len(traces)==3 if t['trace'] else not traces) and all(a['replay_matches_final'] and not a['unknown_frees'] and a['matches_measured_peak'] for a in traces)
    status='success' if code==0 and finite and valid_trace else 'oom' if 'out of memory' in log.lower() else 'failed'
    result=dict(config=config,status=status,return_code=code,steps=len(events),finite=finite,valid_trace=valid_trace,
                peak_allocated_bytes=max((e['peak_allocated_bytes'] for e in events),default=None))
    (path/'result.json').write_text(json.dumps(result,indent=2));print('ATTR_RESULT='+json.dumps(result),flush=True)
    if status=='failed':
        print(log[-6000:],flush=True);raise RuntimeError('Trial failed: '+name)
    return result

def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['preflight','smoke','full','inspect'],default='preflight');args=parser.parse_args()
    guard();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='inspect':
        for path in sorted(OUT.glob('*/result.json')):
            print('ATTR_RESULT='+path.read_text().replace('\n',''),flush=True)
        for path in sorted(OUT.glob('*/telemetry.jsonl')):
            for line in path.read_text().splitlines():
                event=json.loads(line);attribution=event.pop('attribution',None)
                print('ATTR_STEP='+json.dumps(dict(trial=path.parent.name,**event)),flush=True)
                if attribution:
                    stacks=attribution.pop('unclassified_stack_bytes')
                    print('ATTR_PEAK='+json.dumps(dict(trial=path.parent.name,step=event['step'],**attribution)),flush=True)
                    for stack in stacks:print('ATTR_STACK='+json.dumps(dict(trial=path.parent.name,step=event['step'],**stack)),flush=True)
        for path in sorted(OUT.glob('*/train.log')):
            if not (path.parent/'result.json').exists(): print('ATTR_UNFINISHED='+json.dumps(dict(path=str(path),tail=path.read_text(errors='replace')[-2500:])),flush=True)
        return
    if args.mode=='preflight':
        subprocess.run([sys.executable,'test_memory_attribution.py'],check=True)
        first=grid()[0];path=OUT/'preflight'
        cmd=command(first,path)
        if BACKEND=='efficient':
            p=subprocess.run([sys.executable,'src/main.py','--help'],env=environment(first,path),capture_output=True,text=True)
            sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'scripts/memory')]
            from profile_data import build_readers
            for mb in [1,32]:
                data=build_readers(SimpleNamespace(datasets_dir='/workspace-SR006.nfs3/dimativator/fineweb-h200-packed',sequence_length=1024,batch_size=mb,eval_batch_size=mb),rank=0,world_size=1)
                for _ in range(15):assert data['train'].sample_batch()[0].shape==(mb,1024)
        else:
            p=subprocess.run([sys.executable,'stage4/pretrain_gpt.py','--optimizer-state-precision','fp32','--help'],env=environment(first,path),capture_output=True,text=True)
        print('IMPORT_EXIT='+str(p.returncode),flush=True)
        if p.returncode:print(p.stderr[-5000:],flush=True);raise RuntimeError('Import failed')
        print('ATTR_PREFLIGHT_OK='+BACKEND,flush=True);return
    selected=grid()
    if args.mode=='smoke':selected=[t for t in selected if t['mb']==1 and t['repeat']==0 and t['allocator']=='native' and (t['method'],t['states']) in [('Muon','fp8'),('SOAP','fp32')]]
    results=[run(t) for t in selected]
    print('ATTR_COMPLETE='+json.dumps(dict(backend=BACKEND,mode=args.mode,trials=len(results),statuses=dict((s,sum(r['status']==s for r in results)) for s in ['success','oom']))),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;traceback.print_exc();print('ATTR_EXIT=1',flush=True)
    else:print('ATTR_EXIT=0',flush=True)
