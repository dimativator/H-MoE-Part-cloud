"""Read prior OOM evidence or run two sequential fresh-process SOAP probes."""
import json
import math
from pathlib import Path
import subprocess
import sys
import run_memory_attribution as base

ROOT=Path(__file__).resolve().parent
OUT=Path('/home/jovyan/soap-oom-diagnostic-20261110')
base.OUT=OUT

def inspect():
    import torch
    assert hasattr(torch._C,'_cuda_attach_out_of_memory_observer')
    print('SOAP_OBSERVER_AVAILABLE='+torch.__version__,flush=True)
    old=Path('/home/jovyan/memory-attribution-20261110/efficient')
    for path in sorted(old.glob('SOAP_fp32_mb32_*')):
        print('SOAP_OLD='+json.dumps(dict(path=str(path),result=json.loads((path/'result.json').read_text()),log_tail=(path/'train.log').read_text(errors='replace')[-6500:],snapshots=[p.name for p in path.glob('snapshot*')])),flush=True)
    for path in OUT.glob('*/result.json'):
        print('SOAP_RESULT='+path.read_text(),flush=True)
    for path in OUT.glob('*/oom_summary.json'):
        print('SOAP_SAVED_OOM='+path.read_text().replace('\n',''),flush=True)

def run(allocator):
    path=OUT/allocator
    if path.exists():raise RuntimeError('Refuse to overwrite probe '+str(path))
    path.mkdir()
    t=dict(method='SOAP',states='fp32',mb=32,repeat=0,allocator=allocator,steps=15,trace=False)
    cmd=base.command(t,path);cmd[cmd.index('scripts/memory/attribution_train.py')]='soap_oom_probe.py'
    env=base.environment(t,path)
    (path/'config.json').write_text(json.dumps(dict(trial=t,command=cmd,allocator=env['PYTORCH_CUDA_ALLOC_CONF'],revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    with (path/'train.log').open('w') as log:
        base.CHILD=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while base.CHILD.poll() is None:
            try:base.CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:base.guard()
    code=base.CHILD.returncode;base.CHILD=None
    rows=[json.loads(s) for s in (path/'probe_steps.jsonl').read_text().splitlines()] if (path/'probe_steps.jsonl').exists() else []
    finite=all(all(math.isfinite(v) for v in r['loss'].values()) for r in rows)
    status='success' if code==0 and len(rows)==15 and finite else 'oom' if (path/'oom_summary.json').exists() else 'failed'
    result=dict(allocator=allocator,status=status,returncode=code,steps=len(rows),finite=finite,peak=max((r['peak'] for r in rows),default=None))
    (path/'result.json').write_text(json.dumps(result));print('SOAP_RESULT='+json.dumps(result),flush=True)
    if (path/'oom_summary.json').exists():print('SOAP_SAVED_OOM='+(path/'oom_summary.json').read_text().replace('\n',''),flush=True)
    if status=='failed':print((path/'train.log').read_text(errors='replace')[-5000:]);raise RuntimeError('Probe failed')

if __name__=='__main__':
    try:
        base.guard();OUT.mkdir(exist_ok=True)
        if '--run' in sys.argv:
            for allocator in ['native','expandable']:base.guard();run(allocator)
        else:inspect()
    except Exception:
        import traceback;traceback.print_exc();print('SOAP_EXIT=1',flush=True)
    else:print('SOAP_EXIT=0',flush=True)
