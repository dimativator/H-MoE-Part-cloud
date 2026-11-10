"""SOAP allocator diagnosis, preserving the existing training arithmetic."""
import gzip
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT=Path(__file__).resolve().parent

def summarize(snapshot):
    totals={};free=[];split=0;unsplit=0
    for segment in snapshot['segments']:
        occupied=any(b['state']!='inactive' for b in segment['blocks'])
        for block in segment['blocks']:
            totals[block['state']]=totals.get(block['state'],0)+block['size']
            if block['state']=='inactive':
                free.append(block['size'])
                if occupied:split+=block['size']
                else:unsplit+=block['size']
    return dict(block_bytes=totals,largest_inactive_block=max(free,default=0),
                inactive_split=split,inactive_unsplit=unsplit,
                reserved=sum(s['total_size'] for s in snapshot['segments']),
                inactive_blocks=len(free),segments=len(snapshot['segments']))

def dump(path,value):
    stat=os.statvfs(path.parent)
    if stat.f_bavail*stat.f_frsize<=7_000_000_000:
        raise RuntimeError('CRITICAL CLOUD.RU STORAGE: snapshot reserve unavailable')
    with gzip.open(path,'wt') as f:json.dump(value,f)

class Probe:
    def __init__(self,inventory):
        import torch
        self.torch=torch;self.inventory=inventory;self.root=Path(os.environ['ATTRIBUTION_DIR'])
        self.step=0;self.phase='setup';self.oom=None
        torch._C._cuda_attach_out_of_memory_observer(self.on_oom)
        import subprocess
        (self.root/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),allocator=os.environ['PYTORCH_CUDA_ALLOC_CONF'],processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))))
        global ACTIVE
        ACTIVE=self
    def on_oom(self,device,allocation,device_allocated,device_free):
        snapshot=self.torch.cuda.memory._snapshot()
        self.oom=dict(step=self.step,phase=self.phase,device=device,request=allocation,
                      device_allocated_callback=device_allocated,device_free_callback=device_free,
                      counters=self.torch.cuda.memory_stats(),snapshot=snapshot,
                      python_stack=traceback.format_stack())
    def begin(self):
        self.step+=1;self.phase='forward_backward'
        self.torch.cuda.memory._record_memory_history(max_entries=1000000)
        self.initial=self.torch.cuda.memory._snapshot()
    def before_optimizer(self):
        self.phase='optimizer';self.torch.cuda.synchronize()
        self.fb_peak=self.torch.cuda.max_memory_allocated()
    def finish(self,peak,loss):
        self.torch.cuda.synchronize();snap=self.torch.cuda.memory._snapshot()
        dump(self.root/f'step{self.step}.json.gz',dict(initial=self.initial,final=snap,step=self.step))
        self.torch.cuda.memory._record_memory_history(enabled=None)
        row=dict(step=self.step,loss=loss,peak=max(peak,self.torch.cuda.max_memory_allocated()),forward_backward_peak=self.fb_peak,phase_end=summarize(snap))
        with (self.root/'probe_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('SOAP_STEP='+json.dumps(row),flush=True)

ACTIVE=None
if __name__=='__main__':
    sys.path[:0]=[str(ROOT/'scripts/memory'),str(ROOT/'src'),str(ROOT)]
    import memory_attribution
    memory_attribution.Tracker=Probe
    try:runpy.run_path(str(ROOT/'scripts/memory/attribution_train.py'),run_name='__main__')
    except Exception:
        if ACTIVE is not None and ACTIVE.oom is not None:
            value=ACTIVE.oom
            dump(ACTIVE.root/'oom.json.gz',value)
            summary={k:v for k,v in value.items() if k not in ('snapshot','counters')}
            summary.update(summarize(value['snapshot']))
            summary['oom_trace']=[e for trace in value['snapshot']['device_traces'] for e in trace if e['action']=='oom'][-1:]
            (ACTIVE.root/'oom_summary.json').write_text(json.dumps(summary,indent=2))
            print('SOAP_OOM='+json.dumps(summary),flush=True)
        raise
