"""CPU-only comparison of saved SOAP OOM and successful allocation traces."""
import gzip,json
from pathlib import Path
import run_memory_attribution as base
from soap_oom_probe import summarize
ROOT=Path('/home/jovyan/soap-oom-diagnostic-20261110')
base.OUT=ROOT

def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def main():
    base.guard()
    oom=load(ROOT/'native/oom.json.gz');snap=oom['snapshot']
    oome=next(e for e in reversed(snap['device_traces'][0]) if e['action']=='oom')
    print('SOAP_NATIVE='+json.dumps(dict(step=oom['step'],request=oom['request'],device_free=oom['device_free_callback'],**summarize(snap))),flush=True)
    for allocator in ['native','expandable']:
        folder=ROOT/allocator
        print('SOAP_CONFIG='+json.dumps(dict(allocator=allocator,config=json.loads((folder/'config.json').read_text()),runtime=json.loads((folder/'runtime.json').read_text()))),flush=True)
        print('SOAP_FINAL='+json.dumps(dict(allocator=allocator,result=json.loads((folder/'result.json').read_text()),steps=[json.loads(s) for s in (folder/'probe_steps.jsonl').read_text().splitlines()])),flush=True)
    success=load(ROOT/f'expandable/step{oom["step"]}.json.gz')
    initial=success['initial'];final=success['final']
    live={b['address']:b.get('requested_size',b['size']) for s in initial['segments'] for b in s['blocks'] if b['state']=='active_allocated'}
    before=len(initial['device_traces'][0]);events=final['device_traces'][0][before:];unknown=0;matches=[]
    for index,e in enumerate(events):
        if e['action']=='alloc':
            frames=e.get('frames',[])
            if e['size']==oome['size'] and any('nll_loss_backward' in f.get('name','') for f in frames):
                matches.append(dict(step=oom['step'],event_index=index,request=e['size'],requested_live_before=sum(live.values()),requested_live_after=sum(live.values())+e['size'],matching_frames=[f['name'] for f in frames if 'nll_loss_backward' in f.get('name','')][:4]))
            assert e['addr'] not in live
            live[e['addr']]=e['size']
        elif e['action']=='free_requested':
            if live.pop(e['addr'],None) is None:unknown+=1
    expected=sum(b.get('requested_size',b['size']) for s in final['segments'] for b in s['blocks'] if b['state']=='active_allocated')
    print('SOAP_MATCH='+json.dumps(dict(matches=matches,requested_replay_final=sum(live.values()),snapshot_requested_final=expected,unknown_frees=unknown,exact_requested_replay=sum(live.values())==expected)),flush=True)
    assert matches and unknown==0 and sum(live.values())==expected
    print('SOAP_ANALYSIS_OK',flush=True)
if __name__=='__main__':main()
