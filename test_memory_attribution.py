"""CPU regression for peak-time accounting and reused storage addresses."""
from memory_attribution import replay

def snap(blocks,events):
    return dict(segments=[dict(address=100,total_size=1000,blocks=blocks)],device_traces=[events])
def block(pointer,size):
    return dict(address=pointer,size=size,state='active_allocated',requested_size=size)
initial=snap([block(100,100)],[])
events=[dict(action='alloc',addr=200,size=200),dict(action='free_requested',addr=200,size=200),
        dict(action='alloc',addr=200,size=50),dict(action='snapshot')]
final=snap([block(100,100),block(200,50)],events)
r=replay(initial,final,{-1:{'model':{100:100}},3:{'optimizer_state':{200:50}}},0,300)
assert r['matches_measured_peak'] and r['replay_matches_final'] and r['unknown_frees']==0
assert r['components_at_peak']['model']==100
assert r['components_at_peak']['optimizer_phase_unclassified']==200
assert 'optimizer_state' not in r['components_at_peak']
assert sum(r['components_at_peak'].values())==300 and r['reserved_at_peak_bytes']==1000
print('REPLAY_CPU_TEST_OK')
