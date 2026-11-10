"""CPU regression: requested vs rounded sizes, pending frees and address reuse."""
from memory_attribution import replay

def snap(blocks,events):
    return dict(segments=[dict(address=0,total_size=4096,blocks=blocks)],device_traces=[events])
def block(pointer,size,requested):
    return dict(address=pointer,size=size,state='active_allocated',requested_size=requested)
initial=snap([block(0,512,100)],[])
events=[dict(action='alloc',addr=512,size=600),dict(action='free_requested',addr=512,size=600),
        dict(action='free_completed',addr=512,size=600),dict(action='alloc',addr=512,size=4),dict(action='snapshot')]
final=snap([block(0,512,100),block(512,512,4)],events)
r=replay(initial,final,{-1:{'model':{0:100}},4:{'optimizer_state':{512:4}}},0,1536)
assert r['matches_measured_peak'] and r['replay_matches_final'] and r['unknown_frees']==0
assert r['components_at_peak']['model']==100 and r['components_at_peak']['persistent_allocation_padding']==412
assert r['components_at_peak']['optimizer_phase_unclassified']==1024
assert 'optimizer_state' not in r['components_at_peak']
assert sum(r['components_at_peak'].values())==1536 and r['reserved_at_peak_bytes']==4096
assert r['requested_at_peak_bytes']==700 and r['allocation_padding_at_peak_bytes']==836
assert r['inactive_split_at_peak_bytes']==2560
# Large allocations consume the tail when the remainder is at most1MiB.
initial=dict(segments=[dict(address=0,total_size=4*1048576,blocks=[])],device_traces=[[]])
events=[dict(action='alloc',addr=0,size=3*1048576)]
final=dict(segments=[dict(address=0,total_size=4*1048576,blocks=[block(0,4*1048576,3*1048576)])],device_traces=[events])
r=replay(initial,final,{},0,4*1048576)
assert r['matches_measured_peak'] and r['replay_matches_final'] and r['allocation_padding_at_peak_bytes']==1048576
print('REPLAY_CPU_TEST_OK')
