"""Rebatch four original rank-local packed microsteps into one batch32."""
import copy
import json
from pathlib import Path
import runpy
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]


class RebatchReader:
    requires_checkpoint_state=False
    batch_size=32
    sequence_length=1024
    def __init__(self, source):
        self.source=source
        self.step=0
    def set_step(self, step:int)->None:
        self.source.set_step(step*4)
        self.step=step
    def sample_batch(self):
        import torch
        batches=[self.source.sample_batch() for _ in range(4)]
        self.step+=1
        return tuple(torch.cat([b[i] for b in batches],dim=0) for i in (0,1))


def build_readers(args,*,rank:int,world_size:int)->dict:
    if world_size!=4 or args.batch_size!=32 or args.acc_steps!=1:
        raise RuntimeError('Exactly4GPU,per-rankbatch32,accumulation1 required')
    source_args=copy.copy(args);source_args.batch_size=8
    result=ORIGINAL_BUILD(source_args,rank=rank,world_size=world_size)
    result['train']=RebatchReader(result['train'])
    print('FULL_BATCH32_DATA='+json.dumps(dict(rank=rank,world_size=4,microbatch=32,accumulation=1,
        global_batch=128,source_microsteps_per_update=4,source_per_rank_batch=8)),flush=True)
    return result


if __name__=='__main__':
    from data import fineweb_packed
    ORIGINAL_BUILD=fineweb_packed.build_packed_fineweb_readers
    fineweb_packed.build_packed_fineweb_readers=build_readers
    sys.argv[0]=str(REPO/'src/main.py')
    runpy.run_path(sys.argv[0],run_name='__main__')
