"""Efficient-Training entry with allocator attribution and verified fixed tokens."""
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from data import fineweb_packed
from profile_data import build_readers
from optim.mem_profile import MemProfiler
from memory_attribution import Tracker, tensor_storages

fineweb_packed.build_packed_fineweb_readers=build_readers
original_init=MemProfiler.__init__
original_begin=MemProfiler.step_begin
original_before=MemProfiler.before_opt_step
original_end=MemProfiler.step_end

def init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    def inventory():
        parameters=list(self.model.parameters())
        return dict(model=tensor_storages(parameters), grads=tensor_storages([p.grad for p in parameters]),
                    optimizer_state=tensor_storages(dict(self.opt.state)),
                    model_buffers=tensor_storages(list(self.model.buffers())))
    self.attribution=Tracker(inventory)

def begin(self,step):
    original_begin(self,step)
    self.attribution.begin()

def before(self):
    self.attribution_loss=float(sys._getframe(1).f_locals['loss'].detach())
    self.attribution.before_optimizer()
    original_before(self)

def end(self):
    peak=max(self._rec['peak_fwd_bwd'],self._rec['peak_opt_step'])
    self.attribution.finish(peak,dict(loss=self.attribution_loss))
    return original_end(self)

MemProfiler.__init__=init
MemProfiler.step_begin=begin
MemProfiler.before_opt_step=before
MemProfiler.step_end=end
sys.argv[0]=str(ROOT/'src/main.py')
runpy.run_path(sys.argv[0],run_name='__main__')
