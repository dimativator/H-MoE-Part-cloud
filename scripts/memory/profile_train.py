"""Run unchanged training with last-step Muon source-line allocation telemetry."""
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from optim.mem_profile import MemProfiler, persistent_bytes

before = MemProfiler.before_opt_step
after = MemProfiler.after_opt_step
write = MemProfiler._write


def trace(frame, event, arg):
    if not frame.f_code.co_filename.endswith('/third_party/lite/muonlite.py'):
        return None
    if event == 'line':
        profiler = trace.profiler
        if len(profiler.line_events) < 15000:
            p = frame.f_locals.get('p')
            state = frame.f_locals.get('state', {})
            profiler.line_events.append(dict(function=frame.f_code.co_name, next_line=frame.f_lineno,
                parameter=state.get('name') if isinstance(state, dict) else None,
                elements=p.numel() if isinstance(p, torch.Tensor) else None,
                allocated=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated()))
    return trace


def before_step(self):
    self._rec['persistent_before_opt'] = persistent_bytes(self.model, self.opt)
    before(self)
    if len(self.records) == self.n_steps - 1:
        self.line_events = []
        trace.profiler = self
        sys.settrace(trace)


def after_step(self):
    sys.settrace(None)
    after(self)
    if hasattr(self, 'line_events'):
        self.path.with_suffix('.lines.json').write_text(json.dumps(self.line_events))


def write_profile(self):
    self.config['optimizer_class'] = type(self.opt).__module__ + '.' + type(self.opt).__name__
    self.config['parameter_dtypes'] = sorted({str(p.dtype) for p in self.model.parameters()})
    write(self)

MemProfiler.before_opt_step = before_step
MemProfiler.after_opt_step = after_step
MemProfiler._write = write_profile
sys.argv[0] = str(ROOT / 'src/main.py')
runpy.run_path(sys.argv[0], run_name='__main__')
