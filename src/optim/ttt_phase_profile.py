"""Opt-in CUDA timeline timings and actual Newton-Schulz call counts."""
import importlib
import json
import os
from pathlib import Path
import time

import torch


class PhaseProfile:
    def __init__(self, opt, cfg):
        self.opt, self.cfg = opt, cfg
        self.rows, self.current, self.originals = [], None, []
        self.wall_start = self.wall_end = None
        self.matrices = sum(bool(opt.state[p].get('use_muon')) for group in opt.param_groups for p in group['params'])
        if self.matrices <= 0 or cfg.acc_steps * cfg.batch_size != 128:
            raise RuntimeError('Expected reference optimizer and effective batch128')
        for name in ('third_party.lion_muon.muon', 'third_party.lion_muon.lion_muon'):
            module = importlib.import_module(name)
            original = module.zeropower_via_newtonschulz5
            self.originals.append((module, original))
            def wrapped(*args, _original=original, **kwargs):
                if self.current is None:
                    return _original(*args, **kwargs)
                self.current['ns_calls'] += 1
                self.start('NS')
                result = _original(*args, **kwargs)
                self.stop('NS')
                return result
            module.zeropower_via_newtonschulz5 = wrapped

    def begin_step(self, iteration: int) -> None:
        self.current = None
        if not 40 <= iteration < 90:
            return
        if self.wall_start is None:
            torch.cuda.synchronize()
            self.wall_start = time.perf_counter()
        self.current = dict(iteration=iteration, ns_calls=0, events={})
        self.start('total_gpu')

    def start(self, name: str) -> None:
        if self.current is None:
            return
        pair = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
        self.current['events'].setdefault(name, []).append(pair)
        pair[0].record()

    def stop(self, name: str) -> None:
        if self.current is not None:
            self.current['events'][name][-1][1].record()

    def end_step(self) -> None:
        if self.current is not None:
            self.stop('total_gpu')
            self.rows.append(self.current)
            if self.current['iteration'] == 89:
                torch.cuda.synchronize()
                self.wall_end = time.perf_counter()
            self.current = None

    def finish(self) -> dict:
        torch.cuda.synchronize()
        if [r['iteration'] for r in self.rows] != list(range(40, 90)):
            raise RuntimeError('Incomplete phase profiling window')
        wall_ms = (self.wall_end - self.wall_start) * 1000 / len(self.rows)
        results = []
        for row in self.rows:
            ns_expected = self.matrices if self.cfg.opt == 'muon_reference' or row['iteration'] % 5 == 0 else 0
            if row['ns_calls'] != ns_expected:
                raise RuntimeError(f'NS cadence mismatch at {row["iteration"]}: {row["ns_calls"]} != {ns_expected}')
            milliseconds = {name: sum(a.elapsed_time(b) for a,b in pairs) for name,pairs in row['events'].items()}
            milliseconds.setdefault('NS', 0.0)
            milliseconds['optimizer_excluding_NS'] = milliseconds['optimizer'] - milliseconds['NS']
            if milliseconds['optimizer_excluding_NS'] < -0.1:
                raise RuntimeError('Nested NS time exceeds optimizer time')
            results.append(dict(iteration=row['iteration'], ns_calls=row['ns_calls'],
                cuda_event_records=2*sum(len(pairs) for pairs in row['events'].values()), milliseconds=milliseconds))
        phases = sorted({key for row in results for key in row['milliseconds']})
        value = dict(optimizer=self.cfg.opt, measurement_gpus=1, microbatch=self.cfg.batch_size,
            accumulation=self.cfg.acc_steps, measured_steps=len(results), warmup_steps=40,
            matrices=self.matrices, observed_ns_steps=sum(r['ns_calls']>0 for r in results),
            observed_ns_calls=sum(r['ns_calls'] for r in results), mean_wall_ms=wall_ms,
            mean_ms={name:sum(row['milliseconds'].get(name,0.0) for row in results)/len(results) for name in phases},
            rows=results, method='CUDA event timeline, no per-phase synchronization. NS is nested in optimizer; optimizer_excluding_NS is a residual.')
        target = Path(os.environ['LION_TTT_PHASE_PROFILE'])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value,indent=2))
        print('PHASE_PROFILE='+json.dumps(value),flush=True)
        for module, original in self.originals:
            module.zeropower_via_newtonschulz5 = original
        return value


def from_environment(opt, cfg):
    return PhaseProfile(opt,cfg) if os.environ.get('LION_TTT_PHASE_PROFILE') else None
