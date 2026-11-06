"""Per-step, per-phase CUDA memory profile for the training loop (--mem-profile).

Each profiled step records
  - allocated memory at step start (parameters, gradients, optimizer state, caches),
  - memory held after the first microbatch forward (saved activations + logits),
  - the peak over forward/backward of all microbatches (plus gradient clipping),
  - the peak during opt.step(),
  - which of the two phases sets the step peak,
  - persistent bytes by category, counted once per storage.

The JSON written at the end holds the per-step records and a summary. Peaks are
torch.cuda.max_memory_allocated(), so caching-allocator slack is excluded; reserved
memory is recorded separately.
"""

import json
import os
import time
from pathlib import Path

import torch

_GB = 1e9


def _storage_key(t):
    try:
        return (t.device, t.untyped_storage().data_ptr())
    except (RuntimeError, NotImplementedError):
        return (t.device, t.data_ptr())


def _storage_nbytes(t):
    try:
        return t.untyped_storage().nbytes()
    except (RuntimeError, NotImplementedError):
        return t.numel() * t.element_size()


def _walk_tensors(obj, visited, depth=0):
    """Yield tensors reachable from containers and plain objects (optimizer state)."""
    if depth > 6 or id(obj) in visited:
        return
    visited.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_tensors(v, visited, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk_tensors(v, visited, depth + 1)
    elif isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
        return
    elif hasattr(obj, "__dict__"):
        # e.g. projector objects stored in optimizer state
        yield from _walk_tensors(vars(obj), visited, depth + 1)


def _iter_optimizers(opt, seen=None):
    """The optimizer plus any optimizers it wraps (multi-optimizer wrappers)."""
    seen = set() if seen is None else seen
    if id(opt) in seen:
        return
    seen.add(id(opt))
    yield opt
    for value in vars(opt).values():
        items = value.values() if isinstance(value, dict) else (
            value if isinstance(value, (list, tuple)) else [value]
        )
        for item in items:
            if isinstance(item, torch.optim.Optimizer):
                yield from _iter_optimizers(item, seen)


def persistent_bytes(model, opt):
    """Bytes held by parameters, gradients, optimizer state and other model tensors.

    Every storage is counted once, in the first category that reaches it.
    """
    counted = set()

    def add(tensors):
        total = 0
        for t in tensors:
            key = _storage_key(t)
            if key in counted:
                continue
            counted.add(key)
            total += _storage_nbytes(t)
        return total

    params = list(model.parameters())
    out = {
        "params": add(params),
        "grads": add(p.grad for p in params if p.grad is not None),
    }

    state_tensors = []
    for o in _iter_optimizers(opt):
        state = getattr(o, "state", None)
        if state is not None:
            state_tensors.extend(_walk_tensors(dict(state), set()))
    out["optimizer_state"] = add(state_tensors)

    # Buffers and plain tensor attributes, e.g. FP8 weight caches and RoPE tables.
    other = list(model.buffers())
    for module in model.modules():
        other.extend(v for v in vars(module).values() if isinstance(v, torch.Tensor))
    device_type = params[0].device.type if params else "cpu"
    out["other_model_tensors"] = add(t for t in other if t.device.type == device_type)
    return out


class MemProfiler:
    """Hooks called from the training loop; see src/optim/base.py."""

    def __init__(self, cfg, model, opt, rank=0, world_size=1):
        self.model = model
        self.opt = opt
        self.n_steps = cfg.mem_profile_steps
        self.cuda = torch.cuda.is_available() and "cuda" in str(cfg.device)
        path = Path(cfg.mem_profile)
        if world_size > 1:
            path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
        self.path = path
        self.snapshot_path = cfg.mem_snapshot
        if self.snapshot_path and world_size > 1:
            p = Path(self.snapshot_path)
            self.snapshot_path = str(p.with_name(f"{p.stem}.rank{rank}{p.suffix}"))
        self.config = {
            "opt": cfg.opt,
            "fp8": bool(getattr(cfg, "fp8", False)),
            "fp8_optim": bool(getattr(cfg, "fp8_optim", False)),
            "batch_size": cfg.batch_size,
            "acc_steps": cfg.acc_steps,
            "sequence_length": cfg.sequence_length,
            "n_layer": getattr(cfg, "n_layer", None),
            "n_embd": getattr(cfg, "n_embd", None),
            "activation_checkpointing": getattr(cfg, "activation_checkpointing", "none"),
            "activation_offload": bool(getattr(cfg, "activation_offload", False)),
            "world_size": world_size,
            "device_name": torch.cuda.get_device_name() if self.cuda else "cpu",
        }
        self.records = []
        self._rec = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _allocated(self):
        return torch.cuda.memory_allocated() if self.cuda else 0

    def _peak(self):
        return torch.cuda.max_memory_allocated() if self.cuda else 0

    def _reset_peak(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    # ── hooks ────────────────────────────────────────────────────────────────
    def step_begin(self, step):
        is_snapshot_step = self.snapshot_path and self.cuda and len(self.records) == self.n_steps - 1
        if is_snapshot_step:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        self._sync()
        self._reset_peak()
        self._rec = {
            "step": step,
            "allocated_start": self._allocated(),
            "snapshot": bool(is_snapshot_step),
            "t0": time.perf_counter(),
        }

    def before_forward(self, microstep_idx):
        if microstep_idx == 0:
            self._rec["allocated_before_fwd0"] = self._allocated()

    def after_forward(self, microstep_idx):
        if microstep_idx == 0:
            self._rec["allocated_after_fwd0"] = self._allocated()

    def before_opt_step(self):
        self._rec["peak_fwd_bwd"] = self._peak()
        self._rec["allocated_before_opt"] = self._allocated()
        self._reset_peak()

    def after_opt_step(self):
        self._rec["peak_opt_step"] = self._peak()
        self._rec["allocated_after_opt"] = self._allocated()

    def step_end(self):
        """Finish the step; returns True once all profiled steps are written."""
        self._sync()
        rec = self._rec
        rec["step_time_s"] = time.perf_counter() - rec.pop("t0")
        rec["reserved_end"] = torch.cuda.memory_reserved() if self.cuda else 0
        rec["peak_step"] = max(rec["peak_fwd_bwd"], rec["peak_opt_step"])
        rec["binding_phase"] = (
            "opt_step" if rec["peak_opt_step"] > rec["peak_fwd_bwd"] else "fwd_bwd"
        )
        rec["saved_activations_fwd0"] = rec["allocated_after_fwd0"] - rec["allocated_before_fwd0"]
        rec["persistent_end"] = persistent_bytes(self.model, self.opt)
        if rec["snapshot"]:
            torch.cuda.memory._dump_snapshot(self.snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
        self.records.append(rec)
        self._rec = None
        if len(self.records) < self.n_steps:
            return False
        self._write()
        return True

    def _write(self):
        steady = self.records[1:] or self.records

        def worst(records, key):
            return max(r[key] for r in records)

        summary = {
            "peak_gb": worst(self.records, "peak_step") / _GB,
            "peak_steady_gb": worst(steady, "peak_step") / _GB,
            "peak_fwd_bwd_gb": worst(steady, "peak_fwd_bwd") / _GB,
            "peak_opt_step_gb": worst(steady, "peak_opt_step") / _GB,
            "binding_phase_steady": max(steady, key=lambda r: r["peak_step"])["binding_phase"],
            "allocated_start_gb": steady[-1]["allocated_start"] / _GB,
            "saved_activations_fwd0_gb": steady[-1]["saved_activations_fwd0"] / _GB,
            "persistent_gb": {k: v / _GB for k, v in steady[-1]["persistent_end"].items()},
            "reserved_gb": worst(self.records, "reserved_end") / _GB,
            "step_time_s_steady": sum(r["step_time_s"] for r in steady) / len(steady),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config, "summary": summary, "steps": self.records}
        self.path.write_text(json.dumps(payload, indent=2))
        print(f"[MEM PROFILE] wrote {self.path}: peak={summary['peak_gb']:.2f} GB "
              f"(steady {summary['peak_steady_gb']:.2f} GB, binds in "
              f"{summary['binding_phase_steady']})", flush=True)
