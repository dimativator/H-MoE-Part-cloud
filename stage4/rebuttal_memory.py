"""Opt-in instrumentation for the Megatron rebuttal memory experiments."""

from contextlib import nullcontext
import gzip
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def emit(event: str, **values) -> None:
    record = dict(event=event, rank=dist.get_rank(), timestamp=time.time(), **values)
    if os.environ.get("REBUTTAL_RUN_DIR"):
        path = Path(os.environ["REBUTTAL_RUN_DIR"]) / f"rank{dist.get_rank()}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    brief = {k: v for k, v in record.items() if k != "inventory"}
    print("REBUTTAL=" + json.dumps(brief), flush=True)


def tensors(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item, seen)
    elif hasattr(value, "basis"):
        yield from tensors(value.basis, seen)


def storage_inventory(values) -> dict:
    records = {}
    dtypes = {}
    for tensor in tensors(values):
        if tensor.is_cuda:
            storage = tensor.untyped_storage()
            records[storage.data_ptr()] = storage.nbytes()
            dtypes[storage.data_ptr()] = str(tensor.dtype)
    dtype_bytes = {}
    for pointer, size in records.items():
        dtype_bytes[dtypes[pointer]] = dtype_bytes.get(dtypes[pointer], 0) + size
    return dict(bytes=sum(records.values()), storages=records, dtype_bytes=dtype_bytes)


def leaves(optimizer):
    for child in getattr(optimizer, "chained_optimizers", [optimizer]):
        yield child, child.optimizer


def inventory(model, optimizer) -> dict:
    wrappers = list(leaves(optimizer))
    return {
        "model": storage_inventory([list(chunk.parameters()) for chunk in model]),
        "grads": storage_inventory([[getattr(p, "main_grad", None), p.grad]
                                     for chunk in model for p in chunk.parameters()]),
        "master": storage_inventory([getattr(w, "fp32_from_float16_groups", [])
                                      for w, _ in wrappers]),
        "optimizer_state": storage_inventory([dict(raw.state) for _, raw in wrappers]),
    }


def install_state_sharding(optimizer) -> None:
    """Shard whole parameter states; replicate model, grads and FP32 masters.

    This is optimizer-state-only sharding, not Megatron's flattened ZeRO-1.
    Whole matrices preserve the semantics of matrix-aware optimizers.
    """
    size, rank = dist.get_world_size(), dist.get_rank()
    loads = [0] * size
    wrappers = list(leaves(optimizer))
    ordered = [(raw, p) for _, raw in wrappers for group in raw.param_groups
               for p in group["params"]]
    owners = {}
    for _, parameter in sorted(ordered, key=lambda item: -item[1].numel()):
        owner = min(range(size), key=lambda i: loads[i])
        owners[id(parameter)] = owner
        loads[owner] += parameter.numel()
    for _, raw in wrappers:
        parameters = [p for group in raw.param_groups for p in group["params"]]
        for parameter in parameters:
            if owners[id(parameter)] != rank:
                raw.state.pop(parameter, None)
        original = raw.step

        def step(*args, _raw=raw, _params=parameters, _original=original, **kwargs):
            hidden = []
            for parameter in _params:
                if owners[id(parameter)] != rank:
                    hidden.append((parameter, parameter.grad))
                    parameter.grad = None
            try:
                result = _original(*args, **kwargs)
            finally:
                for parameter, grad in hidden:
                    parameter.grad = grad
            for parameter in _params:
                dist.broadcast(parameter.data, src=owners[id(parameter)])
            return result

        raw.step = step
    emit("state_sharding", scheme="whole_parameter_optimizer_states_only",
         master_params_sharded=False, parameter_elements_per_owner=loads)


def install(precision: str) -> None:
    import megatron.training.training as training
    import megatron.core.optimizer as mcore_optimizer
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
    from megatron.core.models.gpt.gpt_model import GPTModel
    import megatron.core.transformer.mlp as mlp
    from stage4.fp8_optimizer_states import make_fp8_adamw
    from stage4.fused_fp8_adamw import make_fused_fp8_adamw

    if precision == "fp8":
        for name in ("galore", "apollo", "frugal", "slim_adam"):
            entry = _EMERGING_OPTIMIZERS[name]
            entry.optimizer_cls = make_fp8_adamw(entry.optimizer_cls)
        if os.environ.get("REBUTTAL_FUSED_FALLBACK") == "1":
            # Use the same fused update for AdamW and the Muon fallback.
            mcore_optimizer.Adam = make_fused_fp8_adamw(mcore_optimizer.Adam)

    ac = os.environ.get("REBUTTAL_AC", "none")
    if ac == "selective":
        original_swiglu = mlp.bias_swiglu_impl

        def recompute_swiglu(*args, **kwargs):
            return checkpoint(original_swiglu, *args, use_reentrant=False, **kwargs)

        mlp.bias_swiglu_impl = recompute_swiglu

    original_forward = GPTModel.forward

    def forward(self, *args, **kwargs):
        context = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if os.environ.get("REBUTTAL_OFFLOAD") == "1" else nullcontext())
        with context:
            return original_forward(self, *args, **kwargs)

    GPTModel.forward = forward

    original_setup = training.setup_model_and_optimizer
    state = {"step": 0}

    def setup(*args, **kwargs):
        from megatron.training import get_args
        config = get_args()
        if config.save is not None or config.load is not None:
            raise RuntimeError("Rebuttal memory runs forbid checkpoint save/load")
        model, optimizer, scheduler = original_setup(*args, **kwargs)
        if os.environ.get("REBUTTAL_SHARD") == "1":
            install_state_sharding(optimizer)
        original_step = optimizer.step

        def optimizer_step(*args, **kwargs):
            torch.cuda.synchronize()
            state["forward_backward_peak"] = torch.cuda.max_memory_allocated()
            state["before_optimizer"] = torch.cuda.memory_allocated()
            if state.get("trace"):
                state["before_optimizer_inventory"] = inventory(model, optimizer)
                state["before_optimizer_snapshot"] = torch.cuda.memory._snapshot()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = original_step(*args, **kwargs)
            torch.cuda.synchronize()
            state["optimizer_ms"] = (time.perf_counter() - start) * 1000
            state["optimizer_peak"] = torch.cuda.max_memory_allocated()
            return result

        optimizer.step = optimizer_step
        state.update(model=model, optimizer=optimizer)
        emit("setup", checkpoint_saving=False, ac=ac,
             offload=os.environ.get("REBUTTAL_OFFLOAD") == "1",
             optimizer_classes=[type(raw).__name__ for _, raw in leaves(optimizer)],
             gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__,
             grad_dtypes=sorted({str(p.main_grad.dtype) for chunk in model
                                 for p in chunk.parameters() if hasattr(p, "main_grad")}),
             inventory=inventory(model, optimizer))
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = setup
    original_train_step = training.train_step

    def train_step(*args, **kwargs):
        state["step"] += 1
        step = state["step"]
        trace = os.environ.get("REBUTTAL_TRACE") == "1" and step in (1, 6, 11)
        state["trace"] = trace
        if trace:
            state["initial_inventory"] = inventory(state["model"], state["optimizer"])
            state["initial_snapshot"] = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(max_entries=100000)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original_train_step(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        details = inventory(state["model"], state["optimizer"])
        emit("step", step=step, step_ms=elapsed, optimizer_ms=state["optimizer_ms"],
             losses={name: float(value) for name, value in result[0].items()},
             skipped_iteration=int(result[1]),
             peak_allocated_bytes=max(state["forward_backward_peak"], state["optimizer_peak"]),
             forward_backward_peak_bytes=state["forward_backward_peak"],
             optimizer_peak_bytes=state["optimizer_peak"],
             before_optimizer_bytes=state["before_optimizer"],
             allocated_bytes=allocated, reserved_bytes=reserved,
             reserved_minus_allocated_bytes=reserved - allocated,
             inventory=details)
        if trace:
            path = Path(os.environ["REBUTTAL_RUN_DIR"]) / f"allocator-rank{dist.get_rank()}-step{step}.json.gz"
            snapshot = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            with gzip.open(path, "wt") as handle:
                json.dump(dict(snapshot=snapshot, inventory=details, step=step,
                               initial_snapshot=state.pop("initial_snapshot"),
                               initial_inventory=state.pop("initial_inventory"),
                               before_optimizer_snapshot=state.pop("before_optimizer_snapshot"),
                               before_optimizer_inventory=state.pop("before_optimizer_inventory")), handle)
            emit("allocator_log", path=str(path))
        return result

    training.train_step = train_step
