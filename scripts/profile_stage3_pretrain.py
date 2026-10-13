"""Label model operations and profile exactly steps 11--60 of the Stage III run."""

import functools
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Megatron-LM"))
sys.path.insert(0, str(ROOT))

import torch
import megatron.training.training as training


TRACE_DIR = Path(os.environ["STAGE3_TRACE_DIR"])
TRACE_DIR.mkdir(parents=True, exist_ok=True)
module_names = {}
active = False
profiler = None
chunk = 0


def save_trace(prof) -> None:
    global chunk
    chunk += 1
    trace_path = TRACE_DIR / f"chunk-{chunk:02d}.json.gz"
    prof.export_chrome_trace(str(trace_path))
    rows = []
    for event in prof.key_averages(group_by_input_shape=True):
        rows.append({
            "name": event.key,
            "count": event.count,
            "input_shapes": event.input_shapes,
            "self_cpu_us": event.self_cpu_time_total,
            "cpu_us": event.cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "device_us": event.device_time_total,
        })
    payload = {"chunk": chunk, "first_iteration": 11 + 5 * (chunk - 1),
               "last_iteration": 10 + 5 * chunk, "rows": rows}
    (TRACE_DIR / f"chunk-{chunk:02d}-summary.json").write_text(json.dumps(payload) + "\n")
    print(f"STAGE3_TRACE_SAVED chunk={chunk} path={trace_path} bytes={trace_path.stat().st_size}", flush=True)


original_get_model = training.get_model


@functools.wraps(original_get_model)
def labeled_get_model(*args, **kwargs):
    models = original_get_model(*args, **kwargs)
    for index, model in enumerate(models):
        for name, module in model.named_modules():
            if any(tag in name for tag in ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2",
                                          "core_attention", "output_layer", "word_embeddings")):
                module_names[id(module)] = f"stage3::module[{index}].{name}::{type(module).__name__}"
    return models


training.get_model = labeled_get_model
original_call = torch.nn.Module._call_impl


@functools.wraps(original_call)
def labeled_module_call(self, *args, **kwargs):
    label = module_names.get(id(self)) if active else None
    if label is not None:
        with torch.profiler.record_function(label):
            return original_call(self, *args, **kwargs)
    return original_call(self, *args, **kwargs)


torch.nn.Module._call_impl = labeled_module_call
original_train_step = training.train_step


@functools.wraps(original_train_step)
def profiled_train_step(forward_step_func, data_iterator, model, optimizer,
                        opt_param_scheduler, config, forward_backward_func, iteration=None):
    global active, profiler
    if iteration == 10:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=10),
            on_trace_ready=save_trace, record_shapes=True, with_stack=False,
        )
        profiler.start()
        active = True
        original_optimizer_step = optimizer.step

        @functools.wraps(original_optimizer_step)
        def labeled_optimizer_step(*args, **kwargs):
            with torch.profiler.record_function("stage3::optimizer_step"):
                return original_optimizer_step(*args, **kwargs)

        optimizer.step = labeled_optimizer_step
    if not active:
        return original_train_step(forward_step_func, data_iterator, model, optimizer,
                                   opt_param_scheduler, config, forward_backward_func, iteration)

    @functools.wraps(forward_backward_func)
    def labeled_forward_backward(*args, **kwargs):
        with torch.profiler.record_function("stage3::forward_backward"):
            return forward_backward_func(*args, **kwargs)

    with torch.profiler.record_function(f"stage3::iteration[{iteration + 1}]"):
        result = original_train_step(forward_step_func, data_iterator, model, optimizer,
                                     opt_param_scheduler, config, labeled_forward_backward, iteration)
    profiler.step()
    if iteration == 59:
        profiler.stop()
        active = False
    return result


training.train_step = profiled_train_step
runpy.run_path(str(ROOT / "stage4/pretrain_gpt.py"), run_name="__main__")
