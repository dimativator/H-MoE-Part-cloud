"""Run the original Stage III benchmark with CPU/CUDA traces, without changing training."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ["BENCHMARK_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    print("PROFILE_PROTOCOL model=500m warmup=10 measured=50 microbatch=32 accumulation=4", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], check=True)
    print("DISK", shutil.disk_usage(output), flush=True)
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise RuntimeError("Profiling needs at least 8 GiB free for traces")
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one visible allocated GPU")
    spec = importlib.util.spec_from_file_location("stage3_benchmark", root / "scripts/benchmark_training_step.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    original_build = benchmark.build_command

    def profiled_command(*args, **kwargs):
        command = original_build(*args, **kwargs)
        position = command.index("stage4/pretrain_gpt.py")
        command[position] = "scripts/profile_stage3_pretrain.py"
        return command

    benchmark.build_command = profiled_command
    # Patch only the launcher. Model, optimizer, data and numerical flags are unchanged.
    original_run = benchmark.run_one

    def run_with_trace_path(root, output_dir, args, model, precision, optimizer, batch):
        trace_dir = output_dir / "profiles" / f"{model}_{precision}_{optimizer}"
        os.environ["STAGE3_TRACE_DIR"] = str(trace_dir)
        result = original_run(root, output_dir, args, model, precision, optimizer, batch)
        traces = sorted(trace_dir.glob("chunk-*.json.gz"))
        result["trace_directory"] = str(trace_dir)
        result["trace_chunks"] = len(traces)
        result["timing_includes_profiler_overhead"] = True
        if result["status"] == "ok" and len(traces) != 10:
            result["status"] = "error"
            result["message"] = f"Expected ten five-step traces, found {len(traces)}"
        print("PROFILE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "ok":
            print(Path(result["log"]).read_text()[-10000:], flush=True)
        return result

    benchmark.run_one = run_with_trace_path
    sys.argv = [
        str(root / "scripts/benchmark_training_step.py"),
        "--models", "500m", "--precisions", "bf16,fp8_act",
        "--optimizers", "adam,muon,ademamix", "--batches", "128",
        "--micro-batch-size", "32", "--warmup-steps", "10", "--measure-steps", "50",
        "--timeout-seconds", "900", "--output-dir", str(output),
    ]
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
