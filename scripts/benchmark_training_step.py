#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "257m": {"layers": 12, "hidden": 1024, "ffn": 2816, "heads": 8},
    "500m": {"layers": 18, "hidden": 1280, "ffn": 3584, "heads": 20},
    "1b": {"layers": 16, "hidden": 2048, "ffn": 5504, "heads": 16},
    "3b": {"layers": 24, "hidden": 3072, "ffn": 8192, "heads": 24},
    "5b": {"layers": 32, "hidden": 3456, "ffn": 9216, "heads": 27},
}
PRECISIONS = ("bf16", "fp8_act", "full_fp8")
OPTIMIZERS = (
    "adam",
    "muon",
    "soap",
    "ademamix",
    "galore",
    "frugal",
    "frugal_muon_muon",
    "slim_adam",
    "apollo",
)
MUON_OPTIMIZERS = {"muon", "frugal_muon_muon"}
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)
SUPPORTED_BATCHES = (*DEFAULT_BATCHES, 64, 128)
PERIODIC_OPTIMIZERS = {"soap", "galore", "frugal", "frugal_muon_muon", "apollo"}
ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
PARAMETER_RE = re.compile(r"number of parameters.*?:\s*([0-9]+)")
MEMORY_RE = re.compile(
    r"stage4 peak memory bytes: allocated=(\d+) reserved=(\d+)"
)
OOM_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "NVTE_ERROR_CUDA_ERROR",
)


def csv_values(value, allowed=None, cast=str):
    result = [cast(item) for item in value.split(",") if item]
    if allowed is not None:
        unknown = sorted(set(result) - set(allowed))
        if unknown:
            raise argparse.ArgumentTypeError(f"unsupported values: {unknown}")
    return result


def batch_layout(global_batch, micro_batch_cap, data_parallel_size):
    if global_batch % data_parallel_size != 0:
        raise ValueError(
            f"global batch {global_batch} must be divisible by data parallel size "
            f"{data_parallel_size}"
        )
    per_rank_batch = global_batch // data_parallel_size
    micro_batch = min(per_rank_batch, micro_batch_cap or per_rank_batch)
    if per_rank_batch % micro_batch != 0:
        raise ValueError(
            f"per-rank batch {per_rank_batch} must be divisible by micro batch "
            f"{micro_batch}"
        )
    return micro_batch, per_rank_batch // micro_batch


def periodic_updates_in_window(warmup_steps, measure_steps, update_gap):
    """Count zero-based periodic updates inside the measured optimizer steps."""
    first_measured_step = warmup_steps + 1
    last_measured_step = warmup_steps + measure_steps
    return sum(
        (step - 1) % update_gap == 0
        for step in range(first_measured_step, last_measured_step + 1)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="257m,500m")
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="cap the micro batch and use gradient accumulation for larger global batches",
    )
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--periodic-update-gap", type=int, default=50)
    parser.add_argument("--projection-density", type=float, default=0.25)
    parser.add_argument(
        "--muon-state-precision",
        choices=("bfloat16", "fp8"),
        default="bfloat16",
        help="persistent Muon momentum precision, independent of activation precision",
    )
    parser.add_argument("--muon-use-syrk", action="store_true")
    parser.add_argument("--muon-batched-newton-schulz", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("BENCHMARK_OUTPUT_DIR", "/home/jovyan/hmoe-cloud/step-time")),
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    args.models = csv_values(args.models, MODELS)
    args.precisions = csv_values(args.precisions, PRECISIONS)
    args.optimizers = csv_values(args.optimizers, OPTIMIZERS)
    args.batches = csv_values(args.batches, SUPPORTED_BATCHES, int)
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("warmup and measurement lengths must be positive")
    if args.micro_batch_size is not None and args.micro_batch_size < 1:
        parser.error("micro batch size must be positive")
    if args.data_parallel_size < 1:
        parser.error("data parallel size must be positive")
    if args.pipeline_parallel_size < 1:
        parser.error("pipeline parallel size must be positive")
    if args.periodic_update_gap < 1:
        parser.error("periodic update gap must be positive")
    if not 0.0 < args.projection_density <= 1.0:
        parser.error("projection density must be in (0, 1]")
    if args.muon_use_syrk and args.muon_batched_newton_schulz:
        parser.error("--muon-use-syrk and --muon-batched-newton-schulz are mutually exclusive")
    for batch in args.batches:
        try:
            batch_layout(batch, args.micro_batch_size, args.data_parallel_size)
        except ValueError as error:
            parser.error(str(error))
    return args


def hardware_metadata():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unavailable"
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = "unavailable"
        cuda_version = "unavailable"
    return {
        "gpu": gpu,
        "hostname": platform.node(),
        "torch": torch_version,
        "cuda": cuda_version,
    }


def build_command(
    root,
    model_name,
    precision,
    optimizer,
    global_batch,
    micro_batch,
    data_parallel_size,
    warmup,
    measured,
    muon_use_syrk=False,
    muon_batched_newton_schulz=False,
    pipeline_parallel_size=1,
    periodic_update_gap=50,
    projection_density=0.25,
    muon_state_precision="bfloat16",
):
    model = MODELS[model_name]
    total_steps = warmup + measured
    world_size = data_parallel_size * pipeline_parallel_size
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(world_size),
        "stage4/pretrain_gpt.py",
        "--optimizer-state-precision",
        (
            "fp8"
            if precision == "full_fp8"
            or (optimizer in MUON_OPTIMIZERS and muon_state_precision == "fp8")
            else "fp32"
        ),
        "--num-layers",
        str(model["layers"]),
        "--hidden-size",
        str(model["hidden"]),
        "--ffn-hidden-size",
        str(model["ffn"]),
        "--num-attention-heads",
        str(model["heads"]),
        "--seq-length",
        "1024",
        "--max-position-embeddings",
        "1024",
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        "1.0",
        "--swiglu",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-5",
        "--disable-bias-linear",
        "--hidden-dropout",
        "0.0",
        "--attention-dropout",
        "0.0",
        "--make-vocab-size-divisible-by",
        "128",
        "--untie-embeddings-and-output-weights",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        str(pipeline_parallel_size),
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--optimizer",
        optimizer,
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.999" if optimizer == "ademamix" else "0.99",
        "--adam-eps",
        "1e-8",
        "--muon-momentum",
        "0.95",
        "--muon-scale-mode",
        "spectral",
        "--muon-extra-scale-factor",
        "0.2",
        "--muon-coefficient-type",
        "quintic",
        "--muon-num-ns-steps",
        "5",
        "--muon-fp32-matmul-prec",
        "medium",
        "--frugal-density",
        str(projection_density),
        "--frugal-update-gap",
        str(periodic_update_gap),
        "--soap-precondition-frequency",
        str(periodic_update_gap),
        "--lr",
        "1e-3",
        "--min-lr",
        "1e-3",
        "--lr-decay-style",
        "constant",
        "--lr-decay-iters",
        str(total_steps),
        "--weight-decay",
        "0.1",
        "--clip-grad",
        "1.0",
        "--micro-batch-size",
        str(micro_batch),
        "--global-batch-size",
        str(global_batch),
        "--train-iters",
        str(total_steps),
        "--mock-data",
        "--num-workers",
        "0",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "50257",
        "--null-tokenizer-eod-id",
        "50256",
        "--null-tokenizer-pad-id",
        "-1",
        "--seed",
        "1234",
        "--init-method-std",
        "0.02",
        "--eval-interval",
        "1000000",
        "--eval-iters",
        "0",
        "--log-interval",
        "1",
    ]
    if precision != "bf16":
        command.extend(
            [
                "--fp8-format",
                "hybrid",
                "--fp8-recipe",
                "delayed",
                "--fp8-amax-history-len",
                "1",
                "--fp8-amax-compute-algo",
                "most_recent",
            ]
        )
    if optimizer == "muon" and muon_use_syrk:
        command.append("--muon-use-syrk")
    if optimizer == "muon" and muon_batched_newton_schulz:
        command.append("--muon-batched-newton-schulz")
    return command


def write_summary(output_dir, results, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "model",
        "precision",
        "optimizer",
        "muon_state_precision",
        "batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "data_parallel_size",
        "pipeline_parallel_size",
        "world_size",
        "periodic_update_gap",
        "periodic_updates_in_measurement",
        "status",
        "mean_step_ms",
        "median_step_ms",
        "stdev_step_ms",
        "min_step_ms",
        "max_step_ms",
        "samples",
        "parameter_count",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "return_code",
        "log",
        "message",
    ]
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})


def result_key(result):
    return (
        result["model"],
        result["precision"],
        result["optimizer"],
        result.get(
            "muon_state_precision",
            "fp8"
            if result["precision"] == "full_fp8"
            and result["optimizer"] in MUON_OPTIMIZERS
            else "bfloat16"
            if result["optimizer"] in MUON_OPTIMIZERS
            else None,
        ),
        result["batch_size"],
        result.get("micro_batch_size", result["batch_size"]),
        result.get("data_parallel_size", 1),
        result.get("pipeline_parallel_size", 1),
    )


def load_previous(output_dir):
    path = output_dir / "results.json"
    if not path.exists():
        return [], {}
    payload = json.loads(path.read_text())
    return payload.get("results", []), payload.get("metadata", {})


def run_one(root, output_dir, args, model, precision, optimizer, batch):
    micro_batch, accumulation_steps = batch_layout(
        batch, args.micro_batch_size, args.data_parallel_size
    )
    stem = f"{model}_{precision}_{optimizer}_bs{batch}"
    effective_muon_state_precision = (
        "fp8"
        if optimizer in MUON_OPTIMIZERS
        and (precision == "full_fp8" or args.muon_state_precision == "fp8")
        else "bfloat16"
        if optimizer in MUON_OPTIMIZERS
        else None
    )
    if effective_muon_state_precision == "fp8" and precision != "full_fp8":
        stem += "_states_fp8"
    if micro_batch != batch:
        stem += f"_mb{micro_batch}"
    if args.data_parallel_size != 1:
        stem += f"_dp{args.data_parallel_size}"
    if args.pipeline_parallel_size != 1:
        stem += f"_pp{args.pipeline_parallel_size}"
    log_path = output_dir / "logs" / f"{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        root,
        model,
        precision,
        optimizer,
        batch,
        micro_batch,
        args.data_parallel_size,
        args.warmup_steps,
        args.measure_steps,
        args.muon_use_syrk,
        args.muon_batched_newton_schulz,
        args.pipeline_parallel_size,
        args.periodic_update_gap,
        args.projection_density,
        args.muon_state_precision,
    )
    env = os.environ.copy()
    source_paths = [
        str(root / "third_party" / "Megatron-LM"),
        str(root / "third_party" / "emerging-optimizers"),
        str(root),
    ]
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["STAGE4_REPORT_PEAK_MEMORY"] = "1"
    env.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout_seconds)
        return_code = process.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return_code = process.returncode
        timed_out = True
    log_path.write_text("COMMAND: " + " ".join(command) + "\n\n" + output)

    times = [
        float(match.group(2))
        for match in ITERATION_RE.finditer(output)
        if int(match.group(1)) > args.warmup_steps
    ][: args.measure_steps]
    parameter_matches = PARAMETER_RE.findall(output)
    memory_matches = MEMORY_RE.findall(output)
    base = {
        "model": model,
        "precision": precision,
        "optimizer": optimizer,
        "muon_state_precision": effective_muon_state_precision,
        "batch_size": batch,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "data_parallel_size": args.data_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "world_size": args.data_parallel_size * args.pipeline_parallel_size,
        "periodic_update_gap": (
            args.periodic_update_gap if optimizer in PERIODIC_OPTIMIZERS else None
        ),
        "periodic_updates_in_measurement": (
            periodic_updates_in_window(
                args.warmup_steps,
                args.measure_steps,
                args.periodic_update_gap,
            )
            if optimizer in PERIODIC_OPTIMIZERS
            else 0
        ),
        "return_code": return_code,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "parameter_count": int(parameter_matches[-1]) if parameter_matches else None,
        "peak_allocated_bytes": int(memory_matches[-1][0]) if memory_matches else None,
        "peak_reserved_bytes": int(memory_matches[-1][1]) if memory_matches else None,
        "log": str(log_path),
    }
    if any(marker.lower() in output.lower() for marker in OOM_MARKERS):
        return {**base, "status": "oom", "samples": len(times), "message": "CUDA OOM"}
    if timed_out:
        return {**base, "status": "timeout", "samples": len(times), "message": "timeout"}
    if return_code != 0:
        return {**base, "status": "error", "samples": len(times), "message": "non-zero exit"}
    if len(times) != args.measure_steps:
        return {
            **base,
            "status": "error",
            "samples": len(times),
            "message": f"expected {args.measure_steps} timed steps",
        }
    return {
        **base,
        "status": "ok",
        "mean_step_ms": round(statistics.mean(times), 4),
        "median_step_ms": round(statistics.median(times), 4),
        "stdev_step_ms": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "min_step_ms": min(times),
        "max_step_ms": max(times),
        "samples": len(times),
        "raw_step_ms": times,
        "message": "",
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous, previous_metadata = load_previous(args.output_dir)
    results_by_key = {result_key(result): result for result in previous}
    metadata = {**previous_metadata, **hardware_metadata()}
    metadata.update(
        {
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "sequence_length": 1024,
            "micro_batch_size_cap": args.micro_batch_size,
            "data_parallel_size": args.data_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "world_size": args.data_parallel_size * args.pipeline_parallel_size,
            "periodic_update_gap": args.periodic_update_gap,
            "projection_density": args.projection_density,
            "muon_use_syrk": args.muon_use_syrk,
            "muon_batched_newton_schulz": args.muon_batched_newton_schulz,
            "muon_state_precision": args.muon_state_precision,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        }
    )

    for model in args.models:
        for precision in args.precisions:
            for optimizer in args.optimizers:
                oom_micro_batch = None
                for batch in args.batches:
                    micro_batch, accumulation_steps = batch_layout(
                        batch, args.micro_batch_size, args.data_parallel_size
                    )
                    key = (
                        model,
                        precision,
                        optimizer,
                        (
                            "fp8"
                            if optimizer in MUON_OPTIMIZERS
                            and (
                                precision == "full_fp8"
                                or args.muon_state_precision == "fp8"
                            )
                            else "bfloat16"
                            if optimizer in MUON_OPTIMIZERS
                            else None
                        ),
                        batch,
                        micro_batch,
                        args.data_parallel_size,
                        args.pipeline_parallel_size,
                    )
                    if oom_micro_batch is not None and micro_batch > oom_micro_batch:
                        result = {
                            "model": model,
                            "precision": precision,
                            "optimizer": optimizer,
                            "muon_state_precision": (
                                "fp8"
                                if optimizer in MUON_OPTIMIZERS
                                and (
                                    precision == "full_fp8"
                                    or args.muon_state_precision == "fp8"
                                )
                                else "bfloat16"
                                if optimizer in MUON_OPTIMIZERS
                                else None
                            ),
                            "batch_size": batch,
                            "micro_batch_size": micro_batch,
                            "gradient_accumulation_steps": accumulation_steps,
                            "data_parallel_size": args.data_parallel_size,
                            "pipeline_parallel_size": args.pipeline_parallel_size,
                            "world_size": args.data_parallel_size * args.pipeline_parallel_size,
                            "status": "skipped_after_oom",
                            "samples": 0,
                            "message": "micro batch is larger than the first OOM micro batch",
                        }
                    elif key in results_by_key and not args.rerun:
                        print(f"SKIP {key}: already recorded", flush=True)
                        if results_by_key[key]["status"] == "oom":
                            oom_micro_batch = micro_batch
                        continue
                    else:
                        print(
                            f"RUN model={model} precision={precision} optimizer={optimizer} "
                            f"global_batch={batch} micro_batch={micro_batch} "
                            f"accumulation_steps={accumulation_steps} "
                            f"data_parallel_size={args.data_parallel_size}",
                            f"pipeline_parallel_size={args.pipeline_parallel_size}",
                            flush=True,
                        )
                        result = run_one(
                            root, args.output_dir, args, model, precision, optimizer, batch
                        )
                        print(
                            f"RESULT model={model} precision={precision} optimizer={optimizer} "
                            f"global_batch={batch} micro_batch={micro_batch} "
                            f"accumulation_steps={accumulation_steps} "
                            f"data_parallel_size={args.data_parallel_size} "
                            f"pipeline_parallel_size={args.pipeline_parallel_size} "
                            f"status={result['status']} "
                            f"mean_step_ms={result.get('mean_step_ms')}",
                            flush=True,
                        )
                        if result["status"] == "oom":
                            oom_micro_batch = micro_batch
                    results_by_key[key] = result
                    write_summary(
                        args.output_dir,
                        sorted(results_by_key.values(), key=result_key),
                        metadata,
                    )

    results = sorted(results_by_key.values(), key=result_key)
    write_summary(args.output_dir, results, metadata)
    failures = [result for result in results if result["status"] in {"error", "timeout"}]
    print(f"SUMMARY total={len(results)} failures={len(failures)} output={args.output_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
