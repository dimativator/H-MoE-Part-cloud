#!/usr/bin/env python3
"""Benchmark Muon optimizer-state communication under Megatron PP x DP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "tiny": {"layers": 4, "hidden": 256, "ffn": 768, "heads": 4},
    "257m": {"layers": 12, "hidden": 1024, "ffn": 2816, "heads": 8},
    "500m": {"layers": 18, "hidden": 1280, "ffn": 3584, "heads": 20},
    "4.9b": {"layers": 32, "hidden": 3456, "ffn": 9216, "heads": 27},
    "9.9b-pp2": {"layers": 64, "hidden": 3456, "ffn": 9216, "heads": 27},
    "5.0b-wide": {"layers": 14, "hidden": 5120, "ffn": 13824, "heads": 40},
}
METHODS = {
    "muon_bf16_states": ("muon", "bfloat16"),
    "muon_fp8_states": ("muon", "fp8"),
    "frugal_muon_muon_bf16_states": ("frugal_muon_muon", "bfloat16"),
    "frugal_muon_muon_fp8_states": ("frugal_muon_muon", "fp8"),
}
PROFILE_PREFIX = "[OPTIMIZER STATE COMM PROFILE] "
ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
MEMORY_RE = re.compile(r"stage4 peak memory bytes: allocated=(\d+) reserved=(\d+)")
PHASES = (
    "newton_schulz_ms",
    "wire_encode_ms",
    "state_all_gather_ms",
    "wire_decode_ms",
    "other_ms",
    "optimizer_ms",
)


def _csv(value: str, allowed) -> list[str]:
    values = [item for item in value.split(",") if item]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported values: {unknown}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="5.0b-wide")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--transformer-impl",
        choices=("transformer_engine", "local"),
        default="transformer_engine",
    )
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=30)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--update-gap", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/megatron_muon_state_comm"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.models = _csv(args.models, MODELS)
    args.methods = _csv(args.methods, METHODS)
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.global_batch_size % args.data_parallel_size:
        parser.error("global batch size must be divisible by data parallel size")
    if args.global_batch_size < args.micro_batch_size * args.data_parallel_size:
        parser.error("global batch is smaller than one micro batch per DP rank")
    if any(METHODS[method][0] == "frugal_muon_muon" for method in args.methods):
        if args.tensor_parallel_size != 1:
            parser.error("FRUGAL Muon-Muon currently requires tensor parallel size 1")
    return args


def build_command(
    root: Path,
    *,
    model_name: str,
    method: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    density: float,
    update_gap: int,
    external_distributed: bool = False,
    transformer_impl: str = "transformer_engine",
) -> list[str]:
    del root
    model = MODELS[model_name]
    optimizer, state_precision = METHODS[method]
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    total_steps = warmup_steps + measure_steps
    launcher = [sys.executable, "stage4/pretrain_gpt.py"]
    if not external_distributed:
        launcher = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(world_size),
            "stage4/pretrain_gpt.py",
        ]
    command = launcher + [
        "--optimizer-state-precision",
        state_precision,
        "--num-layers",
        str(model["layers"]),
        "--hidden-size",
        str(model["hidden"]),
        "--ffn-hidden-size",
        str(model["ffn"]),
        "--num-attention-heads",
        str(model["heads"]),
        "--seq-length",
        str(sequence_length),
        "--max-position-embeddings",
        str(sequence_length),
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
        str(tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_parallel_size),
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--transformer-impl",
        transformer_impl,
        "--optimizer",
        optimizer,
        "--muon-momentum",
        "0.95",
        "--muon-nesterov",
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
        "--muon-distributed-state-sharding",
        "--muon-profile-state-communication",
        "--frugal-density",
        str(density),
        "--frugal-update-gap",
        str(update_gap),
        "--frugal-coord-choice",
        "columns",
        "--frugal-inactive-lr-scale",
        "1.0",
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
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
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
    if transformer_impl == "local":
        command.extend(
            (
                "--no-rope-fusion",
                "--no-persist-layer-norm",
                "--no-masked-softmax-fusion",
                "--attention-backend",
                "unfused",
            )
        )
    return command


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def parse_output(output: str, warmup_steps: int, measure_steps: int) -> dict:
    step_times = [
        float(match.group(2))
        for match in ITERATION_RE.finditer(output)
        if int(match.group(1)) > warmup_steps
    ][:measure_steps]
    profiles = defaultdict(list)
    for line in output.splitlines():
        if PROFILE_PREFIX not in line:
            continue
        profile = json.loads(line.split(PROFILE_PREFIX, 1)[1])
        if warmup_steps <= int(profile["step"]) < warmup_steps + measure_steps:
            profiles[int(profile["step"])].append(profile)

    result = {
        "samples": len(step_times),
        "mean_step_ms": _mean(step_times),
        "_step_times": step_times,
        "_profiles": [
            profile for per_rank in profiles.values() for profile in per_rank
        ],
    }
    critical_profiles = [
        max(per_rank, key=lambda profile: float(profile["optimizer_ms"]))
        for _, per_rank in sorted(profiles.items())
    ]
    for phase in PHASES:
        result[phase] = _mean([float(profile[phase]) for profile in critical_profiles])
    for key in ("stateful_wire_bytes", "stateless_wire_bytes", "received_wire_bytes"):
        per_step = [float(profile[key]) for profile in critical_profiles]
        result[key] = int(statistics.mean(per_step)) if per_step else None
    memories = [(int(a), int(r)) for a, r in MEMORY_RE.findall(output)]
    result["peak_allocated_bytes"] = max((item[0] for item in memories), default=None)
    result["peak_reserved_bytes"] = max((item[1] for item in memories), default=None)
    return result


def run_one(root: Path, args: argparse.Namespace, model: str, method: str) -> dict:
    external_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    command = build_command(
        root,
        model_name=model,
        method=method,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        density=args.density,
        update_gap=args.update_gap,
        external_distributed=external_distributed,
        transformer_impl=args.transformer_impl,
    )
    if args.dry_run:
        print(" ".join(command))
        return {}
    env = os.environ.copy()
    source_paths = [
        str(root / "third_party" / "Megatron-LM"),
        str(root / "third_party" / "emerging-optimizers"),
        str(root),
    ]
    if env.get("PYTHONPATH"):
        source_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["STAGE4_REPORT_PEAK_MEMORY"] = "1"
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
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        output, _ = process.communicate()
        timed_out = True
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    rank_suffix = f"_rank{rank}" if external_distributed else ""
    log_path = (
        log_dir
        / f"{model}_{method}_tp{args.tensor_parallel_size}_pp{args.pipeline_parallel_size}_dp{args.data_parallel_size}{rank_suffix}.log"
    )
    log_path.write_text("COMMAND: " + " ".join(command) + "\n\n" + output)
    row = {
        "model": model,
        "method": method,
        "states": METHODS[method][1],
        "tp": args.tensor_parallel_size,
        "pp": args.pipeline_parallel_size,
        "dp": args.data_parallel_size,
        "world_size": args.tensor_parallel_size
        * args.pipeline_parallel_size
        * args.data_parallel_size,
        "gradient_sync": "megatron_ddp_allreduce",
        "optimizer_state_sync": "row_shard_allgather_within_dp_group",
        "model_weight_sharding": "pipeline_parallel",
        "parameter_sync": "replicated_within_dp_group",
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "return_code": process.returncode,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "rank": rank,
    }
    row.update(parse_output(output, args.warmup_steps, args.measure_steps))
    row["status"] = (
        "timeout"
        if timed_out
        else (
            "ok"
            if process.returncode == 0 and row["samples"] == args.measure_steps
            else "error"
        )
    )
    return row


def aggregate_external(rank_rows: list[list[dict]]) -> list[dict]:
    grouped = defaultdict(list)
    for rows in rank_rows:
        for row in rows:
            grouped[(row["model"], row["method"])].append(row)

    aggregated = []
    for key, rows in grouped.items():
        rank_zero = next((row for row in rows if row["rank"] == 0), rows[0])
        result = {k: v for k, v in rank_zero.items() if not k.startswith("_")}
        all_profiles = defaultdict(list)
        for row in rows:
            for profile in row.get("_profiles", []):
                all_profiles[int(profile["step"])].append(profile)
        critical_profiles = [
            max(profiles, key=lambda profile: float(profile["optimizer_ms"]))
            for _, profiles in sorted(all_profiles.items())
        ]
        for phase in PHASES:
            result[phase] = _mean(
                [float(profile[phase]) for profile in critical_profiles]
            )
        for field in (
            "stateful_wire_bytes",
            "stateless_wire_bytes",
            "received_wire_bytes",
        ):
            values = [float(profile[field]) for profile in critical_profiles]
            result[field] = int(statistics.mean(values)) if values else None
        step_times = next(
            (row["_step_times"] for row in rows if row["_step_times"]), []
        )
        result["samples"] = len(step_times)
        result["mean_step_ms"] = _mean(step_times)
        result["peak_allocated_bytes"] = max(
            (
                row["peak_allocated_bytes"]
                for row in rows
                if row["peak_allocated_bytes"] is not None
            ),
            default=None,
        )
        if any(row["status"] != "ok" for row in rows):
            result["status"] = "error"
        result["rank"] = "critical-rank"
        aggregated.append(result)
    return sorted(aggregated, key=lambda row: (row["model"], row["method"]))


def _public_rows(rows: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def write_results(args: argparse.Namespace, rows: list[dict]) -> None:
    for row in rows:
        optimizer_ms = row.get("optimizer_ms")
        step_ms = row.get("mean_step_ms")
        row["outside_optimizer_ms"] = (
            round(step_ms - optimizer_ms, 4)
            if step_ms is not None and optimizer_ms is not None
            else None
        )
        communication_phases = (
            row.get("wire_encode_ms"),
            row.get("state_all_gather_ms"),
            row.get("wire_decode_ms"),
        )
        row["total_state_communication_ms"] = (
            round(sum(communication_phases), 4)
            if all(value is not None for value in communication_phases)
            else None
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| Model | Method | States | Step | Outside optimizer | Optimizer | Newton-Schulz | Wire encode | State all-gather | Wire decode | Total state communication | Other | Peak memory |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = "FRUGAL Muon-Muon" if row["method"].startswith("frugal") else "Muon"
        state = "FP8" if row["states"] == "fp8" else "BF16"

        def timing(key):
            value = row.get(key)
            return "n/a" if value is None else f"{value:.1f}"

        peak = row.get("peak_allocated_bytes")
        peak_text = "n/a" if peak is None else f"{peak / 2**30:.2f} GB"
        table.append(
            "| "
            + " | ".join(
                (
                    row["model"],
                    method,
                    state,
                    timing("mean_step_ms"),
                    timing("outside_optimizer_ms"),
                    timing("optimizer_ms"),
                    timing("newton_schulz_ms"),
                    timing("wire_encode_ms"),
                    timing("state_all_gather_ms"),
                    timing("wire_decode_ms"),
                    timing("total_state_communication_ms"),
                    timing("other_ms"),
                    peak_text,
                )
            )
            + " |"
        )
    (args.output_dir / "results.md").write_text(
        "# Muon optimizer-state communication\n\n"
        "All times are milliseconds per training step. Phase values use the slowest "
        "rank for every step and then average over measured steps.\n\n"
        + "\n".join(table)
        + "\n"
    )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    external_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    external_rank = int(os.environ.get("RANK", "0"))
    expected_world_size = (
        args.tensor_parallel_size
        * args.pipeline_parallel_size
        * args.data_parallel_size
    )
    if external_world_size > 1 and external_world_size != expected_world_size:
        raise SystemExit(
            f"external WORLD_SIZE={external_world_size} != TP*PP*DP={expected_world_size}"
        )
    for model in args.models:
        for method in args.methods:
            print(f"RUN model={model} method={method}", flush=True)
            row = run_one(root, args, model, method)
            if row:
                rows.append(row)
                if external_world_size == 1:
                    write_results(args, _public_rows(rows))
                print(
                    f"RESULT status={row['status']} step={row['mean_step_ms']} "
                    f"comm={row['state_all_gather_ms']}",
                    flush=True,
                )
    if external_world_size == 1:
        return int(any(row["status"] != "ok" for row in rows))

    rank_path = args.output_dir / f"rank{external_rank}_results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path.write_text(json.dumps(rows, indent=2) + "\n")
    if external_rank != 0:
        return int(any(row["status"] != "ok" for row in rows))

    deadline = time.monotonic() + 120
    expected_paths = [
        args.output_dir / f"rank{rank}_results.json"
        for rank in range(external_world_size)
    ]
    while time.monotonic() < deadline and not all(
        path.exists() for path in expected_paths
    ):
        time.sleep(1)
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise SystemExit(f"timed out waiting for rank results: {missing}")
    rank_rows = [json.loads(path.read_text()) for path in expected_paths]
    rows = aggregate_external(rank_rows)
    write_results(args, rows)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
