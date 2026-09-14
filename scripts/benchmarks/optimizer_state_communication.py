#!/usr/bin/env python3
"""Run distributed training-step benchmarks for Muon state communication."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


MODEL_CONFIGS = {
    "tiny": {"n_layer": 4, "n_embd": 256, "n_head": 4},
    "130M": {"n_layer": 12, "n_embd": 768, "n_head": 12},
    "257M": {"n_layer": 12, "n_embd": 1024, "n_head": 8},
    "500M": {"n_layer": 18, "n_embd": 1280, "n_head": 20},
    "1B": {"n_layer": 30, "n_embd": 1536, "n_head": 12},
    "2.8B": {"n_layer": 32, "n_embd": 2560, "n_head": 20},
    "4.3B": {"n_layer": 32, "n_embd": 3200, "n_head": 25},
    "4.8B": {"n_layer": 31, "n_embd": 3456, "n_head": 27},
    "5.0B": {"n_layer": 30, "n_embd": 3584, "n_head": 28},
}
METHODS = {
    "muon": {"optimizer": "muon", "fp8": False},
    "muon_fp8_states": {"optimizer": "muon", "fp8": True},
    "frugal_muon_muon": {"optimizer": "coord_muon", "fp8": False},
    "frugal_muon_muon_fp8_states": {"optimizer": "coord_muon", "fp8": True},
}
RESULT_PREFIX = "[STEP TIME BENCH RESULT] "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIGS, default=["tiny"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--world-sizes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--gpu-indices", default="0,1")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--update-gap", type=int, default=50)
    parser.add_argument(
        "--fp8-expansion",
        choices=["false", "expand"],
        default="expand",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets-dir", default=os.environ.get("DATASETS_DIR", "./datasets"))
    parser.add_argument("--eval-cache-dir", default=os.environ.get("EVAL_CACHE_DIR", "./evals_cache"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def method_args(args: argparse.Namespace, method: str) -> list[str]:
    spec = METHODS[method]
    command_args = [
        "--opt", spec["optimizer"],
        "--optimizer-state-sharding",
        "--optimizer-state-wire-dtype", "fp8" if spec["fp8"] else "bfloat16",
        "--optimizer-comm-profile",
    ]
    if spec["optimizer"] == "muon":
        command_args.extend(["--lite-muon-theta", "0.95", "--lite-ns-steps", "5"])
    else:
        command_args.extend([
            "--density", "{density}",
            "--update_gap", "{update_gap}",
            "--coord_choice", "columns",
            "--momentum", "0.95",
            "--nesterov",
            "--muon_ns_steps", "5",
            "--non_proj_opt", "muon",
        ])
    if spec["fp8"]:
        command_args.extend([
            "--fp8-optim",
            "--fp8-qgroup-size", "128",
            "--fp8-first-order-bit", "E4M3",
            "--fp8-second-order-bit", "E4M3",
            "--fp8-expansion", args.fp8_expansion,
        ])
    return command_args


def build_command(
    args: argparse.Namespace,
    model: str,
    method: str,
    world_size: int,
) -> list[str]:
    cfg = MODEL_CONFIGS[model]
    total_steps = args.warmup_steps + args.measure_steps
    command = [
        args.python, "-m", "torch.distributed.run",
        "--standalone", f"--nproc-per-node={world_size}",
        "src/main.py",
        "--distributed-backend", "nccl",
        "--experiment-name", f"state_comm_{model}_{method}_p{world_size}",
        "--dataset", "fineweb",
        "--datasets-dir", args.datasets_dir,
        "--eval-cache-dir", args.eval_cache_dir,
        "--sequence-length", str(args.sequence_length),
        "--streaming",
        "--workers", str(args.workers),
        "--model", "llama",
        "--n-layer", str(cfg["n_layer"]),
        "--n-embd", str(cfg["n_embd"]),
        "--n-head", str(cfg["n_head"]),
        "--multiple-of", "256",
        "--dtype", "bfloat16",
        "--lr", "1e-3",
        "--weight-decay", "0.1",
        "--grad-clip", "1.0",
        "--scheduler", "none",
        "--warmup-steps", "0",
        "--iterations", str(total_steps),
        "--batch-size", str(args.local_batch_size * world_size),
        "--eval-batch-size", "1",
        "--acc-steps", "1",
        "--eval-interval", str(total_steps + 1),
        "--eval-batches", "1",
        "--log-interval", "0",
        "--no-local-save",
    ]
    command.extend(
        value.format(density=args.density, update_gap=args.update_gap)
        for value in method_args(args, method)
    )
    return command


def gpu_snapshot(gpu_indices: list[str]) -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return str(exc)


def append_row(path: Path, row: dict[str, Any]) -> None:
    jsonl_path = path / "results.jsonl"
    with jsonl_path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")

    flat_keys = sorted({key for item in read_jsonl(jsonl_path) for key in item})
    rows = read_jsonl(jsonl_path)
    with (path / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_one(
    args: argparse.Namespace,
    output_dir: Path,
    model: str,
    method: str,
    world_size: int,
    visible_gpus: list[str],
) -> dict[str, Any]:
    name = f"{model}_{method}_p{world_size}"
    command = build_command(args, model, method, world_size)
    log_path = output_dir / f"{name}.log"
    snapshot = gpu_snapshot(visible_gpus)
    print(f"\n[GPU STATUS BEFORE {name}]\n{snapshot}")
    print(f"[RUN] {shlex.join(command)}", flush=True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(visible_gpus[:world_size]),
        "STEP_TIME_BENCH": "1",
        "STEP_TIME_WARMUP_STEPS": str(args.warmup_steps),
        "STEP_TIME_MEASURE_STEPS": str(args.measure_steps),
        "TOKENIZERS_PARALLELISM": "false",
    })
    metric = None
    tail: list[str] = []
    with log_path.open("w") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            tail.append(line.rstrip())
            tail = tail[-60:]
            if RESULT_PREFIX in line:
                metric = json.loads(line.split(RESULT_PREFIX, 1)[1])
        return_code = process.wait()

    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "method": method,
        "world_size": world_size,
        "local_batch_size": args.local_batch_size,
        "sequence_length": args.sequence_length,
        "density": args.density if "frugal" in method else 1.0,
        "gradient_sync": "ddp_allreduce",
        "optimizer_state_sync": "row_shard_allgather",
        "parameter_sync": "none_replicated_parameters",
        "gpu_indices": ",".join(visible_gpus[:world_size]),
        "gpu_snapshot_before": snapshot.replace("\n", " | "),
        "return_code": return_code,
        "log_file": str(log_path),
    }
    if metric is not None and return_code == 0:
        row.update(metric)
        row["status"] = "ok"
    else:
        row["status"] = "failed"
        row["error"] = "\n".join(tail)[-4000:]
    return row


def main() -> int:
    args = parse_args()
    visible_gpus = [item.strip() for item in args.gpu_indices.split(",") if item.strip()]
    if not visible_gpus:
        raise SystemExit("--gpu-indices must contain at least one GPU")
    if max(args.world_sizes) > len(visible_gpus):
        raise SystemExit("--gpu-indices does not provide enough GPUs")
    if any(size < 1 for size in args.world_sizes):
        raise SystemExit("world sizes must be positive")

    output_dir = args.output_dir or (
        Path("logs/optimizer_state_comm") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if args.dry_run:
        for world_size in args.world_sizes:
            for model in args.models:
                for method in args.methods:
                    print(shlex.join(build_command(args, model, method, world_size)))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir.resolve()}")
    for world_size in args.world_sizes:
        for model in args.models:
            for method in args.methods:
                row = run_one(
                    args, output_dir, model, method, world_size, visible_gpus
                )
                append_row(output_dir, row)
                if row["status"] != "ok":
                    print(f"[FAILED] {model} {method} P={world_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
