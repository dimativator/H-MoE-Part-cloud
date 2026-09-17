#!/usr/bin/env python3
"""Measure single-GPU peak memory for FRUGAL Muon-Muon on Cloud.ru.

Each cell is an isolated 10-step process. Batch size doubles until the first
CUDA OOM, and every completed attempt is appended to a resumable CSV.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


MODELS = {
    "257M": {"n_layer": 12, "n_embd": 1024, "n_head": 8},
    "500M": {"n_layer": 18, "n_embd": 1280, "n_head": 20},
}
PRECISIONS = ("bf16", "fp8_act", "fp8_states", "fp8_full")
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
CSV_FIELDS = (
    "timestamp_utc",
    "model",
    "parameter_count_m",
    "optimizer",
    "optimizer_cli",
    "precision",
    "batch_size",
    "gpu",
    "status",
    "peak_memory_gb",
    "gpu_used_before_mib",
    "gpu_free_before_mib",
    "duration_sec",
    "return_code",
    "expensive_step_policy",
    "log_path",
    "note",
)
OOM_RE = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|out of memory|CUBLAS_STATUS_ALLOC_FAILED",
    re.IGNORECASE,
)
PEAK_RE = re.compile(r"peak_mem=([0-9.]+)GB")
PARAM_RE = re.compile(r"number of parameters:\s*([0-9.]+)M")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/workspace-SR006.nfs3/dimativator/"
            "frugal-muon-memory-cloud-20260917"
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("/workspace-SR006.nfs3/dimativator/fineweb-h200-packed"),
    )
    parser.add_argument("--eval-cache-dir", type=Path, default=Path("/home/jovyan/evals_cache"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def precision_flags(precision: str) -> list[str]:
    activation = [
        "--fp8",
        "--fp8-fabit", "E4M3",
        "--fp8-fwbit", "E4M3",
        "--fp8-babit", "E5M2",
        "--fp8-bwbit", "E5M2",
        "--fp8-group-size", "16",
    ]
    states = [
        "--fp8-optim",
        "--fp8-qgroup-size", "128",
        "--fp8-first-order-bit", "E4M3",
        "--fp8-second-order-bit", "E4M3",
        "--fp8-expansion", "expand",
    ]
    if precision == "bf16":
        return []
    if precision == "fp8_act":
        return activation
    if precision == "fp8_states":
        return states
    if precision == "fp8_full":
        return activation + states
    raise ValueError(f"Unknown precision: {precision}")


def command_for(args: argparse.Namespace, model: str, precision: str, batch_size: int) -> list[str]:
    shape = MODELS[model]
    experiment = f"memory_{model.lower()}_frugal_muon_{precision}_bs{batch_size}"
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "src/main.py",
        "--distributed-backend", "nccl",
        "--experiment-name", experiment,
        "--seed", "0",
        "--data-seed", "1337",
        "--dataset", "fineweb",
        "--datasets-dir", str(args.datasets_dir),
        "--fineweb-replay-world-size", "2",
        "--fineweb-replay-layout", "serial",
        "--eval-cache-dir", str(args.eval_cache_dir),
        "--sequence-length", "1024",
        "--streaming",
        "--workers", "2",
        "--model", "llama",
        "--n-layer", str(shape["n_layer"]),
        "--n-embd", str(shape["n_embd"]),
        "--n-head", str(shape["n_head"]),
        "--multiple-of", "256",
        "--dtype", "bfloat16",
        "--opt", "coord_muon",
        "--non_proj_opt", "adamw",
        "--lr", "1e-3",
        "--weight-decay", "1e-4",
        "--beta1", "0.9",
        "--beta2", "0.99",
        "--momentum", "0.95",
        "--nesterov",
        "--muon_ns_steps", "5",
        "--density", "0.25",
        "--update_gap", "5",
        "--coord_choice", "columns",
        "--grad-clip", "1.0",
        "--scheduler", "none",
        "--warmup-steps", "0",
        "--iterations", str(args.steps),
        "--batch-size", str(batch_size),
        "--eval-batch-size", str(batch_size),
        "--acc-steps", "1",
        "--eval-interval", "1000000",
        "--eval-batches", "1",
        "--log-interval", "1",
        "--no-local-save",
    ]
    return command + precision_flags(precision)


def gpu_snapshot() -> tuple[str, int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {len(output)}")
    name, used, free = output[0].split(", ")
    return name, int(used), int(free)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_state(path: Path, **payload: object) -> None:
    payload["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    if not (args.datasets_dir / "packed_metadata.json").is_file():
        raise FileNotFoundError(f"Packed dataset is missing: {args.datasets_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    state_path = args.output_dir / "state.json"

    rows = read_rows(csv_path)
    completed = {
        (row["model"], row["precision"], int(row["batch_size"]))
        for row in rows
        if row["status"] in {"success", "oom"}
    }
    models = ("257M",) if args.smoke else tuple(MODELS)
    precisions = ("bf16",) if args.smoke else PRECISIONS
    batches = (1,) if args.smoke else BATCH_SIZES

    for model in models:
        for precision in precisions:
            previous_oom = any(
                row["model"] == model
                and row["precision"] == precision
                and row["status"] == "oom"
                for row in rows
            )
            if previous_oom:
                continue
            for batch_size in batches:
                key = (model, precision, batch_size)
                if key in completed:
                    previous = [
                        row for row in rows
                        if row["model"] == model
                        and row["precision"] == precision
                        and row["batch_size"] == str(batch_size)
                    ][-1]
                    if previous["status"] == "oom":
                        break
                    continue

                gpu_name, used_before, free_before = gpu_snapshot()
                command = command_for(args, model, precision, batch_size)
                log_path = logs_dir / f"{model}_FRUGAL_Muon_Muon_{precision}_bs{batch_size}.log"
                write_state(
                    state_path,
                    status="running",
                    model=model,
                    precision=precision,
                    batch_size=batch_size,
                    gpu_name=gpu_name,
                    command=command,
                )
                env = os.environ.copy()
                env.update(
                    PYTHONUNBUFFERED="1",
                    TOKENIZERS_PARALLELISM="false",
                    PYTORCH_ALLOC_CONF="expandable_segments:True",
                    TRITON_CACHE_DIR="/tmp/triton-frugal-muon-memory",
                )
                started = time.monotonic()
                timed_out = False
                with log_path.open("w") as log:
                    log.write("COMMAND: " + " ".join(command) + "\n")
                    log.flush()
                    try:
                        result = subprocess.run(
                            command,
                            cwd=args.repo,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout_sec,
                        )
                        return_code = result.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        return_code = 124

                duration = time.monotonic() - started
                log_text = log_path.read_text(errors="replace")
                peaks = [float(value) for value in PEAK_RE.findall(log_text)]
                params = PARAM_RE.search(log_text)
                oom = bool(OOM_RE.search(log_text))
                if timed_out:
                    status = "timeout"
                elif oom:
                    status = "oom"
                elif return_code == 0 and len(peaks) >= args.steps:
                    status = "success"
                else:
                    status = "error"

                if status == "success":
                    note = f"Cloud.ru {gpu_name}; captured {len(peaks)} optimizer steps."
                elif status == "oom":
                    note = f"Cloud.ru {gpu_name}; first CUDA OOM; larger batches skipped."
                elif status == "timeout":
                    note = f"Cloud.ru {gpu_name}; exceeded {args.timeout_sec} seconds."
                else:
                    note = f"Cloud.ru {gpu_name}; non-OOM failure; inspect log."
                row = {
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model": model,
                    "parameter_count_m": params.group(1) if params else "",
                    "optimizer": "FRUGAL Muon Muon",
                    "optimizer_cli": "coord_muon+adamw_non_proj",
                    "precision": precision,
                    "batch_size": batch_size,
                    "gpu": 0,
                    "status": status,
                    "peak_memory_gb": max(peaks) if peaks else "",
                    "gpu_used_before_mib": used_before,
                    "gpu_free_before_mib": free_before,
                    "duration_sec": round(duration, 2),
                    "return_code": return_code,
                    "expensive_step_policy": "update_gap=5",
                    "log_path": str(log_path),
                    "note": note,
                }
                append_row(csv_path, row)
                rows.append({field: str(value) for field, value in row.items()})
                completed.add(key)
                print("RESULT=" + json.dumps(row, sort_keys=True), flush=True)

                if status in {"error", "timeout"}:
                    write_state(state_path, status=status, row=row)
                    return 1
                if status == "oom":
                    break

    write_state(
        state_path,
        status="smoke_complete" if args.smoke else "complete",
        rows=len(read_rows(csv_path)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
