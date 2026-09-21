#!/usr/bin/env python3
"""Import completed Frugal Muon JSONL histories into W&B exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


EXPECTED_EXPERIMENTS = {
    f"llama500M_frugal_muon_adamw_{precision}_{phase}_2gpu"
    for precision in ("bf16", "fp8_act", "fp8_full", "bf16_fp8_states")
    for phase in ("2xC", "1xC_resume")
}
NON_METRIC_KEYS = {
    "cloud_job",
    "event",
    "experiment",
    "group",
    "inspector_job",
    "queue",
    "timestamp",
}
MPI_PREFIX_RE = re.compile(r"\[\d+,\d+\]<stdout>:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "/workspace-SR006.nfs3/dimativator/"
            "frugal-muon-500m-2gpu-20260912"
        ),
    )
    parser.add_argument("--entity", default="andrey")
    parser.add_argument("--project", default="fp8-pretrain")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        record = clean_mpi_fragments(record)
        if not isinstance(record.get("iter"), int):
            continue
        records.append(record)
    return records


def clean_mpi_fragments(value: Any) -> Any:
    if isinstance(value, str):
        return MPI_PREFIX_RE.sub("", value)
    if isinstance(value, dict):
        return {
            clean_mpi_fragments(key): clean_mpi_fragments(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [clean_mpi_fragments(item) for item in value]
    return value


def merge_by_iteration(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: (item["iter"], item.get("timestamp", 0))):
        iteration = record["iter"]
        row = merged.setdefault(iteration, {"iter": iteration})
        for key, value in record.items():
            if key not in NON_METRIC_KEYS and key != "iter":
                row[key] = value
        if "timestamp" in record:
            row["_timestamp"] = record["timestamp"]
    return [merged[iteration] for iteration in sorted(merged)]


def deterministic_run_id(experiment: str) -> str:
    digest = hashlib.sha1(experiment.encode("utf-8")).hexdigest()[:12]
    return f"frugalmm-{digest}"


def find_experiments(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*.jsonl"):
        experiment = path.parent.name if path.name == "metrics.jsonl" else path.stem
        if experiment in EXPECTED_EXPERIMENTS:
            found[experiment] = path
    missing = EXPECTED_EXPERIMENTS - found.keys()
    if missing:
        raise FileNotFoundError(f"missing experiment logs: {sorted(missing)}")
    return found


def upload_one(
    path: Path,
    *,
    entity: str,
    project: str,
    wandb: Any,
) -> str:
    records = load_records(path)
    experiment = path.parent.name
    if path.name != "metrics.jsonl":
        experiment = path.stem
    group = (
        "1xChinchilla_resume_500M_frugal_muon_2gpu_cloud"
        if "_1xC_resume_" in experiment
        else "2xChinchilla_500M_frugal_muon_2gpu_cloud"
    )
    if not any("final-val/loss" in record for record in records):
        raise ValueError(f"final validation is missing: {experiment}")
    history = merge_by_iteration(records)
    run_id = deterministic_run_id(experiment)

    try:
        existing = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
    except wandb.errors.CommError:
        existing = None
    if existing is not None and existing.summary.get("historical_import_complete"):
        print(f"UPLOAD_SKIPPED experiment={experiment} run_id={run_id}")
        return existing.url

    precision = experiment.removeprefix("llama500M_frugal_muon_adamw_").split(
        "_2xC_2gpu"
    )[0].split("_1xC_resume_2gpu")[0]
    phase = "1xC_resume" if "_1xC_resume_" in experiment else "2xC"
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        name=experiment,
        group=group,
        job_type="historical-jsonl-import",
        tags=["frugal-muon-muon", "500M", "2gpu", "cloudru", phase, precision],
        config={
            "model": "llama-500M",
            "optimizer": "coord_muon",
            "non_projection_optimizer": "adamw",
            "precision": precision,
            "phase": phase,
            "global_batch_size": 128,
            "sequence_length": 1024,
            "density": 0.25,
            "update_gap": 50,
            "coord_choice": "columns",
            "learning_rate": 2e-3,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
            "momentum": 0.95,
            "nesterov": True,
            "muon_ns_steps": 5,
            "source_metrics_jsonl": str(path),
            "historical_import": True,
        },
    )
    assert run is not None
    for row in history:
        run.log(row, step=row["iter"])
    run.summary["historical_import_complete"] = True
    run.summary["source_record_count"] = len(records)
    run.summary["imported_iteration_count"] = len(history)
    run.finish()
    url = f"{os.environ.get('WANDB_BASE_URL', 'https://wandb.ai')}/{entity}/{project}/runs/{run_id}"
    print(
        f"UPLOAD_COMPLETE experiment={experiment} run_id={run_id} "
        f"records={len(records)} iterations={len(history)} url={url}"
    )
    return url


def main() -> int:
    args = parse_args()
    if not args.results_root.is_dir():
        raise FileNotFoundError(args.results_root)
    os.environ.setdefault("WANDB_BASE_URL", "https://wandb-radfan.ru")
    if "WANDB_API_KEY" not in os.environ and Path("/home/jovyan/.netrc").is_file():
        os.environ["HOME"] = "/home/jovyan"

    import wandb

    experiments = find_experiments(args.results_root)
    for experiment in sorted(experiments):
        upload_one(
            experiments[experiment],
            entity=args.entity,
            project=args.project,
            wandb=wandb,
        )
    print(f"ALL_UPLOADS_COMPLETE={len(experiments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
