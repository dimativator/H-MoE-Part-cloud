#!/usr/bin/env python3
"""Inspect or sync only completed precision runs, retaining their original IDs."""

import json
import math
import os
from pathlib import Path
import subprocess

import wandb
import yaml


def main() -> None:
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    sync = os.environ.get("SYNC_COMPLETED", "0") == "1"
    api = wandb.Api(timeout=60) if sync else None
    for precision, root in (
        ("bf16", Path("/home/jovyan/dimativator/500m-slimadam-precisions-20261002")),
        ("fp8_act", Path("/workspace-SR006.nfs2/dimativator/500m-slimadam-precisions-20261002")),
    ):
        for phase, group, target in (
            ("2xC", "2xChinchilla_500M_slimadam_wd1e-4_precisions_2gpu", 150914),
            ("1xC_decay", "1xChinchilla_decay_500M_slimadam_wd1e-4_precisions_2gpu", 75457),
        ):
            name = f"llama500M_slim_adam_{precision}_wd1e-4_{phase}_warmup2000_2gpu"
            metrics = root / group / name / "metrics.jsonl"
            final_rows = [json.loads(line) for line in metrics.read_text().splitlines()]
            final_rows = [row for row in final_rows if row.get("iter") == target]
            assert {"validation", "downstream", "lm_eval"}.issubset(
                {row.get("event") for row in final_rows}
            ), f"Incomplete final evaluations: {name}"
            losses = [row["final-val/loss"] for row in final_rows if "final-val/loss" in row]
            assert len(losses) == 1 and math.isfinite(losses[0]), name
            matches = []
            for run_dir in (root / "wandb").rglob("offline-run-*"):
                config_file = run_dir / "files/config.yaml"
                if not config_file.is_file():
                    continue
                config = yaml.safe_load(config_file.read_text())
                if config.get("experiment_name", {}).get("value") == name:
                    matches.append(run_dir)
            assert len(matches) == 1, f"Expected one offline run for {name}, got {len(matches)}"
            run_dir = matches[0]
            run_id = run_dir.name.rsplit("-", 1)[1]
            synced = (run_dir / f"run-{run_id}.wandb.synced").exists()
            print("WANDB_OFFLINE", json.dumps({"name": name, "id": run_id,
                  "path": str(run_dir), "synced": synced, "final_val": losses[0]}), flush=True)
            if not sync:
                continue
            assert api is not None
            remote = None
            try:
                remote = api.run(f"andrey/fp8-pretrain/{run_id}")
            except wandb.errors.CommError as exc:
                if "not find run" not in str(exc).lower() and "not found" not in str(exc).lower():
                    raise
            if remote is None or remote.summary.get("final-val/loss") != losses[0]:
                assert not synced, f"Synced marker exists but remote final differs: {run_id}"
                subprocess.run(["wandb", "sync", "--entity", "andrey", "--project",
                                "fp8-pretrain", "--mark-synced", str(run_dir)], check=True)
                api.flush()
                remote = api.run(f"andrey/fp8-pretrain/{run_id}")
            assert remote.summary.get("final-val/loss") == losses[0], run_id
            assert remote.config.get("experiment_name") == name, run_id
            print("WANDB_VERIFIED", json.dumps({"name": name, "id": run_id,
                  "final_val": losses[0], "url": remote.url}), flush=True)


if __name__ == "__main__":
    main()
