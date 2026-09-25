#!/usr/bin/env python3
"""List Stage3 BF16 offline W&B runs without modifying them."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/jovyan/dimativator/stage3_257m_bf16_native_20260921/wandb_offline")


def config_value(config: dict[str, object], key: str) -> object:
    item = config.get(key)
    return item.get("value") if isinstance(item, dict) else item


def main() -> None:
    import yaml

    print(f"ROOT={ROOT} exists={ROOT.is_dir()}")
    for run_file in sorted(ROOT.rglob("run-*.wandb")):
        run_dir = run_file.parent
        config_file = run_dir / "files" / "config.yaml"
        metadata_file = run_dir / "files" / "wandb-metadata.json"
        config = (yaml.safe_load(config_file.read_text()) or {}) if config_file.is_file() else {}
        metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
        args = metadata.get("args") or []
        experiment_arg = next(
            (args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--experiment-name"),
            None,
        )
        print(json.dumps({
            "dir": str(run_dir),
            "run_file_bytes": run_file.stat().st_size,
            "experiment_name": config_value(config, "experiment_name"),
            "wandb_project": config_value(config, "wandb_project"),
            "wandb_group": config_value(config, "wandb_group"),
            "args_experiment_name": experiment_arg,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
