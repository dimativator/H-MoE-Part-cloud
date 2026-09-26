#!/usr/bin/env python3
"""Verify the seven completed Stage3 uploads in the target W&B project."""

import os

import wandb


def main() -> None:
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    api = wandb.Api(timeout=60)
    run_ids = (
        "u91eypyp",
        "t4yfqnan",
        "j1l970l8",
        "2mzhzp6d",
        "c0ad718l",
        "moxb61fy",
        "yxgwpvkr",
    )
    for run_id in run_ids:
        run = api.run(f"andrey/fp8-pretrain/{run_id}")
        config = run.config
        summary = run.summary
        print(
            "WANDB_VERIFIED "
            f"id={run_id} name={run.name!r} group={run.group!r} "
            f"lr={config.get('lr')!r} "
            f"final_val={summary.get('final-val/loss', summary.get('val/loss'))!r} "
            f"url={run.url}",
            flush=True,
        )


if __name__ == "__main__":
    main()
