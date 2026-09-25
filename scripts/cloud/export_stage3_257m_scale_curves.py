"""Read-only export of SCALE and AdamW 1xC validation histories."""

import json
from pathlib import Path


SCALE_ROOT = Path("/home/jovyan/dimativator/stage3_257m_scale_20260925")
ADAMW_ROOT = Path("/home/jovyan/dimativator/stage3_257m_bf16_1xc_seeds_20260923")


def read_validation(path: Path) -> list[list[float]]:
    points: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("event") == "validation":
                points.append([entry["iter"], entry["val/loss"]])
    return points


def main() -> None:
    series = {}
    for lr in ("1e-4", "5e-4", "1e-3", "2e-3"):
        experiment = f"257m_scale_bf16_1xC_lr{lr}_cloud_2gpu"
        path = SCALE_ROOT / "1xChinchilla_257M_bf16_scale_lr_sweep" / experiment / "metrics.jsonl"
        series[f"SCALE {lr}"] = read_validation(path)
    for seed in (1, 2, 3):
        experiment = f"257m_adamw_bf16_1xC_seed{seed}_cloud_1gpu"
        path = ADAMW_ROOT / "1xChinchilla_257M_bf16_native_states_seed_sweep" / experiment / "metrics.jsonl"
        series[f"AdamW seed {seed}"] = read_validation(path)
    print("CURVES_JSON=" + json.dumps(series, separators=(",", ":")))


if __name__ == "__main__":
    main()
