"""Read-only export of SCALE and AdamW 1xC validation histories."""

import json
import re
from pathlib import Path


SCALE_ROOT = Path("/home/jovyan/dimativator/stage3_257m_scale_20260925")
ADAMW_ROOT = Path("/home/jovyan/dimativator/stage3_257m_bf16_1xc_seeds_20260923")


EVAL_PATTERN = re.compile(r">Eval: Iter=(\d+)\b[^\r\n]*?val_loss=([0-9.]+)")


def read_validation(path: Path) -> list[list[float]]:
    points: dict[int, float] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for match in EVAL_PATTERN.finditer(line):
                points[int(match.group(1))] = float(match.group(2))
    return [[step, loss] for step, loss in sorted(points.items())]


def main() -> None:
    series = {}
    for lr in ("1e-4", "5e-4", "1e-3", "2e-3"):
        experiment = f"257m_scale_bf16_1xC_lr{lr}_cloud_2gpu"
        path = SCALE_ROOT / "logs" / f"{experiment}_rank0.log"
        series[f"SCALE {lr}"] = read_validation(path)
    for seed in (1, 2, 3):
        experiment = f"257m_adamw_bf16_1xC_seed{seed}_cloud_1gpu"
        path = ADAMW_ROOT / "logs" / f"{experiment}.log"
        series[f"AdamW seed {seed}"] = read_validation(path)
    print("CURVES_JSON=" + json.dumps(series, separators=(",", ":")))


if __name__ == "__main__":
    main()
