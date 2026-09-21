#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q tests/test_fineweb_live_sharded.py
python - "${1:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_blocks = 157100 * 4 * 16
for rank in range(2):
    payload = json.loads(
        (root / f"train_rank{rank}.third.state.json").read_text()
    )
    emitted = int(payload["stream_state"]["emitted_block_count"])
    if emitted != expected_blocks:
        raise RuntimeError(
            f"rank {rank} emitted blocks {emitted}, expected {expected_blocks}"
        )
    print(f"LIVE_SOURCE_STATE_OK rank={rank} emitted_blocks={emitted}")
print("STAGE3_BF16_LAUNCH_TESTS_OK")
PY
