#!/usr/bin/env bash
set -euo pipefail

readonly data_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed

echo DISK_STATUS
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3
echo PACKED_METADATA
python - "${data_dir}/packed_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads(path.read_text())
print("format", metadata.get("format"))
print("world_size", metadata.get("world_size"))
print("batch_size", metadata.get("batch_size"))
print("sequence_length", metadata.get("sequence_length"))
for rank in metadata["ranks"]:
    print("rank", rank["rank"], "blocks", rank["blocks"])
    for segment in rank.get("segments", []):
        segment_path = path.parent / segment["file"]
        print(
            "segment",
            rank["rank"],
            segment["file"],
            "bytes", segment["bytes"],
            "exists", segment_path.exists(),
            "resolved", segment_path.resolve(),
        )
required_blocks = 314000 * 4 * int(metadata["batch_size"])
print("required_blocks_per_rank_8xc", required_blocks)
print("PACKED_CAPACITY_OK", all(int(rank["blocks"]) >= required_blocks for rank in metadata["ranks"]))
PY
