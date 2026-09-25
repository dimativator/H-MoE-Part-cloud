#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/adamw-source}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}

echo "DATE=$(date --iso-8601=seconds) HOST=$(hostname)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /tmp
test -f "${DATASETS_DIR}/packed_metadata.json"
mkdir -p "${SOURCE_DIR}"

SOURCE_DIR="${SOURCE_DIR}" DATASETS_DIR="${DATASETS_DIR}" python - <<'PY'
import os
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

root = Path(os.environ['SOURCE_DIR'])
data_root = Path(os.environ['DATASETS_DIR'])
metadata = json.loads((data_root / 'packed_metadata.json').read_text())
print(f'PACKED_FORMAT={metadata["format"]} PACKED_ITERATIONS={metadata["iterations"]}', flush=True)
if int(metadata['iterations']) < 92807:
    raise RuntimeError('Packed FineWeb does not cover the time-matched run')
repo = 'DimaTivator/stage3_dense'
paths = {}
for name in ('main.pt', 'worker_0.pt'):
    path = Path(hf_hub_download(
        repo_id=repo,
        filename=f'adamw_fp8_states/pre_decay/{name}',
        local_dir=root,
    ))
    paths[name] = path
    print(f'SOURCE_FILE={path} BYTES={path.stat().st_size}', flush=True)

main = torch.load(paths['main.pt'], map_location='cpu', mmap=True, weights_only=False)
worker = torch.load(paths['worker_0.pt'], map_location='cpu', weights_only=False)
iteration = int(main['itr'])
reader = worker.get('train_reader_state')
print(f'SOURCE_ITER={iteration}', flush=True)
print(f'SOURCE_SCHEDULER={main.get("scheduler")}', flush=True)
print(f'SOURCE_WORKER_KEYS={list(worker)}', flush=True)
if isinstance(reader, dict):
    print(f'SOURCE_READER_TYPE={reader.get("reader_type")}', flush=True)
    print(f'SOURCE_READER_RANK={reader.get("rank")}', flush=True)
    print(f'SOURCE_READER_STEP={reader.get("step")}', flush=True)
    print(f'SOURCE_READER_BATCH={reader.get("batch_size")}', flush=True)
    print(f'SOURCE_READER_STREAM_KEYS={list(reader.get("stream_state", {}))}', flush=True)
    print(f'SOURCE_READER_STREAM_RANK={reader.get("stream_state", {}).get("rank")}', flush=True)
    print(f'SOURCE_READER_STREAM_WORLD={reader.get("stream_state", {}).get("world_size")}', flush=True)
    print(f'SOURCE_READER_MANIFEST={reader.get("stream_state", {}).get("manifest_fingerprint")}', flush=True)
    print(f'SOURCE_READER_SPLIT={reader.get("stream_state", {}).get("split_plan_fingerprint")}', flush=True)
else:
    print(f'SOURCE_READER_TYPE={type(reader).__name__}', flush=True)
print(f'SOURCE_OPTIMIZER_KEYS={list(main["optimizer"])}', flush=True)
groups = main['optimizer']['param_groups']
print(f'SOURCE_OPTIMIZER_GROUPS={len(groups)}', flush=True)
if groups:
    group_meta = {k: v for k, v in groups[0].items() if k != 'params'}
    print(f'SOURCE_FIRST_GROUP_META={group_meta}', flush=True)
state = main['optimizer']['state']
if state:
    first = next(iter(state.values()))
    print(f'SOURCE_FIRST_STATE_FIELDS={list(first)}', flush=True)
if iteration != 67911:
    raise RuntimeError(f'Expected pre-decay iteration 67911, found {iteration}')
if not isinstance(reader, dict) or int(reader.get('step', -1)) != iteration * 4:
    raise RuntimeError('Missing or inconsistent FineWeb reader cursor')
print('ADAMW_SOURCE_AUDIT_OK', flush=True)
PY
