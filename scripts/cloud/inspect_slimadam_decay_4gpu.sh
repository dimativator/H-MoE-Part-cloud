#!/usr/bin/env bash
set -euo pipefail

readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud
readonly data_dir=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly result_root=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states

df -h "${data_dir}"
find "${data_dir}" -maxdepth 1 -type f \
    \( -name '*continuation*state*.json' -o -name '*third*' \) \
    -printf 'PACKED_AUX_FILE %f %s bytes\n' | sort

for scale in 1 2 4; do
    for rank in 0 1 2 3; do
        log_file="${log_dir}/257m_slim_adam_fp8_states_${scale}xC_decay_cloud_4gpu_rank${rank}.log"
        echo "DECAY_LOG scale=${scale} rank=${rank} path=${log_file}"
        if [[ ! -f "${log_file}" ]]; then
            echo "DECAY_LOG_MISSING scale=${scale} rank=${rank}"
            continue
        fi
        echo "DECAY_LOG_BYTES=$(stat -c %s "${log_file}")"
        tr '\r' '\n' < "${log_file}" | grep -aE 'Train: Iter=|>Eval:|DECAY_EXIT|Traceback|Error|Exception|No space left|Killed|OutOfMemory|CUDA out of memory' | tail -n 12 || true
        if [[ "${scale}" == 4 ]]; then
            echo "DECAY_LOG_TAIL scale=4 rank=${rank}"
            tr '\r' '\n' < "${log_file}" | tail -n 55
        fi
    done
done

latest_full=$(find "${log_dir}" -maxdepth 1 -type f \
    -name '257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100_full_*.log' \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
echo "FULL_LOG=${latest_full:-missing}"
if [[ -n "${latest_full}" ]]; then
    tr '\r' '\n' < "${latest_full}" | grep -aE 'Train: Iter=|>Eval:|Traceback|Error|TRAIN_EXIT' | tail -n 25 || true
fi

python - "${data_dir}/packed_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
batch_size = int(metadata["batch_size"])
print(f"PACKED_SOURCE world_size={metadata['world_size']} batch_size={batch_size}")
for rank in metadata["ranks"]:
    steps = int(rank["blocks"]) // batch_size
    print(f"PACKED_CAPACITY rank={rank['rank']} blocks={rank['blocks']} source_microsteps={steps}")
PY

for scale in 1 2 4; do
    ckpt_root="${result_root}/257m_slim_adam_fp8_states_${scale}xC_decay_cloud_4gpu/ckpts"
    echo "DECAY_CHECKPOINTS scale=${scale}"
    if [[ -d "${ckpt_root}" ]]; then
        find "${ckpt_root}" -maxdepth 2 -type f -name 'main.pt' -printf '%P %s bytes\n' | sort
        if [[ -f "${ckpt_root}/latest/main.pt" ]]; then
            python - "${ckpt_root}/latest" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
main = torch.load(root / "main.pt", map_location="cpu", mmap=True, weights_only=False)
print(f"LATEST_DECAY_CHECKPOINT path={root} iteration={main.get('itr')}")
for worker_path in sorted(root.glob("worker_*.pt")):
    worker = torch.load(worker_path, map_location="cpu", weights_only=False)
    reader = worker.get("train_reader_state", {})
    print(f"LATEST_DECAY_WORKER name={worker_path.name} reader_type={reader.get('reader_type')} step={reader.get('step')}")
PY
        fi
    fi
done
