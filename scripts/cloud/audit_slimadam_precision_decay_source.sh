#!/usr/bin/env bash
set -euo pipefail
echo "AUDIT_UTC=$(date -u --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
python - <<'PY'
import json
import os
from pathlib import Path

import torch

sources = [Path(os.environ['SOURCE_CKPT'])] if os.environ.get('SOURCE_CKPT') else [
    Path(root) / '2xChinchilla_500M_slimadam_wd1e-4_precisions_2gpu' /
    f'llama500M_slim_adam_{precision}_wd1e-4_2xC_warmup2000_2gpu/ckpts/67911'
    for precision, root in (
        ('bf16', '/home/jovyan/dimativator/500m-slimadam-precisions-20261002'),
        ('fp8_act', '/workspace-SR006.nfs2/dimativator/500m-slimadam-precisions-20261002'),
    )
]
workers = int(os.environ.get('CHECKPOINT_WORKERS', '2'))
acc_steps = int(os.environ.get('CHECKPOINT_ACC_STEPS', '8'))
for source in sources:
    main = torch.load(source / 'main.pt', map_location='cpu', mmap=True, weights_only=False)
    assert int(main['itr']) == 67911, f'Unexpected checkpoint iteration: {source}'
    assert main.get('model') and main.get('optimizer'), f'Incomplete main checkpoint: {source}'
    cursors = []
    for rank in range(workers):
        worker = torch.load(source / f'worker_{rank}.pt', map_location='cpu', weights_only=False)
        reader = worker['train_reader_state']
        assert int(reader['rank']) == rank, f'Wrong reader rank: {source}'
        assert int(reader['step']) == 67911 * acc_steps, f'Wrong reader cursor: {source}'
        assert all(name in worker for name in ('rng_torch_cpu', 'rng_torch_gpu', 'rng_np', 'rng_python'))
        cursors.append(int(reader['step']))
    print('DECAY_SOURCE_AUDIT_OK', json.dumps({'path': str(source), 'itr': 67911, 'reader_steps': cursors}), flush=True)
    del main
PY
