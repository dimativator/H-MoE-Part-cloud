"""Check an existing checkpoint and cached downstream tasks without downloads."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from audit_figure6a_checkpoint import audit

CHECKPOINT = Path('/workspace-SR006.nfs3/dimativator/figure6a-adamw-fp8-retrain-20261005/figure6a_correct_checkpoint_20261005/llama500M_adamw_fp8_states_1xC_2gpu_global128/ckpts/67911')
EXPECTED = {
    'main.pt': '580067125fd9f5cf11bb62fc072d95ff04a2556d58f77ae764a5c4f2cfcd82ab',
    'worker_0.pt': '5456cfcfab1011651ba5d332ed789be6bfda14c87f0b5b7b7c72500e2da5f406',
    'worker_1.pt': 'f1a05dfe38220a480cd8c697687ff69c76416a18ef64e229c6048d1c2a02a2f9',
}
for name, expected in EXPECTED.items():
    digest = hashlib.sha256()
    with (CHECKPOINT / name).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == expected, name
    print('SOURCE_SHA256_OK', name, flush=True)
main, workers = audit(CHECKPOINT, 67911)
del main, workers
assert shutil.disk_usage('/home/jovyan').free > 6_000_000_000
dataset = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
metadata = json.loads((dataset / 'packed_metadata.json').read_text())
assert metadata['manifest_fingerprint'] == '7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b'
for rank in range(2):
    assert (dataset / f'train_rank{rank}.uint16').is_file()
assert (dataset / 'validation.uint16').is_file()
print('DATA_SOURCE_OK', flush=True)

from data.utils import get_tokenizer
from evals.downstream import DownstreamEvaluator
cfg = SimpleNamespace(tokenizer='gpt2', sequence_length=1024, device='cpu',
                      eval_batch_size=32, downstream_task_group='basic_v2', downstream_tasks=None)
tokenizer = get_tokenizer(cfg)
evaluator = DownstreamEvaluator(cfg, tokenizer, 'gpt2')
evaluator._ensure_initialized()
for runtime in evaluator._task_runtimes:
    print('DOWNSTREAM_CACHE_OK', runtime['task_name'], len(runtime['dataset']), flush=True)
print('FIGURE6A_DECAY_PREFLIGHT_OK', flush=True)
