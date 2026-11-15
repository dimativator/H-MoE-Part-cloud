"""Rebatch the same verified FineWeb prefix for any divisor of global batch128."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]


def prefix_indices(start: int, count: int) -> list[tuple[int, int]]:
    # Original global order: rank0[16],rank1[16], repeated four times per update.
    return [((i % 32) // 16, (i // 32) * 16 + i % 16) for i in range(start, start + count)]


def build_readers(args, *, rank: int, world_size: int) -> dict:
    import numpy as np
    import torch
    from data import fineweb_packed as packed
    from data.fineweb import FineWebValReader
    if rank != 0 or world_size != 1 or args.batch_size * args.acc_steps != 128:
        raise ValueError('Single GPU and effective batch128 required')
    root = Path(args.datasets_dir)
    meta = json.loads((root / 'packed_metadata.json').read_text())
    for key, expected in dict(manifest_fingerprint=packed.EXPECTED_MANIFEST_FINGERPRINT,
        split_plan_fingerprint=packed.EXPECTED_SPLIT_PLAN_FINGERPRINT,
        validation_blocks_sha256=packed.EXPECTED_VAL_BLOCKS_SHA256, sequence_length=1024).items():
        if meta[key] != expected:
            raise RuntimeError('Unexpected source data identity: ' + key)
    proof=json.loads((Path('/workspace-SR006.nfs3/dimativator/lionmuon-time-target-500m-20261115/single_gpu_timing/data_proof.json')).read_text())
    if proof['code_revision']!=os.environ.get('DATA_PROOF_REVISION',os.environ['CODE_REVISION']):raise RuntimeError('Source integrity proof revision mismatch')
    sources, hashes = [], []
    for source in meta['ranks']:
        segment = source['segments'][0]
        path = root / segment['file']
        if path.stat().st_size != segment['bytes']:
            raise RuntimeError('Source segment size mismatch')
        with path.open('rb') as stream:
            preview = stream.read(15360 * 1025 * 2)
        digest = hashlib.sha256(preview).hexdigest()
        expected=next(p for p in proof['sources'] if p['rank']==source['rank'])
        if path.stat().st_mtime_ns!=expected['mtime_ns'] or digest != expected['timing_prefix_sha256']:
            raise RuntimeError('Verified source prefix hash mismatch')
        hashes.append(digest)
        sources.append(np.memmap(path, mode='r', dtype='<u2', shape=(segment['blocks'],1025)))
    class Reader:
        requires_checkpoint_state = False
        batch_size = args.batch_size
        sequence_length = 1024
        step = 0
        def set_step(self, step: int) -> None:
            if step < 0 or step * self.batch_size > 240 * 128:
                raise ValueError('Benchmark prefix step out of range')
            self.step = step
        def sample_batch(self):
            rows = prefix_indices(self.step * self.batch_size, self.batch_size)
            blocks = np.stack([sources[r][i] for r,i in rows]).astype(np.int64)
            self.step += 1
            blocks = torch.from_numpy(blocks)
            return blocks[:,:-1], blocks[:,1:]
    val = meta['validation']; path = root / val['file']
    if path.stat().st_size != val['bytes'] or packed._sha256_file(path) != val['sha256']:
        raise RuntimeError('Validation hash mismatch')
    blocks = torch.from_numpy(np.fromfile(path,dtype='<u2').astype(np.int64).reshape(-1,1025))
    print('VERIFIED_TIMING_DATA=' + json.dumps(dict(prefix_sha256=hashes, microbatch=args.batch_size,
        accumulation=args.acc_steps, global_batch=128, source_order='rank0[16],rank1[16] four times per optimizer step')),flush=True)
    return dict(train=Reader(),val=FineWebValReader(blocks,args.eval_batch_size,1024))


if __name__ == '__main__':
    from data import fineweb_packed
    fineweb_packed.build_packed_fineweb_readers = build_readers
    sys.argv[0] = str(REPO / 'src/main.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
