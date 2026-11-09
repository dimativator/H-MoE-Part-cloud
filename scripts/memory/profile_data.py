"""Fixed verified packed-token subset for single-GPU memory profiling only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data.fineweb import FineWebValReader
from data import fineweb_packed as packed


def build_readers(args, *, rank, world_size):
    assert rank == 0 and world_size == 1
    root = Path(args.datasets_dir)
    meta = json.loads((root / 'packed_metadata.json').read_text())
    assert meta['format'] in packed.EXPECTED_FORMATS
    assert meta['manifest_fingerprint'] == packed.EXPECTED_MANIFEST_FINGERPRINT
    assert meta['split_plan_fingerprint'] == packed.EXPECTED_SPLIT_PLAN_FINGERPRINT
    assert meta['sequence_length'] == args.sequence_length
    source = meta['ranks'][0]
    segment = source.get('segments', [source])[0]
    path = root / segment['file']
    assert path.stat().st_size == segment['bytes']
    count = source['preview_blocks']
    with path.open('rb') as stream:
        raw = stream.read(count * (args.sequence_length + 1) * 2)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == source['preview_sha256']
    blocks = torch.from_numpy(np.frombuffer(raw, dtype='<u2').astype(np.int64).reshape(count, -1))
    train = FineWebValReader(blocks, args.batch_size, args.sequence_length)
    train.requires_checkpoint_state = False
    val_meta = meta['validation']
    val_path = root / val_meta['file']
    assert val_path.stat().st_size == val_meta['bytes']
    assert packed._sha256_file(val_path) == val_meta['sha256']
    val_blocks = torch.from_numpy(np.fromfile(val_path, dtype='<u2').astype(np.int64).reshape(-1, args.sequence_length + 1))
    val = FineWebValReader(val_blocks, args.eval_batch_size, args.sequence_length)
    print('PROFILE_DATA=' + json.dumps(dict(source_rank=0, preview_blocks=count, sha256=digest,
          batch_size=args.batch_size, validation_sha256=val_meta['sha256'])), flush=True)
    return {'train': train, 'val': val}
