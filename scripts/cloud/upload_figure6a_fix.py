"""Audit and atomically publish the genuine Figure 6(a) checkpoint."""
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, CommitOperationAdd
from audit_figure6a_checkpoint import audit


def main() -> None:
    root = Path(os.environ['RESULTS_DIR'])
    directory = root / 'figure6a_correct_checkpoint_20261005/llama500M_adamw_fp8_states_1xC_2gpu_global128/ckpts/67911'
    marker = f'CHECKPOINT_AUDIT_OK path={directory} itr=67911'
    assert marker in (root / 'logs/full_rank0.log').read_text(), 'Missing post-barrier training audit'
    checkpoint, workers = audit(directory, 67911)
    for state in checkpoint['optimizer']['state'].values():
        assert state['exp_avg'].dtype == torch.float8_e4m3fn
        assert state['exp_avg_sq'].dtype == torch.float8_e4m3fn
    del checkpoint, workers
    gc.collect()
    hashes = {}
    for name in ('main.pt', 'worker_0.pt', 'worker_1.pt'):
        path = directory / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        hashes[name] = {'sha256': digest.hexdigest(), 'size': path.stat().st_size}
    repo, prefix = 'DimaTivator/stage3_dense', 'adamw_fp8_states_fix/pre_decay'
    api = HfApi(token=os.environ['HF_TOKEN'])
    revision = api.model_info(repo).sha
    existing = {f.rfilename: f for f in api.model_info(repo, revision=revision, files_metadata=True).siblings if f.rfilename.startswith(prefix + '/')}

    def verify(files: dict) -> None:
        for name, metadata in hashes.items():
            remote = files[prefix + '/' + name]
            assert remote.size == metadata['size'], f'Remote size mismatch: {name}'
            assert remote.lfs is not None and remote.lfs.sha256 == metadata['sha256'], f'Remote hash mismatch: {name}'

    if existing:
        verify(existing)
        print('HF_UPLOAD_ALREADY_VERIFIED revision=' + revision, flush=True)
        return
    provenance = {
        'training_job': 'lm-mpi-job-047f3524-27e8-441d-95cf-294e2017360d',
        'training_commit': 'dde5c99', 'iteration': 67911, 'scheduler_last_epoch': 67911,
        'global_batch': 128, 'world_size': 2, 'rank_batch': 16, 'rank_accumulation': 4,
        'sequence_length': 1024, 'reader_steps': [271644, 271644],
        'dataset_fingerprint': '7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b',
        'optimizer': 'AdamW FP8 E4M3/E4M3 expand qgroup128',
        'lr': 1e-3, 'weight_decay': 1e-4, 'betas': [0.9, 0.99], 'clip': 1,
        'warmup': 7000, 'decay_steps': 7546, 'target_iteration': 75457,
        'audit': 'CHECKPOINT_AUDIT_OK; genuine worker states; all FP8 moments validated',
        'files': hashes,
    }
    operations = [CommitOperationAdd(path_in_repo=prefix + '/' + name, path_or_fileobj=str(directory / name)) for name in hashes]
    operations.append(CommitOperationAdd(path_in_repo=prefix + '/provenance_audit.json', path_or_fileobj=json.dumps(provenance, indent=2).encode()))
    result = api.create_commit(repo_id=repo, repo_type='model', operations=operations, parent_commit=revision, commit_message='Add audited Figure6a FP8 AdamW global128 fixed pre-decay checkpoint')
    remote = {f.rfilename: f for f in api.model_info(repo, revision=result.oid, files_metadata=True).siblings}
    verify(remote)
    print('HF_UPLOAD_VERIFIED revision=' + result.oid, flush=True)
    print(json.dumps(hashes), flush=True)


if __name__ == '__main__':
    main()
