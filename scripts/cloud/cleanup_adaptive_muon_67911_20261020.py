"""Remove only the two explicitly approved Adaptive Muon checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path('/workspace-SR006.nfs3/dimativator/adaptive-muon-20261011')
GROUP = ROOT / 'adaptive_muon_500M_2xC_20261011'
TARGETS = {
    GROUP / 'normuon_500m_bf16_lr1e-3_wd1e-1_2xC_2gpu/ckpts/67911': 4472542562,
    GROUP / 'muon2_500m_bf16_lr2e-3_wd1e-1_2xC_2gpu/ckpts/67911': 5934339350,
}

def checked(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or path.resolve(strict=True) != path:
        raise RuntimeError(f'Non-canonical or symlink path: {path}')

def identity(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise RuntimeError(f'Not a unique regular file: {path}')
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

def protected():
    result = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d != 'ckpts']
        for name in files:
            if name.endswith(('.log', '.jsonl')):
                p = Path(parent) / name
                s = p.lstat()
                result[str(p)] = (s.st_size, s.st_mtime_ns, s.st_ino)
    if not result:
        raise RuntimeError('No logs/metrics found')
    return result

def main():
    checked(ROOT)
    logs = protected()
    report = {'targets': [], 'protected_files': len(logs), 'backup_verified': False}
    audited = {}
    for target, expected_size in TARGETS.items():
        checked(target)
        files = list(target.iterdir())
        if {p.name for p in files} != {'main.pt', 'worker_0.pt', 'worker_1.pt'}:
            raise RuntimeError(f'Unexpected checkpoint contents: {target}')
        for p in files:
            checked(p)
            before = identity(p)
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if before != identity(p) or (p.name == 'main.pt' and before[2] != expected_size):
                raise RuntimeError(f'Changed checkpoint: {p}')
            audited[p] = before
            report['targets'].append({'path': str(p), 'bytes': before[2], 'sha256': h.hexdigest()})
    if protected() != logs or any(identity(p) != value for p, value in audited.items()):
        raise RuntimeError('Files changed during audit')
    print('ADAPTIVE_CHECKPOINT_AUDIT=' + json.dumps(report), flush=True)
    if os.environ.get('APPLY') != '1':
        return
    receipt = Path('/home/jovyan/dimativator/adaptive-muon-cleanup-20261020.json')
    checked(receipt.parent)
    with receipt.open('x') as f:
        json.dump(report, f, indent=2)
    for target in TARGETS:
        for p in [p for p in audited if p.parent == target]:
            if identity(p) != audited[p]:
                raise RuntimeError(f'File changed before unlink: {p}')
            p.unlink()
        target.rmdir()
        print(f'DELETED_ADAPTIVE_CHECKPOINT={target}', flush=True)
    if protected() != logs or any(os.path.lexists(p) for p in TARGETS):
        raise RuntimeError('Post-delete verification failed')
    report['status'] = 'verified_deleted'
    report['deleted_bytes'] = sum(v[2] for v in audited.values())
    with receipt.open('w') as f:
        json.dump(report, f, indent=2)
    print('ADAPTIVE_CLEANUP_VERIFIED=' + json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
