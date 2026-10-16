"""Remove only checkpoint payloads from the three user-cancelled Horizon campaigns."""
import json
import os
from pathlib import Path
import shutil

ROOTS = (
    Path('/workspace-SR006.nfs1/dimativator/horizon-adam-257m-lr-20261014/horizon_adam_257m_lr_20261014'),
    Path('/workspace-SR006.nfs1/dimativator/horizon-adam-257m-lr-20261014/horizon_adam_257m_diagnostic_20261016'),
    Path('/workspace-SR006.nfs2/dimativator/horizon-adam16-257m-1xc-20261016/horizon_adam16_257m_1xc_20261016'),
)


def main() -> None:
    apply = os.environ.get('APPLY') == '1'
    targets: list[Path] = []
    before = {}
    for root in ROOTS:
        mount = Path('/') / root.parts[1]
        if not mount.is_dir():
            raise RuntimeError(f'Required mount missing: {mount}')
        before[str(mount)] = shutil.disk_usage(mount).free
        if not root.exists():
            print(json.dumps({'absent_campaign': str(root)}), flush=True)
            continue
        if root.is_symlink() or root.resolve() != root:
            raise RuntimeError(f'Noncanonical campaign: {root}')
        for directory, dirs, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            if any((parent / name).is_symlink() for name in dirs):
                raise RuntimeError(f'Symlink directory under campaign: {parent}')
            for name in files:
                path = parent / name
                if 'ckpts' not in path.relative_to(root).parts:
                    continue
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f'Unsafe checkpoint: {path}')
                if path.suffix not in ('.pt', '.pth', '.safetensors', '.tmp'):
                    print(json.dumps({'preserved_unknown_file': str(path)}), flush=True)
                    continue
                targets.append(path)
    total = 0
    for path in targets:
        size = path.stat().st_size
        print(json.dumps({'checkpoint': str(path), 'bytes': size, 'apply': apply}), flush=True)
        if apply:
            path.unlink()
            if path.exists():
                raise RuntimeError(f'Deletion failed: {path}')
        total += size
    after = {mount: shutil.disk_usage(mount).free for mount in before}
    print(json.dumps({'event': 'cleanup_complete', 'apply': apply,
                      'file_count': len(targets), 'payload_bytes': total,
                      'free_before': before, 'free_after': after}), flush=True)


if __name__ == '__main__':
    main()
