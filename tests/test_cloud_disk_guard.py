import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from optim.disk_guard import DiskGuard, DiskSpaceEmergency, RESERVE_BYTES, cleanup_checkpoints


def _distributed_emergency_worker(rank, folder):
    import datetime
    import torch.distributed as dist
    root = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'),
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30))
    local = [dict(path=str(root), device=root.stat().st_dev,
                  available_bytes=1 if rank == 1 else RESERVE_BYTES + 1)]
    def cleanup(*args):
        (root / f'cleanup_rank_{rank}').write_text('called')
        return []
    try:
        with patch('optim.disk_guard.available', return_value=local), patch('optim.disk_guard.cleanup_checkpoints', side_effect=cleanup):
            try:
                DiskGuard(root, [root]).check()
            except DiskSpaceEmergency:
                (root / f'stopped_rank_{rank}').write_text('stopped')
    finally:
        dist.destroy_process_group()


class CloudDiskGuardTest(TestCase):
    def test_threshold_and_pending_checkpoint_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            guard = DiskGuard(root, [root])
            def snapshot(free):
                return [dict(path=str(root), available_bytes=free, device=root.stat().st_dev)]
            with patch('optim.disk_guard.available', return_value=snapshot(RESERVE_BYTES + 1)):
                guard.check()
                self.assertFalse(json.loads((root / 'disk_status.json').read_text())['emergency'])
            for free, extra in ((RESERVE_BYTES - 1, 0), (RESERVE_BYTES, 0), (RESERVE_BYTES + 100, 101)):
                with patch('optim.disk_guard.available', return_value=snapshot(free)):
                    with self.assertRaises(DiskSpaceEmergency):
                        guard.check(extra_bytes=extra)

    def test_cleanup_only_checkpoints_until_recovered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'group/run/ckpts/67911'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'main.pt').write_bytes(b'checkpoint')
            metrics = root / 'group/run/metrics.jsonl'
            metrics.write_text('preserve')
            def snapshot(paths):
                return [dict(path=str(root), device=root.stat().st_dev,
                             available_bytes=1 if checkpoint.exists() else RESERVE_BYTES + 1)]
            with patch('optim.disk_guard.available', side_effect=snapshot):
                deleted = cleanup_checkpoints(root, [root])
            self.assertEqual(deleted, [str(checkpoint.resolve())])
            self.assertEqual(metrics.read_text(), 'preserve')

    def test_cleanup_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            external = Path(outside) / '67911'
            external.mkdir()
            (external / 'main.pt').write_bytes(b'protected')
            parent = root / 'group/run/ckpts'
            parent.mkdir(parents=True)
            (parent / '67911').symlink_to(external, target_is_directory=True)
            low = [dict(path=str(root), available_bytes=1, device=root.stat().st_dev)]
            with patch('optim.disk_guard.available', return_value=low):
                self.assertEqual(cleanup_checkpoints(root, [root]), [])
            self.assertEqual((external / 'main.pt').read_bytes(), b'protected')

    def test_missing_telemetry_stops_training(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = DiskGuard(Path(folder), [Path(folder)])
            with patch('optim.disk_guard.available', side_effect=OSError('unavailable')):
                with self.assertRaises(DiskSpaceEmergency):
                    guard.check()

    def test_two_real_ranks_stop_when_only_one_reports_low_space(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(_distributed_emergency_worker, args=(folder,), nprocs=2, join=True)
            root = Path(folder)
            self.assertTrue((root / 'stopped_rank_0').is_file())
            self.assertTrue((root / 'stopped_rank_1').is_file())
            self.assertTrue((root / 'cleanup_rank_0').is_file())
            self.assertFalse((root / 'cleanup_rank_1').exists())
