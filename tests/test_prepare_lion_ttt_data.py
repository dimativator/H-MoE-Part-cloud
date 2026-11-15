"""CPU-only checks for private dataset routing and disk accounting."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/cloud'))
spec = importlib.util.spec_from_file_location('prepare_lion_ttt_data', REPO / 'scripts/cloud/prepare_lion_ttt_data.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PrepareDataTests(unittest.TestCase):
    def test_exact_bytes_and_private_metadata_leave_source_unchanged(self):
        target = 160000
        blocks = (target - prep.BASE_STEPS) * 64
        self.assertEqual(prep.required_bytes(target), blocks * 1025 * 2 * 2)
        source = dict(ranks=[dict(rank=rank, blocks=prep.BASE_STEPS * 64,
            bytes=prep.BASE_STEPS * 64 * 2050, segments=[
                dict(file='prefix.uint16', blocks=10, bytes=20500, sha256='prefix'),
                dict(file='/workspace-SR006.nfs2/readonly.uint16', blocks=20, bytes=41000, sha256='readonly')])
            for rank in range(2)], validation=dict(file='validation.uint16', sha256='validation'))
        original = copy.deepcopy(source)
        tails = [dict(rank=rank, file=f'tail{rank}.uint16', blocks=blocks, bytes=blocks * 2050, sha256=f'tail{rank}')
            for rank in range(2)]
        with patch.object(prep, 'sha256_file', return_value='original-metadata'), patch.dict(os.environ, CODE_REVISION='prep-commit'):
            result = prep.private_metadata(source, target, tails)
        self.assertEqual(source, original)
        self.assertEqual(result['iterations'], target)
        for rank, entry in enumerate(result['ranks']):
            self.assertEqual(entry['segments'][0]['file'], str(prep.SOURCE / 'prefix.uint16'))
            self.assertEqual(entry['segments'][1]['file'], '/workspace-SR006.nfs2/readonly.uint16')
            self.assertEqual(entry['segments'][-1], tails[rank])
        self.assertEqual(result['validation']['file'], str(prep.SOURCE / 'validation.uint16'))

    def test_state_requires_exact_stream_boundary_and_rank(self):
        metadata = dict(manifest_fingerprint='manifest', split_plan_fingerprint='split')
        state = dict(rank=1, world_size=2, worker_id=0, num_data_workers=1, split='train',
            emitted_block_count=10054400, manifest_fingerprint='manifest', split_plan_fingerprint='split',
            committed_cursor=dict(assigned_row_group_index=20, row_in_group=64), token_buffer=[1, 50256])
        payload = dict(blocks_written=395904, stream_state=state)
        self.assertIs(prep.validate_state(payload, 1, metadata), state)
        for key, invalid in [('rank', 0), ('emitted_block_count', 10054399), ('world_size', 4)]:
            bad = copy.deepcopy(payload); bad['stream_state'][key] = invalid
            with self.assertRaises(ValueError):
                prep.validate_state(bad, 1, metadata)

    def test_pending_write_must_leave_strict_reserve_and_headroom(self):
        pending = 12345
        snapshot = dict(available_bytes={'/workspace-SR006.nfs3': prep.RESERVE + prep.HEADROOM + pending})
        with patch.object(prep.common, 'space', return_value=snapshot), patch.object(prep.common, 'emit'):
            with self.assertRaises(RuntimeError):
                prep.guard(pending)
            snapshot['available_bytes']['/workspace-SR006.nfs3'] += 1
            self.assertEqual(prep.guard(pending), snapshot)


if __name__ == '__main__':
    unittest.main()
