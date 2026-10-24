import importlib.util
import os
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch


class SlimNorMuon500MCloudTest(TestCase):
    def test_full_then_independent_decay_uses_same_optimizer_and_wd(self):
        repo = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('campaign500', repo / 'scripts/cloud/run_adaptive_muon.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {'MODE': '500m_slim_normuon'}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            module.ROOT = Path(folder)
            module.DATA = Path(folder)
            (module.DATA / 'packed_metadata.json').write_text('{"format":"packed_fineweb_h200_v3","iterations":150914,"world_size":2,"batch_size":16}')
            with patch.object(module.subprocess, 'run'), patch.object(module, 'train') as train:
                module.main()
            calls = train.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertEqual([c.args[:5] for c in calls], [
                ('slim_normuon', '1e-3', 500, 'slim_smoke', 150914),
                ('slim_normuon', '1e-3', 500, '2xC', 150914),
                ('slim_normuon', '1e-3', 500, '1xC_decay', 75457),
            ])
            for call in calls:
                self.assertEqual(call.kwargs['weight_decay'], '1e-4')
            self.assertTrue(str(calls[-1].kwargs['resume']).endswith('/ckpts/67911'))
