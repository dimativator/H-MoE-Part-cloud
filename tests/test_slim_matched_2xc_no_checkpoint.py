"""Explicit no-save 2xC launch and packed-budget validation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch


class SlimMatched2XCNoCheckpointTest(TestCase):
    def test_no_checkpoint_2xc_and_legacy_checkpoint_path(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/cloud/run_adaptive_muon.py'
        for disabled in ('1', '0'):
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                    'MODE': 'matched', 'CLOUD_TRAIN_WORLD_SIZE': '4', 'RESULTS_DIR': folder,
                    'CLOUD_DISABLE_CHECKPOINTS': disabled, 'OMPI_COMM_WORLD_SIZE': '4',
                    'OMPI_COMM_WORLD_RANK': '0', 'CAMPAIGN_DATE': '20261103'}):
                spec = importlib.util.spec_from_file_location('campaign_2xc_test', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.final_metric = lambda *args: {'val_loss': 3.0}
                commands = []
                class Process:
                    returncode = 0
                    def poll(self):
                        return 0
                def launch(command, **kwargs):
                    commands.append(command)
                    return Process()
                with patch.object(module.subprocess, 'Popen', side_effect=launch):
                    module.train('slim_normuon_matched', '5e-4', 500, '2xC', 150914, weight_decay='1e-4')
                command = commands[0]
                self.assertEqual(command[command.index('--iterations') + 1], '150914')
                self.assertEqual(command[command.index('--batch-size') + 1], '8')
                self.assertEqual(command[command.index('--acc-steps') + 1], '16')
                self.assertNotIn('--resume-from', command)
                self.assertEqual('--no-local-save' in command, disabled == '1')
                self.assertEqual('--inter-ckpts' in command, disabled != '1')
                marker = json.loads(next((Path(folder) / 'finals').glob('*.json')).read_text())
                self.assertEqual(marker['checkpoint_saving'], disabled != '1')
                self.assertEqual(marker['physical_world_size'], 4)
                self.assertEqual(150914 * 128 * 1024, 19780599808)

    def test_entry_runs_smoke_then_fresh_full_and_rejects_short_data(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/cloud/run_slim_matched_normuon_500m_2xc.py'
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            data = Path(folder)
            metadata = data / 'packed_metadata.json'
            metadata.write_text(json.dumps({'iterations': 150914}))
            campaign = SimpleNamespace(DATA=data, train=Mock())
            fake = SimpleNamespace(campaign=campaign, preflight=Mock())
            with patch.dict(sys.modules, {'run_matched_normuon': fake}):
                spec = importlib.util.spec_from_file_location('slim_2xc_entry_test', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.main()
                self.assertEqual(os.environ['CLOUD_DISABLE_CHECKPOINTS'], '1')
                self.assertEqual(os.environ['CLOUD_TRAIN_WORLD_SIZE'], '4')
                self.assertEqual([c.args for c in campaign.train.call_args_list], [
                    ('slim_normuon_matched', '5e-4', 500, 'matched_smoke', 150914),
                    ('slim_normuon_matched', '5e-4', 500, '2xC', 150914)])
                self.assertTrue(all(c.kwargs == {'weight_decay': '1e-4'} for c in campaign.train.call_args_list))
                metadata.write_text(json.dumps({'iterations': 150913}))
                campaign.train.reset_mock()
                with self.assertRaises(RuntimeError):
                    module.main()
                campaign.train.assert_not_called()
