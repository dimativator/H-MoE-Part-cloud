"""Four-GPU launcher preserves the batch/token budget and no-save contract."""
import importlib.util
import os
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch


class SlimMatched500MFourGPUTest(TestCase):
    def test_mpi_and_torchrun_commands_preserve_global_batch(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/cloud/run_adaptive_muon.py'
        for mpi_world in (1, 4):
            with tempfile.TemporaryDirectory() as folder:
                env = {'MODE': 'matched', 'CLOUD_TRAIN_WORLD_SIZE': '4', 'RESULTS_DIR': folder,
                       'OMPI_COMM_WORLD_SIZE': str(mpi_world), 'OMPI_COMM_WORLD_RANK': '0',
                       'CAMPAIGN_DATE': '20261102'}
                with patch.dict(os.environ, env):
                    spec = importlib.util.spec_from_file_location('four_gpu_campaign', path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                module.final_metric = lambda *args: {'val_loss': 3.0}
                commands = []
                class Process:
                    returncode = 0
                    def poll(self):
                        return 0
                def launch(command, **kwargs):
                    commands.append((command, kwargs['env']))
                    return Process()
                with patch.object(module.subprocess, 'Popen', side_effect=launch):
                    module.train('slim_normuon_matched', '5e-4', 500, 'matched_smoke', 75457, weight_decay='1e-4')
                    module.train('slim_normuon_matched', '5e-4', 500, '1xC', 75457, weight_decay='1e-4')
                for command, launched_env in commands:
                    for flag, value in [('--batch-size', '8'), ('--acc-steps', '16'), ('--iterations', '75457'),
                                        ('--weight-decay', '1e-4'), ('--muon_adamw_lr_scale', '2.0')]:
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertIn('--no-local-save', command)
                    self.assertNotIn('--resume-from', command)
                    self.assertIn('_4gpu', command[command.index('--experiment-name') + 1])
                    if mpi_world == 4:
                        self.assertEqual(launched_env['WORLD_SIZE'], '4')
                    else:
                        self.assertIn('--nproc_per_node=4', command)
                self.assertIn('--early-stop-iteration', commands[0][0])
                self.assertNotIn('--early-stop-iteration', commands[1][0])
