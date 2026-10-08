import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config.base import parse_args
from distributed.ddp import DataParallelDistributedBackend

spec = importlib.util.spec_from_file_location("cloud_campaign", ROOT / "scripts/cloud/sign_ademamix_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CloudSignCampaignTest(unittest.TestCase):
    def parsed(self, size, target, world, **kwargs):
        command = campaign.command(size, "2e-3", target, "trial", "group",
                                   Path("/results"), Path("/packed"), world, **kwargs)
        return parse_args(argparse.ArgumentParser(), command[2:], None)

    def test_grid_and_global_batch(self):
        self.assertEqual(set(sum(campaign.LR_QUEUES.values(), [])), {"2e-3", "1e-3", "5e-4", "1e-4"})
        for world in (1, 2):
            args = self.parsed("257m", 39250, world)
            backend = DataParallelDistributedBackend.__new__(DataParallelDistributedBackend)
            backend.local_rank = 0
            backend.get_world_size = lambda: world
            args = backend.get_adjusted_args_for_process(args)
            self.assertEqual(world * args.batch_size * args.acc_steps, 128)
            self.assertEqual(args.fineweb_replay_world_size, 2 if world == 1 else 1)
            self.assertTrue(args.no_local_save)
            self.assertEqual(args.wandb_project, "fp8-pretrain")

    def test_decay_keeps_trunk_momentum_horizon(self):
        for target, checkpoint in [(39250, 35325), (78500, 70650)]:
            args = self.parsed("257m", target, 2, resume=Path(f"/checkpoint/{checkpoint}"))
            self.assertEqual(args.ademamix_sign_alpha_warmup_steps, 157000)
            self.assertEqual(args.ademamix_sign_beta3_warmup_steps, 157000)
            self.assertTrue(args.decay_from_checkpoint)
            self.assertEqual(args.wsd_fract_decay, 1)
            self.assertEqual(args.warmup_steps, 0)
            self.assertTrue(args.no_local_save)
        args = self.parsed("500m", 75457, 2, resume=Path("/checkpoint/67911"))
        self.assertEqual(args.ademamix_sign_alpha_warmup_steps, 150914)
        self.assertEqual(args.weight_decay, 1e-4)

    def test_only_required_checkpoints(self):
        args = self.parsed("257m", 157000, 2, milestones=(35325, 70650))
        self.assertEqual(args.inter_ckpts, [35325, 70650])
        self.assertEqual(args.latest_ckpt_interval, 0)
        self.assertEqual(args.permanent_ckpt_interval, 0)
        self.assertFalse(args.no_local_save)

    def test_cleanup_preserves_logs_and_rejects_outside_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            experiment = output / "group/trial"
            (experiment / "ckpts/35325").mkdir(parents=True)
            (experiment / "ckpts/35325/main.pt").write_bytes(b"test")
            metrics = experiment / "metrics.jsonl"
            metrics.write_text("saved metrics\n")
            campaign.cleanup_checkpoint(output, experiment)
            self.assertFalse((experiment / "ckpts").exists())
            self.assertEqual(metrics.read_text(), "saved metrics\n")
            (root / "outside/ckpts").mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                campaign.cleanup_checkpoint(output, root / "outside")

    def test_numerical_failure_detection_ignores_partial_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text('{"iter": 50, "train/loss": 3.0}\n{"iter":')
            self.assertIsNone(campaign.nonfinite_loss(path))
            path.write_text('{"iter": 50, "train/loss": 3.0}\n{"iter": 100, "train/loss": NaN}\n')
            self.assertEqual(campaign.nonfinite_loss(path)["step"], 100)

    def test_strict_logger_overflow_is_numerical_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.jsonl"
            log = Path(temporary) / "rank0.log"
            metrics.write_text('{"iter": 20250, "train/loss": 253.5}\n')
            log.write_text("_log_local_metric\nValueError: Out of range float values are not JSON compliant: inf")
            self.assertEqual(campaign.nonfinite_loss(metrics, log)["value"], "nonfinite")
            log.write_text("RuntimeError: CUDA out of memory")
            self.assertIsNone(campaign.nonfinite_loss(metrics, log))
            log.write_text("ValueError: Out of range float values are not JSON compliant: inf")
            self.assertIsNone(campaign.nonfinite_loss(metrics, log))

    def test_two_gpu_single_worker_uses_torchrun(self):
        launch = [sys.executable, "src/main.py", "--experiment-name", "trial"]
        actual = campaign.training_launcher(launch, world=2, mpi_size=1)
        self.assertEqual(actual[:5], [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                     "--nproc_per_node=2"])
        self.assertEqual(actual[5:], launch[1:])
        self.assertEqual(campaign.training_launcher(launch, world=1, mpi_size=1), launch)
        self.assertEqual(campaign.training_launcher(launch, world=2, mpi_size=2), launch)


if __name__ == "__main__":
    unittest.main()
