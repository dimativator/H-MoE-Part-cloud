from argparse import ArgumentParser
from copy import deepcopy
from datetime import timedelta
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch

from config.base import parse_args
from optim.horizon_adam import HorizonAdam, build_horizon_adam
from optim.horizon_codec import decode_blocks, encode_blocks, fit_codec, pack_codes, start_probe_window, unpack_codes, update_probes


def _ddp_worker(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
                                        init_method=(Path(directory) / "rendezvous").as_uri(),
                                        timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(4)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(64, 64))
        opt = HorizonAdam(model.parameters(), horizon=2)
        for step in range(5):
            torch.manual_seed(step * 100 + rank)
            opt.zero_grad(set_to_none=True)
            model(torch.randn(3, 64)).square().mean().backward()
            opt.step()
        torch.save(dict(model=model.module.state_dict(), optimizer=opt.state_dict()),
                   Path(directory) / f"rank_{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


class HorizonCodecTest(TestCase):
    def test_bit_packing_roundtrip_all_supported_widths(self):
        for bits in range(2, 9):
            codes = torch.arange(1 << bits).repeat(3).reshape(3, -1)
            if codes.shape[-1] < 8:
                codes = codes.repeat(1, 2)
            packed = pack_codes(codes, bits)
            self.assertEqual(packed.dtype, torch.uint8)
            self.assertEqual(packed.numel() * 8, codes.numel() * bits)
            torch.testing.assert_close(unpack_codes(packed, bits), codes.float(), rtol=0, atol=0)

    def test_codec_inverse_and_utility_bit_allocation(self):
        source = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        codec = fit_codec(source, torch.diag(torch.tensor([1., 100.])), 8, 1e-4)
        torch.testing.assert_close(codec["decoder"] @ codec["encoder"], torch.eye(2))
        self.assertEqual(codec["bits"], (6, 2))
        self.assertGreater(codec["spectrum"][0], codec["spectrum"][1])
        for total_bits in range(4, 17):
            fitted = fit_codec(source, torch.zeros(2, 2), total_bits, 1e-4)
            self.assertEqual(sum(fitted["bits"]), total_bits)
            self.assertTrue(all(2 <= width <= 8 for width in fitted["bits"]))

    def test_stochastic_rounding_preserves_quantizer_endpoints(self):
        values = torch.tensor([-1., 1.]).repeat(8).view(1, 8, 2)
        values[:, 4:] *= -1
        with patch("optim.horizon_codec.rounding_uniform", return_value=torch.full((8,), 1 - 2 ** -24)):
            packed, centers, scales = encode_blocks(values, torch.eye(2), (8, 8), True, 0, 0)
        decoded = decode_blocks(packed, centers, scales, torch.eye(2), (8, 8), 8)
        torch.testing.assert_close(decoded, values, rtol=0, atol=0)

    def test_horizon_metric_matches_autograd_unrolled_adam(self):
        initial_m = torch.tensor([0.1, -0.2, 0.05])
        initial_v = torch.tensor([0.01, 0.03, 0.02])
        gradients = [torch.tensor([0.3, -0.1, 0.2]), torch.tensor([-0.2, 0.4, 0.1]),
                     torch.tensor([0.05, -0.3, 0.4])]
        rates = [0.02, 0.01, 0.003]
        group = dict(betas=(0.8, 0.95), eps=1e-6, v_floor=1e-30, horizon=3,
                     total_bits=8, ridge=1e-4, lr=rates[0])
        state = dict(step=5, probe_m=initial_m.clone(), probe_v=initial_v.clone())
        start_probe_window(state, group["v_floor"])

        def replay(perturbation):
            m = initial_m.double() + perturbation[:, 0]
            v = initial_v.double() * perturbation[:, 1].exp()
            updates = []
            for k, (gradient, lr) in enumerate(zip(gradients, rates), start=6):
                m = 0.8 * m + 0.2 * gradient.double()
                v = 0.95 * v + 0.05 * gradient.double().square()
                updates.append(lr * (m / (1 - 0.8 ** k)) / ((v / (1 - 0.95 ** k)).sqrt() + group["eps"]))
            return torch.stack(updates)

        jacobian = torch.autograd.functional.jacobian(replay, torch.zeros(3, 2, dtype=torch.float64))
        expected = sum(jacobian[k, i, i, :, None] @ jacobian[k, i, i, None, :]
                       for k in range(3) for i in range(3)) / 3
        with patch("optim.horizon_codec.fit_codec", wraps=fit_codec) as fitted:
            for gradient, lr in zip(gradients, rates):
                state["step"] += 1
                group["lr"] = lr
                update_probes(state, gradient, group)
            self.assertEqual(fitted.call_count, 1)
            torch.testing.assert_close(fitted.call_args.args[1].double(), expected, rtol=2e-6, atol=1e-10)


class HorizonAdamTest(TestCase):
    def test_cli(self):
        args = parse_args(ArgumentParser(), ["--experiment-name", "horizon-test", "--opt", "horizon_adam",
                                           "--horizon-adam-horizon", "8", "--horizon-adam-bits", "12"], None)
        self.assertEqual(args.horizon_adam_horizon, 8)
        self.assertEqual(args.horizon_adam_bits, 12)
        self.assertFalse(args.horizon_adam_nearest_rounding)
        opt = build_horizon_adam([torch.nn.Parameter(torch.ones(8))], args)
        self.assertEqual(opt.param_groups[0]["total_bits"], 12)
        self.assertEqual(opt.param_groups[0]["horizon"], 8)
        args.fp8_optim = True
        with self.assertRaisesRegex(ValueError, "omit --fp8-optim"):
            build_horizon_adam([torch.nn.Parameter(torch.ones(8))], args)

    def test_dense_fallback_matches_adamw_with_groups_and_scheduler(self):
        torch.manual_seed(1)
        parameters = [torch.nn.Parameter(torch.randn(19)), torch.nn.Parameter(torch.randn(3, 7))]
        references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        kwargs = dict(lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1)
        opt = HorizonAdam([dict(params=[parameters[0]]), dict(params=[parameters[1]], weight_decay=0)], **kwargs)
        ref = torch.optim.AdamW([dict(params=[references[0]]), dict(params=[references[1]], weight_decay=0)],
                               **kwargs, foreach=False)
        schedules = [torch.optim.lr_scheduler.StepLR(item, 2, 0.5) for item in (opt, ref)]
        for _ in range(7):
            for parameter, reference in zip(parameters, references):
                gradient = torch.randn_like(parameter)
                parameter.grad, reference.grad = gradient.clone(), gradient.clone()
            opt.step()
            ref.step()
            for parameter, reference in zip(parameters, references):
                torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-7)
            for schedule in schedules:
                schedule.step()

    def test_compressed_first_update_matches_adamw(self):
        for size in (3, 257, 4096):
            parameter = torch.nn.Parameter(torch.randn(size))
            reference = torch.nn.Parameter(parameter.detach().clone())
            kwargs = dict(lr=0.003, betas=(0.9, 0.99), weight_decay=0.1)
            opt = HorizonAdam([parameter], min_size=0, **kwargs)
            ref = torch.optim.AdamW([reference], foreach=False, **kwargs)
            gradient = torch.randn_like(parameter)
            parameter.grad, reference.grad = gradient.clone(), gradient.clone()
            opt.step()
            ref.step()
            torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    def test_chunk_invariance_across_codec_changes_and_partial_blocks(self):
        torch.manual_seed(5)
        first = torch.nn.Parameter(torch.randn(257))
        second = torch.nn.Parameter(first.detach().clone())
        kwargs = dict(min_size=0, horizon=2, block_size=16, probe_size=8, lr=0.003)
        opts = [HorizonAdam([first], chunk_size=32, **kwargs), HorizonAdam([second], chunk_size=128, **kwargs)]
        for _ in range(9):
            gradient = torch.randn_like(first)
            for parameter, opt in zip((first, second), opts):
                parameter.grad = gradient.clone()
                opt.step()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(opts[0].state[first]["packed"], opts[1].state[second]["packed"], rtol=0, atol=0)
        self.assertEqual(opts[0].state[first]["codec_updates"], 4)

    def test_resume_exact_midwindow_and_preserve_bfloat16_metadata(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(4)
            parameter = torch.nn.Parameter(torch.randn(513, dtype=dtype))
            opt = HorizonAdam([parameter], min_size=0, horizon=3, block_size=32, chunk_size=128)
            schedule = torch.optim.lr_scheduler.StepLR(opt, 2, 0.8)
            gradients = [torch.randn_like(parameter) for _ in range(10)]
            for gradient in gradients[:5]:
                parameter.grad = gradient
                opt.step()
                schedule.step()
            checkpoint = io.BytesIO()
            torch.save(dict(parameter=parameter.detach(), optimizer=opt.state_dict(), scheduler=schedule.state_dict()), checkpoint)
            checkpoint.seek(0)
            saved = torch.load(checkpoint, weights_only=True)
            resumed = torch.nn.Parameter(saved["parameter"].clone())
            resumed_opt = HorizonAdam([resumed])
            resumed_schedule = torch.optim.lr_scheduler.StepLR(resumed_opt, 2, 0.8)
            resumed_opt.load_state_dict(saved["optimizer"])
            resumed_schedule.load_state_dict(saved["scheduler"])
            self.assertEqual(resumed_opt.state[resumed]["packed"].dtype, torch.uint8)
            self.assertEqual(resumed_opt.state[resumed]["probe_indices"].dtype, torch.int64)
            self.assertEqual(resumed_opt.state[resumed]["probe_m"].dtype, torch.float32)
            for gradient in gradients[5:]:
                for p, optimizer, scheduler in ((parameter, opt, schedule), (resumed, resumed_opt, resumed_schedule)):
                    p.grad = gradient.clone()
                    optimizer.step()
                    scheduler.step()
                torch.testing.assert_close(parameter, resumed, rtol=0, atol=0)
            for key, value in opt.state[parameter].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, resumed_opt.state[resumed][key], rtol=0, atol=0)
                else:
                    self.assertEqual(value, resumed_opt.state[resumed][key])

    def test_actual_memory_and_bounded_working_chunks(self):
        parameter = torch.nn.Parameter(torch.ones(262144))
        opt = HorizonAdam([parameter], chunk_size=8192, horizon=1)
        with patch.object(opt, "_update_chunk", wraps=opt._update_chunk) as update:
            for _ in range(3):
                parameter.grad = torch.randn_like(parameter)
                opt.step()
            self.assertTrue(all(call.args[0].numel() <= 8192 for call in update.call_args_list))
        state = opt.state[parameter]
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)
        self.assertEqual(state["packed"].numel(), parameter.numel())
        self.assertLess(opt.state_size_bytes() / parameter.numel(), 1.15)

    def test_rounding_is_independent_of_rank_training_rng(self):
        first = torch.nn.Parameter(torch.linspace(-1, 1, 128))
        second = torch.nn.Parameter(first.detach().clone())
        opts = [HorizonAdam([p], min_size=0, horizon=2) for p in (first, second)]
        for step in range(5):
            gradient = torch.cos(torch.arange(128).float() + step)
            for rank, (parameter, opt) in enumerate(zip((first, second), opts)):
                torch.manual_seed(100 * step + rank)
                before = torch.random.get_rng_state().clone()
                parameter.grad = gradient.clone()
                opt.step()
                torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    @skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "requires Gloo")
    def test_ddp_replicas_keep_identical_compressed_states(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(_ddp_worker, args=(directory,), nprocs=2, join=True)
            first, second = [torch.load(Path(directory) / f"rank_{rank}.pt", weights_only=True) for rank in (0, 1)]
        for key in first["model"]:
            torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
        for key, state in first["optimizer"]["state"].items():
            for name, value in state.items():
                other = second["optimizer"]["state"][key][name]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    self.assertEqual(value, other)

    def test_zero_gradients_missing_gradients_and_closure(self):
        for min_size in (0, 4096):
            parameter = torch.nn.Parameter(torch.ones(64))
            missing = torch.nn.Parameter(torch.ones(64))
            opt = HorizonAdam([parameter, missing], min_size=min_size, horizon=2, lr=0.01, weight_decay=0.1)
            for _ in range(8):
                parameter.grad = torch.zeros_like(parameter)
                opt.step()
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.999 ** 8))
            self.assertNotIn(missing, opt.state)
            self.assertTrue(torch.isfinite(parameter).all())
            def closure():
                opt.zero_grad()
                loss = parameter.square().mean()
                loss.backward()
                return loss
            self.assertIsInstance(opt.step(closure), torch.Tensor)

    def test_compressed_optimizer_reduces_quadratic_loss(self):
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.randn(8192))
        opt = HorizonAdam([parameter], lr=0.03, horizon=4, weight_decay=0)
        initial = parameter.square().mean().item()
        for _ in range(40):
            opt.zero_grad()
            parameter.square().mean().backward()
            opt.step()
        self.assertLess(parameter.square().mean().item(), initial * 0.4)

    def test_moment_constraint_prevents_tiny_variance_update_explosion(self):
        for background in (0., 1e-12):
            for stochastic in (False, True):
                parameter = torch.nn.Parameter(torch.zeros(4096))
                opt = HorizonAdam([parameter], lr=0.001, weight_decay=0,
                                  horizon=4, stochastic_rounding=stochastic)
                gradient = torch.full_like(parameter, background)
                gradient[::256] = 1
                for step in range(12):
                    parameter.grad = gradient.clone()
                    opt.step()
                    if step < 4:
                        self.assertLess(parameter.view(-1, 256)[:, 1:].abs().max().item(), 1e-5)
                self.assertTrue(torch.isfinite(parameter).all())
                # A learned lossy codec can move these coordinates, but cannot
                # produce the hundreds-to-thousands-sized updates of the bug.
                self.assertLess(parameter.abs().max().item(), 0.02)

    def test_small_epsilon_and_variance_do_not_underflow_probe_derivative(self):
        parameter = torch.nn.Parameter(torch.zeros(4096))
        opt = HorizonAdam([parameter], v_floor=1e-38, eps=1e-16, horizon=2)
        for _ in range(6):
            parameter.grad = torch.zeros_like(parameter)
            opt.step()
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertTrue(torch.isfinite(opt.state[parameter]["spectrum"]).all())

    def test_moment_bound_matches_weighted_cauchy_schwarz(self):
        for betas in ((0.9, 0.99), (0.99, 0.9), (0., 0.9), (0., 0.)):
            parameter = torch.nn.Parameter(torch.zeros(64))
            opt = HorizonAdam([parameter], min_size=0, betas=betas, horizon=2)
            for step in range(1, 6):
                parameter.grad = torch.randn_like(parameter)
                opt.step()
                if betas[1] > 0:
                    expected = (1 - betas[0]) ** 2 / (1 - betas[1]) * sum(
                        (betas[0] ** 2 / betas[1]) ** k for k in range(step))
                else:
                    expected = 1.
                self.assertAlmostEqual(opt.state[parameter]["moment_bound"], expected)

    def test_invalid_settings_and_foreign_checkpoint(self):
        parameter = torch.nn.Parameter(torch.ones(32))
        for kwargs in (dict(total_bits=3), dict(total_bits=17), dict(horizon=0), dict(block_size=7),
                       dict(chunk_size=17), dict(probe_size=1), dict(v_floor=0), dict(ridge=0),
                       dict(eps=0), dict(lr=float("nan")), dict(betas=(1., 0.99))):
            with self.assertRaises(ValueError):
                HorizonAdam([parameter], **kwargs)
        adam = torch.optim.AdamW([parameter])
        parameter.grad = torch.ones_like(parameter)
        adam.step()
        with self.assertRaisesRegex(ValueError, "HorizonAdam checkpoint"):
            HorizonAdam([parameter]).load_state_dict(deepcopy(adam.state_dict()))
