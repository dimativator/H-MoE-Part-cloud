"""ANVIL II spectral reference, phase boundary, decay, and resume regressions."""

from argparse import ArgumentParser
from copy import deepcopy
import math
from unittest import TestCase
from unittest.mock import patch

import torch

from config.base import parse_args
from optim.anvil2 import ANVIL_MAPS, Anvil2, anvil_cascade, build_anvil2


class Anvil2Test(TestCase):
    def test_no_normalization_matches_two_momentum_spectral_reference(self):
        torch.manual_seed(23)
        for shape in ((7, 3), (3, 7), (4, 4)):
            p = torch.nn.Parameter(torch.randn(shape))
            opt = Anvil2([p], total_steps=8, normalize=False, lr=0.03,
                         weight_decay=0.1, momentum_warmup_steps=0,
                         momentum_cooldown_steps=0)
            fast, slow = torch.zeros_like(p), torch.zeros_like(p)
            for t in range(8):
                grad = torch.randn_like(p)
                p.grad = grad.clone()
                old = p.detach().clone()
                beta = 0.85 if t >= 4 else 0.93
                fast = beta * fast + (1 - beta) * grad
                slow = 0.98 * slow + 0.02 * grad
                blended = 0.4385 * fast + 0.5615 * slow if t >= 4 else fast
                update = anvil_cascade(0.07 * grad + 0.93 * blended)
                eta = 0.03 * math.sqrt(max(1, shape[0] / shape[1]))
                expected = old - eta * update - eta * 0.1 * 0.03 * old * (old * slow >= 0)
                opt.step()
                torch.testing.assert_close(p, expected)
                self.assertEqual(set(opt.state[p]), {'fast_momentum', 'slow_momentum'})
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)

    def test_no_normalization_cli_resume_and_state_mismatch(self):
        args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'anvil2',
                          '--iterations', '8', '--anvil2-no-normalization'], None)
        p = torch.nn.Parameter(torch.randn(3, 5))
        opt = build_anvil2([dict(params=[p])], args, {'projection.weight': p})
        self.assertFalse(opt.control['normalize'])
        p.grad = torch.randn_like(p)
        opt.step()
        q = torch.nn.Parameter(p.detach().clone())
        restored = build_anvil2([dict(params=[q])], args, {'projection.weight': q})
        restored.load_state_dict(deepcopy(opt.state_dict()))
        for _ in range(7):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
            opt.step()
            restored.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            Anvil2([q], total_steps=8).load_state_dict(opt.state_dict())
        old = Anvil2([q], total_steps=8).state_dict()
        del old['anvil2_control']['config']['normalize']
        Anvil2([q], total_steps=8).load_state_dict(old)

    def test_cascade_against_scalar_spectral_maps(self):
        torch.manual_seed(41)
        for shape in ((9, 4), (4, 9), (5, 5)):
            direction = torch.randn(shape).bfloat16().float()
            normalized = direction.double() / (1.05 * direction.double().norm() + 1e-6)
            left, values, right = torch.linalg.svd(normalized, full_matrices=False)
            for a, b, c in ANVIL_MAPS:
                values = a * values + b * values**3 + c * values**5
            expected = (left * values) @ right
            actual = anvil_cascade(direction)
            torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=0.03)
        torch.testing.assert_close(anvil_cascade(torch.zeros(3, 7)), torch.zeros(3, 7))

    def test_engagement_after_half_of_steps_and_slow_accumulates_from_start(self):
        for total in (8, 9):
            p = torch.nn.Parameter(torch.ones(3, 5))
            opt = Anvil2([p], total_steps=total, momentum_warmup_steps=0,
                         momentum_cooldown_steps=0)
            fast = torch.zeros_like(p)
            slow = torch.zeros_like(p)
            boundary = math.ceil(total / 2)
            for t in range(total):
                p.grad = torch.full_like(p, (-1.0)**t * (t + 1))
                engaged = t >= boundary
                fast = (0.85 if engaged else 0.93) * fast + (0.15 if engaged else 0.07) * p.grad
                slow = 0.98 * slow + 0.02 * p.grad
                blended = 0.4385 * fast + 0.5615 * slow if engaged else fast
                expected = 0.07 * p.grad + 0.93 * blended
                with patch('optim.anvil2.anvil_cascade', wraps=anvil_cascade) as cascade:
                    opt.step()
                    torch.testing.assert_close(cascade.call_args.args[0], expected)
                torch.testing.assert_close(opt.state[p]['fast_momentum'], fast)
                torch.testing.assert_close(opt.state[p]['slow_momentum'], slow)
                self.assertEqual(opt.get_step_metrics()['anvil2/engaged'], int(engaged))
            with self.assertRaises(RuntimeError):
                opt.step()

    def test_lane_shapes_norm_preservation_and_gradient_not_mutated(self):
        torch.manual_seed(11)
        for shape in ((7, 3), (3, 7), (4, 4)):
            p = torch.nn.Parameter(torch.randn(shape))
            opt = Anvil2([p], total_steps=4, weight_decay=0)
            for t in range(3):
                grad = torch.randn(shape)
                p.grad = grad.clone()
                before = p.detach().clone()
                mu = opt._momentum()
                fast = opt.state[p].get('fast_momentum', torch.zeros_like(p))
                slow = opt.state[p].get('slow_momentum', torch.zeros_like(p))
                engaged = t >= 2
                fast = (0.85 if engaged else mu) * fast + (0.15 if engaged else 1 - mu) * grad
                slow = 0.98 * slow + 0.02 * grad
                momentum = 0.4385 * fast + 0.5615 * slow if engaged else fast
                spectral = anvil_cascade((1 - mu) * grad + mu * momentum)
                old_energy = opt.state[p].get('lane_energy', 0)
                axis = 1 if shape[0] >= shape[1] else 0
                expected_energy = 0.9 * old_energy + 0.1 * spectral.square().mean(axis, keepdim=True)
                opt.step()
                eta = 0.02 * math.sqrt(max(1, shape[0] / shape[1]))
                torch.testing.assert_close(((before - p) / eta).norm(), spectral.norm())
                torch.testing.assert_close(opt.state[p]['lane_energy'], expected_energy)
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
            self.assertEqual(opt.state[p]['lane_energy'].shape,
                             (shape[0], 1) if shape[0] >= shape[1] else (1, shape[1]))

    def test_matrix_decay_uses_slow_stream_old_weights_and_extra_lr(self):
        for shape in ((4, 2), (2, 4), (2, 2)):
            p = torch.nn.Parameter(torch.ones(shape))
            opt = Anvil2([p], total_steps=4, lr=0.2, weight_decay=0.5)
            slow = torch.tensor([1., -1.]).repeat(p.numel() // 2).reshape(shape)
            opt.state[p].update(fast_momentum=-slow.clone(), slow_momentum=slow.clone(),
                                lane_energy=torch.zeros(shape[0], 1) if shape[0] >= shape[1]
                                else torch.zeros(1, shape[1]))
            p.grad = torch.zeros_like(p)
            with patch('optim.anvil2.anvil_cascade', lambda x: torch.zeros_like(x)):
                opt.step()
            eta = 0.2 * math.sqrt(max(1, shape[0] / shape[1]))
            expected = 1 - eta * 0.5 * 0.2 * (slow >= 0)
            torch.testing.assert_close(p, expected)

    def test_auxiliary_matches_active_muon_adam_except_cautious_decay(self):
        from third_party.lite.muonlite import MuonLite

        p = torch.nn.Parameter(torch.tensor([1., -1., 1., -1., 1.]))
        q = torch.nn.Parameter(p.detach().clone())
        opt = Anvil2([dict(params=[p], is_proj_params=False)], total_steps=5,
                     lr=0.03, weight_decay=0.2, eps=1e-3)
        reference = MuonLite(muon_params=[], adamw_params=[('lm_head.weight', q)],
                             lr=0.03, weight_decay=0, adamw_betas=(0.9, 0.95), adamw_eps=1e-3,
                             beta1=0, beta2=0, chi=1, chi_adamw=1, subspace_ratio=0,
                             total_steps=5, warmup_steps=0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)
        ref_scheduler = torch.optim.lr_scheduler.StepLR(reference, 1, gamma=0.5)
        for _ in range(4):
            # Small gradients expose differences in epsilon placement.
            grad = torch.tensor([2e-4, 2e-4, -2e-4, -2e-4, 0.])
            old = p.detach().clone()
            lr = opt.param_groups[0]['lr']
            p.grad, q.grad = grad.clone(), grad.clone()
            reference.step()
            update = (old - q.detach()) / lr
            with torch.no_grad():
                q.sub_(lr * 0.2 * old * (old * update > 0))
            opt.step()
            torch.testing.assert_close(p, q)
            torch.testing.assert_close(opt.state[p]['exp_avg'], reference.state[q]['moment1'])
            torch.testing.assert_close(opt.state[p]['exp_avg_sq'], reference.state[q]['moment2'])
            self.assertEqual(opt.state[p]['step'], reference.state[q]['step'])
            scheduler.step()
            ref_scheduler.step()

    def test_builder_matches_muon_grouping_shared_lr_decay_and_epsilon(self):
        args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'anvil2',
                          '--iterations', '9', '--lr', '0.02', '--weight-decay', '0.1',
                          '--eps', '0.001', '--muon_adamw_lr_scale', '0.1'], None)
        matrix = torch.nn.Parameter(torch.randn(4, 7))
        head = torch.nn.Parameter(torch.randn(11, 7))
        embedding = torch.nn.Parameter(torch.randn(11, 7))
        norm = torch.nn.Parameter(torch.ones(7))
        groups = [dict(params=[matrix, head], is_proj_params=False, weight_decay=0),
                  dict(params=[embedding, norm], is_proj_params=False, weight_decay=0)]
        named = {'module.proj.weight': matrix, 'module.lm_head.weight': head,
                 'module.transformer.wte.weight': embedding, 'module.norm.weight': norm}
        opt = build_anvil2(groups, args, named)
        self.assertEqual(opt.control['engage_step'], 5)
        mapping = {id(p): g for g in opt.param_groups for p in g['params']}
        self.assertTrue(mapping[id(matrix)]['is_proj_params'])
        for p in (head, embedding, norm):
            self.assertFalse(mapping[id(p)]['is_proj_params'])
            self.assertEqual(mapping[id(p)]['weight_decay'], 0.1)
            self.assertEqual(mapping[id(p)]['betas'], (0.9, 0.95))
            self.assertEqual(mapping[id(p)]['lr'], args.lr)
            self.assertEqual(mapping[id(p)]['eps'], 1e-8)
        self.assertEqual(groups[1]['weight_decay'], 0)
        args.fp8_optim = True
        with self.assertRaises(ValueError):
            build_anvil2(groups, args, named)

    def test_resume_across_switch_with_scheduler_and_fp32_states(self):
        torch.manual_seed(9)
        for dtype in (torch.float32, torch.bfloat16):
            for saved_after in (3, 4, 5):
                params = [torch.nn.Parameter(torch.randn(4, 7, dtype=dtype)),
                          torch.nn.Parameter(torch.randn(7, dtype=dtype))]
                groups = [dict(params=[params[0]]), dict(params=[params[1]], is_proj_params=False)]
                opt = Anvil2(groups, total_steps=8)
                scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
                for _ in range(saved_after):
                    for p in params:
                        p.grad = torch.randn_like(p)
                    opt.step()
                    scheduler.step()
                restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
                restored = Anvil2([dict(params=[restored_params[0]]),
                                  dict(params=[restored_params[1]], is_proj_params=False)], total_steps=8)
                restored_scheduler = torch.optim.lr_scheduler.StepLR(restored, 1, gamma=0.9)
                restored.load_state_dict(deepcopy(opt.state_dict()))
                restored_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
                for _ in range(saved_after, 8):
                    for p, q in zip(params, restored_params):
                        p.grad = torch.randn_like(p)
                        q.grad = p.grad.clone()
                    opt.step()
                    restored.step()
                    scheduler.step()
                    restored_scheduler.step()
                    self.assertEqual(opt.completed_steps, restored.completed_steps)
                    for p, q in zip(params, restored_params):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                        for key, value in opt.state[p].items():
                            if isinstance(value, torch.Tensor):
                                self.assertEqual(restored.state[q][key].dtype, torch.float32)
                                torch.testing.assert_close(value, restored.state[q][key], rtol=0, atol=0)
                mismatch = Anvil2([torch.nn.Parameter(torch.ones(4, 7))], total_steps=9)
                with self.assertRaises(ValueError):
                    mismatch.load_state_dict(opt.state_dict())

    def test_missing_zero_sparse_gradients_and_invalid_configuration(self):
        p = torch.nn.Parameter(torch.ones(3, 5))
        opt = Anvil2([p], total_steps=4)
        opt.step()
        self.assertFalse(opt.state)
        p.grad = torch.zeros_like(p)
        self.assertEqual(opt.step(lambda: torch.tensor(2.)).item(), 2)
        self.assertTrue(torch.isfinite(p).all())
        before = deepcopy(opt.state_dict())
        p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.ones(1), p.shape)
        with self.assertRaises(RuntimeError):
            opt.step()
        self.assertEqual(opt.completed_steps, before['anvil2_control']['completed_steps'])
        for kwargs in (dict(total_steps=0), dict(engage_fraction=1.1), dict(lr=-1),
                       dict(eps=0), dict(betas=(1, 0.95)), dict(momentum_warmup_steps=-1)):
            with self.assertRaises(ValueError):
                Anvil2([p], **{'total_steps': 4, **kwargs})

    def test_momentum_schedule_reference_boundaries(self):
        opt = Anvil2([torch.nn.Parameter(torch.ones(2, 3))], total_steps=1000)
        for t, expected in ((0, 0.85), (120, 0.89), (240, 0.93),
                            (950, 0.93), (975, 0.89), (999, 0.8516)):
            opt.completed_steps = t
            self.assertAlmostEqual(opt._momentum(), expected)

    def test_llama_training_with_gradient_accumulation(self):
        from models.llama import Llama

        torch.manual_seed(3)
        args = parse_args(ArgumentParser(), ['--experiment-name', 'anvil2-smoke', '--opt', 'anvil2',
                          '--iterations', '8', '--model', 'llama', '--device', 'cpu',
                          '--vocab-size', '32', '--n-embd', '16', '--n-head', '2', '--n-layer', '1',
                          '--sequence-length', '8', '--lr', '0.003'], None)
        model = Llama(args)
        named = dict(model.named_parameters())
        groups = [{**g, 'params': [named[name] for name in g['params']]}
                  for g in model.get_parameter_group_specs()]
        opt = build_anvil2(groups, args, named)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 4, gamma=0.5)
        tokens = torch.randint(0, 32, (2, 8))
        losses = []
        for step in range(args.iterations):
            opt.zero_grad()
            for _ in range(2):
                with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                    loss = model(tokens, targets=tokens)['loss']
                (loss / 2).backward()
                self.assertEqual(opt.completed_steps, step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        self.assertEqual(opt.control['engage_step'], 4)
        for name in ('lm_head.weight', 'transformer.wte.weight'):
            self.assertEqual(set(opt.state[named[name]]), {'step', 'exp_avg', 'exp_avg_sq'})
            self.assertEqual(opt.state[named[name]]['step'], args.iterations)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
