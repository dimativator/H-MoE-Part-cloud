import unittest
from unittest.mock import patch

import torch

from optim.lion_muon_ablation import LionMuonSlimNorm, LionMuonAblation, SLIMNORM_TRIALS
from third_party.lion_muon.lion_muon import LionMuon


class SlimNormTest(unittest.TestCase):
    def make(self, variant, kind='attn.v_proj.weight', auxiliary=None):
        p = torch.nn.Parameter(torch.zeros(3, 5))
        opt = LionMuonSlimNorm([p], variant=variant, named_parameters={kind: p},
                              total_steps=101, lr=.001, lion_lr=.00001,
                              adamw_params=auxiliary)
        return opt, p

    def test_six_formulas_and_variance_axes(self):
        q = torch.arange(1., 16.).reshape(3, 5)
        for variant in SLIMNORM_TRIALS:
            for kind in ('attn.q_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight',
                         'attn.o_proj.weight', 'mlp.up_proj.weight', 'mlp.gate_proj.weight', 'mlp.down_proj.weight'):
                opt, p = self.make(variant, kind)
                axis = 0 if variant.startswith('slim') and kind not in ('attn.q_proj.weight', 'attn.k_proj.weight') else 1
                variance = torch.zeros_like(q.mean(axis, keepdim=True))
                for multiplier in (1., 2.):
                    sample = q * multiplier
                    variance = .95 * variance + .05 * sample.square().mean(axis, keepdim=True)
                    expected = sample / (variance + 1e-8).sqrt()
                    if variant in ('neuron_rms_down', 'slim_rms_down'):
                        expected *= .2 / (expected.square().mean().sqrt() + 1e-8)
                    with patch('optim.lion_muon_ablation.zeropower_via_newtonschulz5', return_value=sample):
                        actual = opt._muon_update(sample, opt.state[p], opt.param_groups[0])
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(opt.state[p]['neuron_variance'], variance)

    def test_schedule_pairing_and_endpoints(self):
        control = LionMuonAblation([torch.nn.Parameter(torch.zeros(3, 5))],
                                  variant='probability_down', total_steps=101)
        opts = [self.make(v)[0] for v in SLIMNORM_TRIALS]
        for step in range(101):
            control._step_count = step
            draw = control._is_muon_step()
            for opt in opts:
                opt._step_count = step
                self.assertEqual(opt._is_muon_step(), draw if opt.probability_down else step % 5 == 0)
                self.assertAlmostEqual(opt.last_probability, .35-.3*step/100 if opt.probability_down else .2)

    def test_momentum_lion_and_auxiliary_preserved(self):
        for variant in SLIMNORM_TRIALS:
            aux = torch.nn.Parameter(torch.zeros(5))
            base_aux = torch.nn.Parameter(torch.zeros(5))
            opt, p = self.make(variant, auxiliary=[aux])
            bp = torch.nn.Parameter(torch.zeros_like(p))
            base = LionMuon([bp], lr=.001, lion_lr=.00001, adamw_params=[base_aux])
            for step in range(10):
                p.grad = torch.randn_like(p); bp.grad = p.grad.clone()
                aux.grad = torch.randn_like(aux); base_aux.grad = aux.grad.clone()
                before = p.detach().clone(); bb = bp.detach().clone()
                with patch.object(opt, '_is_muon_step', return_value=step % 5 == 0):
                    opt.step()
                base.step()
                torch.testing.assert_close(opt.state[p]['exp_avg'], base.state[bp]['exp_avg'], rtol=0, atol=0)
                torch.testing.assert_close(aux, base_aux, rtol=0, atol=0)
                if step % 5:
                    torch.testing.assert_close(p-before, bp-bb, atol=1e-8, rtol=1e-4)
            self.assertEqual(opt.param_groups[0]['lion_lr_ratio'], .01)

    def test_launcher(self):
        from scripts.cloud.run_lionmuon_slimnorm import arguments, TRIALS
        self.assertEqual(tuple(TRIALS), SLIMNORM_TRIALS)
        for trial in TRIALS:
            cmd = arguments(trial, 'lion_muon', 75457, 7546, False)
            get = lambda x: cmd[cmd.index(x)+1]
            self.assertEqual(get('--lion-ablation'), trial)
            self.assertEqual(get('--lion-probability-seed'), '20261117')
            self.assertEqual(int(get('--batch-size'))*int(get('--acc-steps')), 128)
            self.assertEqual(get('--latest-ckpt-interval'), '0')
            self.assertEqual(get('--permanent-ckpt-interval'), '0')
            self.assertIn('--no-local-save', cmd)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
