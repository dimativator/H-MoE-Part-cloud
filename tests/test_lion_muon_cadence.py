import random
import unittest

import torch
from optim.lion_muon_cadence import LionMuonCadence, TRIALS
from scripts.cloud.run_lionmuon_cadence import arguments


class CadenceTest(unittest.TestCase):
    def test_full_horizon_and_launch_config(self):
        names = {f"transformer.h.{i}.weight": torch.nn.Parameter(torch.zeros(3, 4)) for i in range(18)}
        for trial in TRIALS:
            opt = LionMuonCadence(list(names.values()), named_parameters=names,
                                  variant=trial, total_steps=75457, lr=.001, lion_lr=.00001)
            rng = random.Random(20261117)
            expected = 0.
            for step in range(75457):
                opt._step_count = step
                decision = opt._is_muon_step()
                expected += opt.last_probability
                if trial == "muon_ema":
                    self.assertEqual(decision, rng.random() < .35 - .30 * (step / 75456))
            self.assertAlmostEqual(expected, .2 * 75457, places=6)
            if opt.stagger or trial == "regular":
                for count, weight in zip(opt.block_counts, opt.weights):
                    self.assertLess(abs(count - weight * expected), 1)
            cmd = arguments(trial, 'lion_muon', 75457, 7546, False)
            self.assertEqual(cmd[cmd.index('--lion-ablation')+1], trial)
            self.assertIn('--no-local-save', cmd)
            self.assertIn('--wandb', cmd)
            for flag in ('--latest-ckpt-interval', '--permanent-ckpt-interval'):
                self.assertEqual(cmd[cmd.index(flag)+1], '0')

    def test_mixed_branch_updates_and_ema(self):
        for trial in ("stagger_uniform", "muon_ema"):
            names = {f"h.{i}.weight": torch.nn.Parameter(torch.ones(3, 4)) for i in range(3)}
            opt = LionMuonCadence(list(names.values()), named_parameters=names, variant=trial,
                                  total_steps=30, lr=.001, lion_lr=.00001, weight_decay=.1)
            for _ in range(12):
                expected = []
                for p in names.values():
                    p.grad = torch.randn_like(p)
                    expected.append((p.detach().clone(), opt.state[p].get("exp_avg", torch.zeros_like(p)).clone()))
                opt.step()
                for i, (p, (old, m)) in enumerate(zip(names.values(), expected)):
                    muon = opt.fires[i if opt.stagger else 0]
                    beta = .99 if muon and trial == "muon_ema" else .95
                    d = beta*m + (1-beta)*p.grad
                    u = opt._muon_update(d, opt.state[p], opt.param_groups[0]) if muon else d.sign()
                    lr = .001 if muon else .00001
                    torch.testing.assert_close(p, old*(1-.1*lr)-lr*u.bfloat16().float())
                    torch.testing.assert_close(opt.state[p]["exp_avg"], .99*m+.01*p.grad)


if __name__ == "__main__":
    unittest.main()
