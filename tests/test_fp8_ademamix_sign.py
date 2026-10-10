import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PACKAGE = 'fp8_sign_test_package'
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'src/optim/sota_opt')]
sys.modules[PACKAGE] = package
FP8AdEMAMixSign = importlib.import_module(f'{PACKAGE}.fp8_ademamix_sign').FP8AdEMAMixSign
from optim.fp8_state import dequantize_fp8_state


class FP8SignTest(unittest.TestCase):
    def qargs(self):
        return SimpleNamespace(first_order_bit='E4M3', first_order_expansion='expand',
                               qgroup_size=128, expand_min=16)

    def test_update_uses_dequantized_mixed_momentum(self):
        p = torch.nn.Parameter(torch.ones(257))
        qargs = self.qargs()
        opt = FP8AdEMAMixSign([p], qargs=qargs, lr=.01, betas=(.5, .75), alpha=2, weight_decay=.2)
        for gradient in [torch.linspace(-2, 2, 257), torch.linspace(1, -3, 257)]:
            state = opt.state[p]
            fast = dequantize_fp8_state(state, 'exp_avg_fast', qargs, signed=True) if state else torch.zeros_like(p)
            slow = dequantize_fp8_state(state, 'exp_avg_slow', qargs, signed=True) if state else torch.zeros_like(p)
            fast = fast * .5 + gradient * .5
            slow = slow * .75 + gradient * .25
            expected = p.detach() * .998 - .01 * (fast + 2 * slow).sign()
            p.grad = gradient
            opt.step()
            torch.testing.assert_close(p, expected)
            self.assertEqual(opt.state[p]['exp_avg_fast'].dtype, torch.float8_e4m3fn)
            self.assertEqual(opt.state[p]['exp_avg_slow'].dtype, torch.float8_e4m3fn)
            self.assertNotIn('exp_avg_sq', opt.state[p])

    def test_checkpoint_resume_preserves_fp8_dtype_and_schedule(self):
        p = torch.nn.Parameter(torch.ones(129))
        opt = FP8AdEMAMixSign([p], qargs=self.qargs(), lr=.01, beta3_warmup=20, alpha_warmup=20)
        for step in range(3):
            p.grad = torch.linspace(-1 + step, 2, 129)
            opt.step()
        restored = torch.nn.Parameter(p.detach().clone())
        resumed = FP8AdEMAMixSign([restored], qargs=self.qargs())
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(resumed.state[restored]['exp_avg_fast'].dtype, torch.float8_e4m3fn)
        for step in range(3):
            p.grad = torch.linspace(2, -1 - step, 129)
            restored.grad = p.grad.clone()
            opt.step(); resumed.step()
            torch.testing.assert_close(restored, p, rtol=0, atol=0)

    def test_zero_beta_and_sparse_rejection(self):
        p = torch.nn.Parameter(torch.ones(5))
        opt = FP8AdEMAMixSign([p], qargs=self.qargs(), betas=(0, .9))
        opt.step()
        self.assertFalse(opt.state[p])
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertIsNone(opt.state[p]['exp_avg_fast'])
        p.grad = torch.ones_like(p).to_sparse()
        with self.assertRaises(RuntimeError): opt.step()


if __name__ == '__main__': unittest.main()
