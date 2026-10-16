from unittest import TestCase
from unittest.mock import patch

import torch

from optim.horizon_adam import HorizonAdam


class HorizonIdentityTest(TestCase):
    def test_identity_matches_explicit_fixed_codec_ablation(self):
        torch.manual_seed(7)
        first = torch.nn.Parameter(torch.randn(513))
        second = torch.nn.Parameter(first.detach().clone())
        identity = HorizonAdam([first], min_size=0, horizon=2, codec_mode='identity')
        reference = HorizonAdam([second], min_size=0, horizon=2)
        for _ in range(20):
            gradient = torch.randn_like(first)
            first.grad = gradient.clone()
            second.grad = gradient.clone()
            identity.step()
            with patch('optim.horizon_adam.update_probes', return_value=None):
                reference.step()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        state = identity.state[first]
        self.assertEqual(state['bits'], (4, 4))
        self.assertEqual(state['codec_updates'], 0)
        torch.testing.assert_close(state['encoder'], torch.eye(2), rtol=0, atol=0)

    def test_identity_resume_preserves_mode_and_updates(self):
        first = torch.nn.Parameter(torch.randn(257))
        optimizer = HorizonAdam([first], min_size=0, codec_mode='identity')
        first.grad = torch.randn_like(first)
        optimizer.step()
        second = torch.nn.Parameter(first.detach().clone())
        resumed = HorizonAdam([second])
        resumed.load_state_dict(optimizer.state_dict())
        self.assertEqual(resumed.param_groups[0]['codec_mode'], 'identity')
        for _ in range(5):
            gradient = torch.randn_like(first)
            for parameter, opt in ((first, optimizer), (second, resumed)):
                parameter.grad = gradient.clone()
                opt.step()
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_legacy_checkpoint_defaults_to_learned_codec(self):
        first = torch.nn.Parameter(torch.randn(257))
        optimizer = HorizonAdam([first], min_size=0, horizon=2)
        first.grad = torch.randn_like(first)
        optimizer.step()
        checkpoint = optimizer.state_dict()
        checkpoint['param_groups'][0].pop('codec_mode')
        resumed = HorizonAdam([first])
        resumed.load_state_dict(checkpoint)
        for _ in range(3):
            first.grad = torch.randn_like(first)
            resumed.step()
        self.assertGreater(resumed.state[first]['codec_updates'], 0)
