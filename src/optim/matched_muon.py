"""Controlled Muon/NorMuon pair with identical NS and update RMS."""

from torch import Tensor

from .adaptive_muon import _AdaptiveMuon
from third_party.normuon.kernels import zeropower_via_newtonschulz5


class MatchedMuon(_AdaptiveMuon):
    """Only row normalization differs between the two experimental variants."""

    _row_variance = True

    def __init__(self, params, row_normalization: bool = False, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group['row_normalization'] = row_normalization

    def _matrix_update(self, g: Tensor, state: dict, group: dict) -> Tensor:
        state['momentum'].lerp_(g, 1 - group['momentum'])
        direction = g.lerp(state['momentum'], group['momentum']) if group['nesterov'] else state['momentum']
        update = zeropower_via_newtonschulz5(direction, steps=group['ns_steps']).float()
        if group['row_normalization']:
            statistic = update.square().mean(dim=-1, keepdim=True)
            state['variance'].lerp_(statistic, 1 - group['beta2'])
            update = update / (state['variance'].sqrt() + 1e-10)
        # This final scaling also restores the common norm after row normalization.
        # Do not apply native aspect scaling: RMS=0.2 is independent of matrix shape.
        rms = update.square().mean().sqrt()
        return update * (0.2 / rms.clamp_min(1e-10))
