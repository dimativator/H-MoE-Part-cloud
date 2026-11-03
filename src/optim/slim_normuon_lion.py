"""Periodic Lion steps with a shared momentum and compressed NorMuon branch."""
import math

import torch
from torch import Tensor

from .slim_normuon import SlimNorMuon
from third_party.normuon.kernels import zeropower_via_newtonschulz5


class SlimNorMuonLion(SlimNorMuon):
    """Use Lion's pre-update direction on both branches, not Nesterov's rule.

    Iterations 1..p-1 use sign; iteration p uses compressed post-NS variance
    and exact RMS0.2. Hidden variance advances only on the latter branch.
    Auxiliary SlimAdam runs every step. Period counters live in param_groups
    so normal optimizer state serialization preserves the global phase.
    """

    def __init__(self, params, *, period: int = 5, lion_lr_ratio: float = 0.01,
                 lion_momentum: float = 0.99, **kwargs):
        if type(period) is not int or period < 1:
            raise ValueError('period must be a positive integer')
        if not math.isfinite(lion_lr_ratio) or lion_lr_ratio <= 0:
            raise ValueError('lion_lr_ratio must be finite and positive')
        if not 0 <= lion_momentum < 1:
            raise ValueError('lion_momentum must be in [0,1)')
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group.update(slim_lion_period=period, lion_lr_ratio=lion_lr_ratio,
                         lion_momentum=lion_momentum, slim_lion_step=0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group['slim_lion_step'] += 1
        super().step()
        return loss

    @staticmethod
    def _is_slim_step(group: dict) -> bool:
        return group['slim_lion_step'] % group['slim_lion_period'] == 0

    def _step_lr(self, group: dict) -> float:
        ratio = group['lion_lr_ratio'] if group['is_proj_params'] and not self._is_slim_step(group) else 1.0
        return group['lr'] * ratio

    def _matrix_update(self, g: Tensor, state: dict, group: dict) -> Tensor:
        # Both branches consume the same OLD Lion momentum, then update it once.
        direction = g.lerp(state['momentum'], group['momentum'])
        if self._is_slim_step(group):
            update = zeropower_via_newtonschulz5(direction, steps=group['ns_steps']).float()
            state['variance'].lerp_(self._squared_statistic(update, group), 1 - group['beta2'])
            update = update / (state['variance'].sqrt() + 1e-10)
            update = update * (0.2 / update.square().mean().sqrt().clamp_min(1e-10))
        else:
            update = direction.sign()
        state['momentum'].lerp_(g, 1 - group['lion_momentum'])
        return update

    def get_step_metrics(self) -> dict:
        group = self.param_groups[0]
        step, period = group['slim_lion_step'], group['slim_lion_period']
        return {'slim_normuon_lion/period': period,
                'slim_normuon_lion/is_slim_step': int(step > 0 and step % period == 0),
                'slim_normuon_lion/slim_steps': step // period,
                'slim_normuon_lion/lion_steps': step - step // period}
