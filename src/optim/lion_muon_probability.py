"""Probability schedules around the vendored LionMuon update (MIT).

Upstream: brain-lab-research/lion-muon, b7960bee8abfda31cb10efd3537431eeca83332f.
The update, shared momentum, scaling and AdamW backup are unchanged.
"""
import torch
import torch.distributed as dist

from third_party.lion_muon.lion_muon import LionMuon


SCHEDULES = ('constant', 'linear_up', 'linear_down', 'decay_up', 'decay_down', 'periodic')


def muon_probability(schedule: str, step: int, total_steps: int) -> float:
    """Zero-based optimizer step; all stochastic schedules average exactly 0.2."""
    if total_steps < 2 or not 0 <= step < total_steps:
        raise ValueError('Invalid probability schedule step or horizon')
    t = step / (total_steps - 1)
    if schedule in ('constant', 'periodic'):
        return 0.2
    if schedule == 'linear_up':
        return 0.05 + 0.30 * t
    if schedule == 'linear_down':
        return 0.35 - 0.30 * t
    # Match the LR scheduler: final floor(0.1*N) optimizer updates are decay.
    decay_steps = int(total_steps * 0.1)
    trunk_steps = total_steps - decay_steps
    if schedule == 'decay_up':
        return (0.2 * total_steps - 0.8 * decay_steps) / trunk_steps if step < trunk_steps else 0.8
    if schedule == 'decay_down':
        return 0.2 * total_steps / trunk_steps if step < trunk_steps else 0.0
    raise ValueError(f'Unknown LionMuon schedule: {schedule}')


class ProbabilityLionMuon(LionMuon):
    def __init__(self, muon_params, *, probability_schedule: str = 'constant',
                 total_steps: int, probability_seed: int = 20261012, **kwargs):
        if probability_schedule not in SCHEDULES or total_steps < 2:
            raise ValueError('Invalid LionMuon probability configuration')
        if kwargs.get('srank_alpha', 0) != 0:
            raise ValueError('Stable-rank trigger cannot override probability scheduling')
        super().__init__(muon_params, **kwargs)
        self.probability_schedule = probability_schedule
        self.total_steps = total_steps
        self.probability_rng = torch.Generator(device='cpu').manual_seed(probability_seed)
        self.muon_step_count = 0
        self.current_probability = 0.2
        self.last_is_muon = False

    def _is_muon_step(self) -> bool:
        self.current_probability = muon_probability(
            self.probability_schedule, self._step_count, self.total_steps)
        if self.probability_schedule == 'periodic':
            selected = super()._is_muon_step()
        else:
            # Dedicated RNG, one draw per optimizer step (never per matrix).
            selected = torch.rand((), generator=self.probability_rng).item() < self.current_probability
        if dist.is_available() and dist.is_initialized():
            device = self.param_groups[0]['params'][0].device if dist.get_backend() == 'nccl' else 'cpu'
            decision = torch.tensor(int(selected), device=device, dtype=torch.int32)
            dist.broadcast(decision, src=0)
            selected = bool(decision.item())
        self.last_is_muon = selected
        self.muon_step_count += int(selected)
        return selected

    def get_step_metrics(self) -> dict:
        return {
            'lion_muon/probability': self.current_probability,
            'lion_muon/is_muon': int(self.last_is_muon),
            'lion_muon/muon_steps': self.muon_step_count,
            'lion_muon/muon_fraction': self.muon_step_count / max(1, self._step_count),
        }

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['probability_control'] = {
            'schedule': self.probability_schedule, 'total_steps': self.total_steps,
            'step_count': self._step_count, 'muon_step_count': self.muon_step_count,
            'rng_state': self.probability_rng.get_state(),
            'current_probability': self.current_probability, 'last_is_muon': self.last_is_muon,
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        control = state_dict.pop('probability_control')
        if control['schedule'] != self.probability_schedule or control['total_steps'] != self.total_steps:
            raise ValueError('LionMuon probability configuration differs from saved state')
        super().load_state_dict(state_dict)
        self._step_count = control['step_count']
        self.muon_step_count = control['muon_step_count']
        self.probability_rng.set_state(control['rng_state'].cpu())
        self.current_probability = control['current_probability']
        self.last_is_muon = control['last_is_muon']
