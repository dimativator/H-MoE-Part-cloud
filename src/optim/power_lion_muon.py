"""Stochastic spectral/elementwise powers of one updated EMA momentum.

This is not the original Lion two-beta update or SoftMuon's temperature rule.
Both branches use the same LR and no extra normalization, so a=1 is momentum
SGD regardless of branch. Spectral powers use a coupled Newton-Schulz/polynomial
iteration; there are no decompositions or matrix inverses in the working step.
"""
import math
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist


def power_exponent(step: int, total_steps: int, start_fraction: float = 0.9,
                   start: float = 0.0, end: float = 1.0, gamma: float = 1.0) -> float:
    """Zero-based updates: hold `start`, then ramp to `end` on the last update."""
    if total_steps < 2 or not 0 <= step < total_steps:
        raise ValueError('Invalid exponent schedule step or horizon')
    if not 0 <= start_fraction < 1 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('Require start_fraction in [0, 1) and finite gamma > 0')
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
        raise ValueError('Require finite exponents 0 <= start <= end')
    progress = max(0.0, (step / (total_steps - 1) - start_fraction) / (1 - start_fraction))
    return start + (end - start) * progress ** gamma


def spectral_power(momentum: torch.Tensor, exponent: float, steps: int = 30,
                   polynomial_terms: int = 12) -> torch.Tensor:
    """Approximate U diag(s**a) Vh without a decomposition, for a >= 0.

    Normalize X=M/||M||_F. Cubic polar NS uses S=(I-X X.T)/2 and
    X_next=(I+S) X. Alongside X, accumulate D_next=(I+S)**(1-a) D,
    starting D=X. All multipliers commute in exact arithmetic. Thus, as
    X's nonzero singular values converge to one, D's converge to s**a.
    Approximate each fractional multiplier by a binomial polynomial in S,
    whose eigenvalues initially lie in [0, 1/2]. Restore ||M||_F**a.

    `steps` controls small-mode resolution (not a uniform error guarantee).
    Tiny modes can remain under-resolved, particularly near a=0. Exact zero
    input is returned unchanged. FP32 roundoff can amplify numerical null
    modes in rank-deficient inputs; use FP64 for high-accuracy references.
    """
    if momentum.ndim != 2:
        raise ValueError('Spectral power requires a matrix')
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError('Exponent must be finite and nonnegative')
    if not isinstance(steps, int) or steps < 1 or not isinstance(polynomial_terms, int) or polynomial_terms < 1:
        raise ValueError('Require positive integer NS steps and polynomial terms')
    matrix = momentum.to(torch.float64 if momentum.dtype == torch.float64 else torch.float32)
    if exponent == 1:
        return matrix.clone()
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        # Use a max-scaled Frobenius norm to avoid overflow/underflow in M*M.
        peak = matrix.abs().amax()
        if peak == 0:
            return torch.zeros_like(matrix)
        matrix = matrix / peak
        norm = matrix.norm()
        x = matrix / norm
        transposed = x.size(0) > x.size(1)
        if transposed:
            x = x.mT
        direction = x.clone()
        identity = torch.eye(x.size(0), device=x.device, dtype=x.dtype)
        power = 1 - exponent
        coefficients = [1.0]
        for k in range(1, polynomial_terms + 1):
            coefficients.append(coefficients[-1] * (power - k + 1) / k)
        for _ in range(steps):
            residual = 0.5 * (identity - x @ x.mT)
            if exponent == 0:
                # Degree-one polynomial is exact for (I+S)**1.
                x = x + residual @ x
                direction = x
            else:
                multiplier = coefficients[-1] * identity
                for coefficient in reversed(coefficients[:-1]):
                    multiplier = multiplier @ residual + coefficient * identity
                direction = multiplier @ direction
                x = x + residual @ x
        if transposed:
            direction = direction.mT
        return direction * (peak.pow(exponent) * norm.pow(exponent))


def elementwise_power(momentum: torch.Tensor, exponent: float) -> torch.Tensor:
    """sign(M) * abs(M)**a; zero coordinates always stay zero."""
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError('Exponent must be finite and nonnegative')
    matrix = momentum.to(torch.float64 if momentum.dtype == torch.float64 else torch.float32)
    if exponent == 1:
        return matrix.clone()
    return matrix.sign() * matrix.abs().pow(exponent)


class PowerLionMuon(torch.optim.Optimizer):
    """Use `use_power=True` for matrices, False for auxiliary AdamW parameters.

    One Bernoulli draw per optimizer step, shared by all matrices and DDP ranks.
    Momentum and AdamW states are FP32 (FP64 for FP64 parameters). DDP gradients
    must already be synchronized by the training backend. No optimizer sharding.
    """

    def __init__(self, params: Iterable[dict], *, total_steps: int, lr: float = 1e-3,
                 weight_decay: float = 1e-4, momentum: float = 0.95,
                 probability: float = 0.2, probability_seed: int = 20261020,
                 start_fraction: float = 0.9, exponent_start: float = 0.0,
                 exponent_end: float = 1.0, exponent_gamma: float = 1.0,
                 ns_steps: int = 30, polynomial_terms: int = 12,
                 betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8):
        power_exponent(0, total_steps, start_fraction, exponent_start, exponent_end, exponent_gamma)
        if not 0 <= probability <= 1:
            raise ValueError('Spectral branch probability must be in [0, 1]')
        if not isinstance(ns_steps, int) or ns_steps < 1 or not isinstance(polynomial_terms, int) or polynomial_terms < 1:
            raise ValueError('Require positive integer NS steps and polynomial terms')
        defaults = {'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum,
                    'betas': betas, 'eps': eps, 'use_power': True}
        super().__init__(params, defaults)
        if not any(group['params'] for group in self.param_groups):
            raise ValueError('PowerLionMuon requires at least one parameter')
        for group in self.param_groups:
            if not math.isfinite(group['lr']) or group['lr'] < 0:
                raise ValueError('Learning rate must be finite and nonnegative')
            if not math.isfinite(group['weight_decay']) or group['weight_decay'] < 0:
                raise ValueError('Weight decay must be finite and nonnegative')
            if not 0 <= group['momentum'] < 1 or not all(0 <= b < 1 for b in group['betas']):
                raise ValueError('Momentum and betas must be in [0, 1)')
            if len(group['betas']) != 2 or not math.isfinite(group['eps']) or group['eps'] <= 0:
                raise ValueError('Require two AdamW betas and finite eps > 0')
            for param in group['params']:
                if not param.is_floating_point() or (group['use_power'] and param.ndim != 2):
                    raise ValueError('Require real floating parameters, with 2D power parameters')
        self.control = {'total_steps': total_steps, 'probability': probability,
                        'start_fraction': start_fraction, 'exponent_start': exponent_start,
                        'exponent_end': exponent_end, 'exponent_gamma': exponent_gamma,
                        'ns_steps': ns_steps, 'polynomial_terms': polynomial_terms}
        self.probability_rng = torch.Generator(device='cpu').manual_seed(probability_seed)
        self.completed_steps = 0
        self.spectral_steps = 0
        self.last_exponent = exponent_start
        self.last_is_spectral = False

    def _select_spectral(self) -> bool:
        selected = torch.rand((), generator=self.probability_rng).item() < self.control['probability']
        if dist.is_available() and dist.is_initialized():
            first_param = next(param for group in self.param_groups for param in group['params'])
            device = first_param.device if dist.get_backend() == 'nccl' else 'cpu'
            decision = torch.tensor(int(selected), device=device, dtype=torch.int32)
            dist.broadcast(decision, src=0)
            selected = bool(decision.item())
        return selected

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        exponent = power_exponent(
            self.completed_steps, self.control['total_steps'], self.control['start_fraction'],
            self.control['exponent_start'], self.control['exponent_end'], self.control['exponent_gamma'])
        # Reject sparse gradients before changing any parameter or state.
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None and param.grad.is_sparse:
                    raise RuntimeError('PowerLionMuon does not support sparse gradients')
        selected = self._select_spectral()
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                dtype = torch.float64 if param.dtype == torch.float64 else torch.float32
                grad = param.grad.to(dtype)
                state = self.state[param]
                if group['use_power']:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                    if selected:
                        update = spectral_power(buf, exponent, self.control['ns_steps'], self.control['polynomial_terms'])
                    else:
                        update = elementwise_power(buf, exponent)
                else:
                    if 'step' not in state:
                        state.update(step=0, exp_avg=torch.zeros_like(grad), exp_avg_sq=torch.zeros_like(grad))
                    state['step'] += 1
                    beta1, beta2 = group['betas']
                    state['exp_avg'].lerp_(grad, 1 - beta1)
                    state['exp_avg_sq'].lerp_(grad.square(), 1 - beta2)
                    m = state['exp_avg'] / (1 - beta1 ** state['step'])
                    v = state['exp_avg_sq'] / (1 - beta2 ** state['step'])
                    update = m / (v.sqrt() + group['eps'])
                param.mul_(1 - group['lr'] * group['weight_decay'])
                param.add_(update.to(param.dtype), alpha=-group['lr'])
        self.completed_steps += 1
        self.spectral_steps += int(selected)
        self.last_exponent = exponent
        self.last_is_spectral = selected
        return loss

    def get_step_metrics(self) -> dict:
        return {'power_lion_muon/exponent': self.last_exponent,
                'power_lion_muon/probability': self.control['probability'],
                'power_lion_muon/is_spectral': int(self.last_is_spectral),
                'power_lion_muon/spectral_fraction': self.spectral_steps / max(1, self.completed_steps)}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state['power_control'] = {'config': self.control.copy(), 'completed_steps': self.completed_steps,
                                 'spectral_steps': self.spectral_steps, 'last_exponent': self.last_exponent,
                                 'last_is_spectral': self.last_is_spectral,
                                 'rng_state': self.probability_rng.get_state()}
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        control = state_dict.pop('power_control')
        if control['config'] != self.control:
            raise ValueError('PowerLionMuon schedule/probability differs from saved state')
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict casts state to parameter dtype. Recover the
        # original FP32 states for BF16/FP16 parameters without a lossy roundtrip.
        for saved_group, group in zip(state_dict['param_groups'], self.param_groups):
            for index, param in zip(saved_group['params'], group['params']):
                dtype = torch.float64 if param.dtype == torch.float64 else torch.float32
                for key, value in state_dict['state'].get(index, {}).items():
                    if isinstance(value, torch.Tensor):
                        self.state[param][key] = value.to(device=param.device, dtype=dtype).clone()
        self.completed_steps = control['completed_steps']
        self.spectral_steps = control['spectral_steps']
        self.last_exponent = control['last_exponent']
        self.last_is_spectral = control['last_is_spectral']
        self.probability_rng.set_state(control['rng_state'].cpu())
