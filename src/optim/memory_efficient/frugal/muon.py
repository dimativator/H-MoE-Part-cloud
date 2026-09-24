"""
FRUGAL Muon optimizers: Muon in the stateful (projected) subspace,
stateless NS-only Muon in the complement subspace.

  Stateful  subspace: accumulate EMA momentum, then apply Newton-Schulz.
  Stateless subspace: apply Newton-Schulz directly to the current gradient
                      (no momentum state stored).
"""

import torch
from typing import Callable, Dict, Iterable, Optional
import torch.nn as nn
from torch.optim import Optimizer

from .proj_optimizer_templates import GaloreOptimizer, CoordOptimizer, BlockOptimizer
from ...sota_opt.dion.newton_schulz_funcs import zeropower_via_newtonschulz5_jordan


# ─── Newton-Schulz helpers ──────────────────────────────────────────────────

def _ns(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a ≥2-D tensor via Newton-Schulz iterations."""
    return zeropower_via_newtonschulz5_jordan(G, epsilon=epsilon)


def _stateless_muon(grad: torch.Tensor, lr: float, epsilon: float, adjust_lr: bool) -> torch.Tensor:
    """Muon update with no momentum: NS(grad) scaled by lr."""
    if grad.ndim >= 2 and grad.numel() > 0:
        g_ns = _ns(grad, epsilon=epsilon).to(dtype=grad.dtype)
        if adjust_lr:
            g_ns = g_ns * (max(grad.shape[-2], grad.shape[-1]) ** 0.5)
    else:
        g_ns = grad.sign()
    return -lr * g_ns


def _coord_stateless_muon(
    grad: torch.Tensor, projector, lr: float, epsilon: float, adjust_lr: bool
) -> torch.Tensor:
    """Apply stateless Muon to the inactive coordinate complement.

    For 'columns'/'rows': extract the complement slice, run NS on it, scatter back.
    For 'randk' (element-wise): NS is ill-defined on scattered scalars, falls back
    to sign update.
    """
    update = torch.zeros_like(grad)
    coord_choice = projector.coord_choice
    active_idx = projector.indices

    if coord_choice == "columns":
        n = grad.shape[1]
        mask = torch.ones(n, dtype=torch.bool, device=grad.device)
        mask[active_idx] = False
        inactive_idx = torch.where(mask)[0]
        if inactive_idx.numel() > 0:
            update[:, inactive_idx] = _stateless_muon(grad[:, inactive_idx], lr, epsilon, adjust_lr)

    elif coord_choice == "rows":
        m = grad.shape[0]
        mask = torch.ones(m, dtype=torch.bool, device=grad.device)
        mask[active_idx] = False
        inactive_idx = torch.where(mask)[0]
        if inactive_idx.numel() > 0:
            update[inactive_idx, :] = _stateless_muon(grad[inactive_idx, :], lr, epsilon, adjust_lr)

    else:  # randk – scattered elements; fall back to sign update
        row_idx, col_idx = active_idx
        active_mask = torch.zeros_like(grad, dtype=torch.bool)
        active_mask[row_idx, col_idx] = True
        inactive_grad = grad * (~active_mask)
        update = -lr * inactive_grad.sign()

    return update


# ─── MuonBase ───────────────────────────────────────────────────────────────

class MuonBase(Optimizer):
    """Single-GPU Muon as a composable base for FRUGAL projection optimizers.

    Stateful update rule:
        m_t  = mu * m_{t-1} + (1 - mu) * g_t     (EMA momentum)
        g̃_t  = g_t + mu * m_t  if nesterov else m_t
        u_t  = NS(g̃_t)  *  (sqrt(max(m, n)) if adjust_lr)
        p_t  = p_{t-1} - lr * u_t
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr: bool = True,
        epsilon: float = 1e-7,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adjust_lr=adjust_lr,
            epsilon=epsilon, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _is_state_empty(self, state: Dict) -> bool:
        return "momentum_buffer" not in state

    @torch.no_grad()
    def _init_state(self, example: Optional[torch.Tensor] = None, state: Optional[Dict] = None) -> Dict:
        assert not (state is None and example is None)
        if state is None:
            state = {}
        if not self._is_state_empty(state):
            # Existing buffer: zero-reset, resize if shape changed
            if example is not None and state["momentum_buffer"].shape != example.shape:
                state["momentum_buffer"] = torch.zeros_like(example)
            else:
                state["momentum_buffer"].zero_()
        else:
            state["momentum_buffer"] = torch.zeros_like(example) if example is not None else None
        state["step"] = 0
        return state

    @torch.no_grad()
    def _compute_update(
        self, grad: torch.Tensor, state: Dict, lr: float, momentum: float,
        nesterov: bool, ns_steps: int, adjust_lr: bool, epsilon: float, **kwargs
    ) -> torch.Tensor:
        state["step"] = state.get("step", 0) + 1

        if state.get("momentum_buffer") is None:
            state["momentum_buffer"] = torch.zeros_like(grad)

        buf = state["momentum_buffer"]
        # EMA: buf = mu * buf + (1 - mu) * grad
        buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

        g = grad.add(buf, alpha=momentum) if nesterov else buf.clone()

        if g.ndim >= 2:
            g_ns = _ns(g, epsilon=epsilon).to(dtype=grad.dtype)
            if adjust_lr:
                g_ns = g_ns * (max(g.shape[-2], g.shape[-1]) ** 0.5)
        else:
            g_ns = g.sign()

        return -lr * g_ns

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(example=p, state=state)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                update = self._compute_update(grad, state, **group)
                p.add_(update)
        return loss


# ─── CoordMuon ──────────────────────────────────────────────────────────────

class CoordMuon(CoordOptimizer, MuonBase):
    """FRUGAL coordinate-projection optimizer with per-subspace Muon.

    Active (projected) coordinates   → Muon with EMA momentum + NS.
    Inactive (complement) coordinates → Muon stateless (NS applied to raw gradient).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,
        # projection params
        proj_params_lr_scale: float = 1.0,
        update_gap: int = 200,
        density: float = 0.25,
        reset_statistics: bool = True,
        inactive_lr_scale: float = 1.0,
        _example_state_init: bool = False,
        coord_choice: str = "columns",
        # Muon params
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr: bool = True,
        epsilon: float = 1e-7,
        weight_decay: float = 0.0,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule="sign_sgd",  # stored but overridden in _proj_params_update
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            coord_choice=coord_choice,
        )
        MuonBase.__init__(
            self, params,
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adjust_lr=adjust_lr,
            epsilon=epsilon, weight_decay=weight_decay,
        )

    @torch.no_grad()
    def _proj_params_update(self, grad: torch.Tensor, state: Dict, group: Dict) -> torch.Tensor:
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]

        # Stateful Muon on the projected subspace
        update = self._compute_update(grad_down, state, **{**group, "lr": active_lr})
        update = state["projector"].project_up(update)

        # Stateless Muon on the complement subspace
        inactive_lr = active_lr * group["inactive_lr_scale"]
        update.add_(
            _coord_stateless_muon(
                grad, state["projector"], inactive_lr,
                group.get("epsilon", 1e-7), group.get("adjust_lr", True),
            )
        )
        return update


# ─── GaloreMuon ─────────────────────────────────────────────────────────────

class GaloreMuon(GaloreOptimizer, MuonBase):
    """FRUGAL GaLore optimizer with per-subspace Muon.

    Active (SVD low-rank) subspace     → Muon with EMA momentum + NS.
    Inactive (orthogonal complement)   → Muon stateless (NS on the full residual gradient).

    Note: NS is applied to the full (m×n) residual, which is correct but expensive
    for large matrices. For compute-sensitive workloads prefer CoordMuon.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,
        # projection params
        proj_params_lr_scale: float = 1.0,
        update_gap: int = 200,
        density: float = 0.25,
        reset_statistics: bool = True,
        inactive_lr_scale: float = 1.0,
        _example_state_init: bool = False,
        proj_side: str = "std",
        proj_type: str = "svd",
        # Muon params
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr: bool = True,
        epsilon: float = 1e-7,
        weight_decay: float = 0.0,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule="sign_sgd",
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            proj_side=proj_side,
            proj_type=proj_type,
        )
        MuonBase.__init__(
            self, params,
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adjust_lr=adjust_lr,
            epsilon=epsilon, weight_decay=weight_decay,
        )

    @torch.no_grad()
    def _proj_params_update(self, grad: torch.Tensor, state: Dict, group: Dict) -> torch.Tensor:
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]

        # Stateful Muon on the SVD low-rank subspace
        update = self._compute_update(grad_down, state, **{**group, "lr": active_lr})
        update = state["projector"].project_up(update)

        if group["inactive_lr_scale"] == 0:
            return update

        # Stateless Muon on the orthogonal complement (full residual)
        inactive_grad = grad - state["projector"].project_up(grad_down)
        inactive_lr = active_lr * group["inactive_lr_scale"]
        update.add_(
            _stateless_muon(
                inactive_grad, inactive_lr,
                group.get("epsilon", 1e-7), group.get("adjust_lr", True),
            )
        )
        return update


# ─── BlockMuon ──────────────────────────────────────────────────────────────

class BlockMuon(BlockOptimizer, MuonBase):
    """FRUGAL block optimizer with per-subspace Muon.

    Active blocks   → Muon with EMA momentum + NS.
    Inactive blocks → Muon stateless (NS on the current gradient, no state).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,
        # projection params
        proj_params_lr_scale: float = 1.0,
        update_gap: int = 200,
        density: float = 0.25,
        reset_statistics: bool = True,
        inactive_lr_scale: float = 1.0,
        _example_state_init: bool = False,
        block_order: str = "random",
        # Muon params
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr: bool = True,
        epsilon: float = 1e-7,
        weight_decay: float = 0.0,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule="sign_sgd",
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            block_order=block_order,
        )
        MuonBase.__init__(
            self, params,
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adjust_lr=adjust_lr,
            epsilon=epsilon, weight_decay=weight_decay,
        )

    @torch.no_grad()
    def _proj_params_update(self, grad: torch.Tensor, state: Dict, group: Dict) -> torch.Tensor:
        if state["active"]:
            active_lr = group["lr"] * group["proj_params_lr_scale"]
            return self._compute_update(grad, state, **{**group, "lr": active_lr})
        # Inactive block: stateless Muon
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        return _stateless_muon(
            grad, inactive_lr,
            group.get("epsilon", 1e-7), group.get("adjust_lr", True),
        )
