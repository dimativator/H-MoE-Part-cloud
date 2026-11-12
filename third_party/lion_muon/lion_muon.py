"""
LionMuon: Alternating Lion and Muon updates.

Lion update rule:
    update = sign(beta1 * m + (1 - beta1) * g)
    m = beta2 * m + (1 - beta2) * g

LionMuon replaces the sign() on Muon steps with Newton-Schulz orthogonalization:
    - Every K steps: p -= lr  * 0.2*sqrt(max(m,n)) * NS(beta1*m + (1-beta1)*g)
    - Otherwise:     p -= slr * sign(beta1*m + (1-beta1)*g)
    - Always:        m = beta2*m + (1-beta2)*g

Both steps share the same momentum buffer m (updated with beta2).
"""

import os

import torch
import torch.distributed as dist

from .muon import zeropower_via_newtonschulz5
from .norm_diagnostics import NormDiagnostics
from .schedule import cos_inf_schedule, cosine_wsd_decay_schedule, wsd_schedule
from .utils import estimate_stable_rank


class LionMuon(torch.optim.Optimizer):
    """
    LionMuon alternates between full Muon (NS orthogonalization) and Lion
    (sign) updates for 2D parameters. Non-2D parameters always use AdamW.

    Arguments:
        muon_params: Parameters to optimize with LionMuon.
        lr: Learning rate for Muon steps.
        lion_lr: Learning rate for Lion steps. If None, defaults to lr.
        beta1: Interpolation coefficient for update direction (default 0.95).
        beta2: Momentum EMA decay (default 0.99).
        ns_steps: Newton-Schulz steps for Muon (default 6).
        muon_every_k: Muon every K steps, Lion for the rest (default 5).
        adamw_params: Parameters to always optimize with AdamW.
        adamw_lr, adamw_betas, adamw_eps, adamw_wd: AdamW hyperparameters.
    """

    def __init__(
        self,
        muon_params,
        lr=2e-3,
        lion_lr=None,
        beta1=0.95,
        beta2=0.99,
        ns_steps=6,
        muon_every_k=5,
        weight_decay=0.0,
        srank_alpha=0.0,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.8, 0.999),
        adamw_eps=1e-8,
        adamw_wd=0.1,
        norm_diag_path=None,
        norm_diag_every_k=50,
    ):
        if lion_lr is None:
            lion_lr = lr * 0.01  # default: 1% of muon lr

        defaults = dict(
            lr=lr,
            lion_lr_ratio=lion_lr / lr if lr > 0 else 0.01,
            beta1=beta1,
            beta2=beta2,
            ns_steps=ns_steps,
            muon_every_k=muon_every_k,
            weight_decay=weight_decay,
            srank_alpha=srank_alpha,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr if lr > 0 else 1.0,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        self._step_count = 0
        self._diag = {
            "srank_ratios": [],
            "update_types": [],
        }
        self._norm_diag = (
            NormDiagnostics(norm_diag_path, log_every_k=norm_diag_every_k)
            if norm_diag_path else None
        )

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def _is_muon_step(self):
        k = self.param_groups[0]["muon_every_k"]
        return k <= 1 or (self._step_count % k) == 0

    def get_cumulative_diagnostics(self):
        ratios = self._diag.get("srank_ratios", [])
        types = self._diag.get("update_types", [])
        if not ratios:
            return {}
        
        n = len(ratios)
        mean_srank = sum(ratios) / n if n > 0 else 0.0
        var_srank = sum((r - mean_srank) ** 2 for r in ratios) / (n - 1) if n > 1 else 0.0
        ns_count = sum(1 for t in types if t == "NS")

        return {
            "srank_mean": mean_srank,
            "srank_var": var_srank,
            "srank_count": n,
            "ns_count": ns_count,
            "srank_ratios": ratios,
        }

    def step(self):
        is_muon_global = self._is_muon_step()
        self._step_count += 1
        srank_alpha = self.param_groups[0].get("srank_alpha", 0.0)
        adaptive = srank_alpha > 0
        if adaptive:
            self._diag["srank_ratios"] = []
            self._diag["update_types"] = []

        # Collect diagnostics tensors only on log steps to avoid the cost of cloning every iteration.
        norm_diag_active = (
            self._norm_diag is not None
            and self.world_size == 1
            and (self._step_count % self._norm_diag.log_every_k == 0)
        )
        diag_payload = [] if norm_diag_active else None

        for group in self.param_groups:

            ############################
            #   LionMuon (2D params)   #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            muon_lr = group["lr"]
            lion_lr = group["lion_lr_ratio"] * group["lr"]  # scales with scheduler
            if not adaptive:
                lr = muon_lr if is_muon_global else lion_lr

            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device="cuda", dtype=torch.bfloat16)
            if adaptive:
                param_lrs = []
            curr_idx = 0

            for i, p in enumerate(params):
                did_muon = False
                if i % self.world_size == self.rank:
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)

                    m = state["exp_avg"]

                    # Lion interpolation: direction to update
                    update = m * beta1 + g * (1 - beta1)

                    # Decide muon vs lion for this param
                    m_rows, n_cols = int(update.size(0)), int(update.size(1))
                    dmin, dmax = min(m_rows, n_cols), max(m_rows, n_cols)
                    if adaptive:
                        srank_ratio, srank_sq = estimate_stable_rank(update, power_iters=3)
                        did_muon = srank_sq <= (srank_alpha * dmin)
                        self._diag["srank_ratios"].append(float(srank_ratio))
                        self._diag["update_types"].append("NS" if did_muon else "sign")
                    else:
                        did_muon = is_muon_global

                    if did_muon:
                        # NS orthogonalization + Moonshot scaling
                        update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                        update *= 0.2 * max(update.size(0), update.size(1)) ** 0.5
                    else:
                        # Lion: pure sign
                        update = update.sign()

                    updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()

                    # Capture diagnostics BEFORE the in-place momentum update so
                    # that m corresponds to M_{t-1} (pre-update buffer) when
                    # computing the (G - M) residual. Clone p so the W_t
                    # snapshot survives the parameter update later in this step.
                    if norm_diag_active:
                        diag_payload.append((
                            p.detach().clone(),
                            g.detach().clone(),
                            m.detach().clone(),
                            update.detach().clone(),
                        ))

                    # Update momentum with beta2 (Lion rule, after computing direction)
                    m.mul_(beta2).add_(g, alpha=1 - beta2)

                if adaptive:
                    param_lrs.append(muon_lr if did_muon else lion_lr)
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            wd = group["weight_decay"]
            for j, p in enumerate(params):
                u = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                plr = param_lrs[j] if adaptive else lr
                if wd > 0:
                    p.data.mul_(1 - plr * wd)
                p.data.add_(u, alpha=-plr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            adamw_lr = group["adamw_lr_ratio"] * group["lr"]
            beta1_aw, beta2_aw = group["adamw_betas"]
            eps = group["adamw_eps"]
            wd = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1_aw)
                buf2.lerp_(g.square(), 1 - beta2_aw)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1_aw ** step
                bias_correction2 = 1 - beta2_aw ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - adamw_lr * wd)
                p.data.add_(g, alpha=-adamw_lr / scale)

        # Norm-ratio diagnostics: record once per log-step using the W_t snapshot
        # captured during the LionMuon loop above (before this method's loops
        # applied the update to p.data, so the stored params are still W_t).
        if norm_diag_active and diag_payload:
            ps, gs, ms, us = zip(*diag_payload)
            self._norm_diag.record(
                step=self._step_count,
                params=list(ps),
                grads=list(gs),
                momenta=list(ms),
                updates=list(us),
            )
            self._norm_diag.maybe_flush()




class LionMuonScheduler:
    """Scheduler for LionMuon, mirrors SignMuonScheduler."""

    def __init__(self, optimizer, cfg):
        self.schedulers = []
        def _make_cos_scheduler(opt, lr):
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-2, total_iters=cfg.warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.iterations - cfg.warmup_steps, eta_min=0
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[cfg.warmup_steps]
            )

        scheduler_map = {
            "cos": _make_cos_scheduler,
            "wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                wsd_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=cfg.wsd_final_lr_scale,
                    decay_type=cfg.decay_type,
                ),
            ),
            "cos_inf": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cos_inf_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    n_inf=cfg.cos_inf_steps,
                    div_factor=1e2,
                    final_div_factor=0.1,
                ),
            ),
            "cos_wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cosine_wsd_decay_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    anneal_end_factor=0.15,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=0.1,
                    decay_type=cfg.decay_type,
                ),
            ),
        }
        for group in optimizer.param_groups:
            scheduler_cls = scheduler_map.get(cfg.scheduler, None)
            if scheduler_cls:
                self.schedulers.append(scheduler_cls(optimizer, group["lr"]))

    def step(self):
        for s in self.schedulers:
            s.step()

    def state_dict(self):
        return {f"scheduler_{i}": s.state_dict() for i, s in enumerate(self.schedulers)}

    def load_state_dict(self, state_dict):
        for i, s in enumerate(self.schedulers):
            s.load_state_dict(state_dict[f"scheduler_{i}"])

