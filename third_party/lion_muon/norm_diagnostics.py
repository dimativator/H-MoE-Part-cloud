"""Per-iteration norm-ratio diagnostics for LionMuon / SignMuon / Muon runs.

Logs (median across 2D matrices):
  (1) ||G||_nuc / ||G||_1
  (2) ||W||_2   / ||W||_inf
  (3) ||LMO(G)||_2 / ||LMO(G)||_inf      (LMO = sign() or NS())
  (4) ||G - M||_F / ||G - M||_nuc
      ||G - M||_F / ||G - M||_1
  (5) ||grad(W_{t+1}) - grad(W_t)||_nuc / ||W_{t+1} - W_t||_2
      ||grad(W_{t+1}) - grad(W_t)||_1   / ||W_{t+1} - W_t||_inf

The ratios are recorded every `log_every_k` optimizer steps and saved as JSON
on every `flush_every_k` records (atomic write). Indexed by global step.

Usage from inside the optimizer step:

    diag = NormDiagnostics(log_path)
    ...
    # at the end of step:
    diag.record(
        step=self._step_count,
        params=[p.detach() for p in params],
        grads=[g for g in grads],
        momenta=[m for m in momenta],
        updates=[u for u in updates],
    )
    diag.maybe_flush()
"""

import json
import os
from statistics import median
from typing import List

import torch


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / b if b > eps else float('nan')


class NormDiagnostics:
    def __init__(self, log_path: str, log_every_k: int = 50, flush_every_k: int = 10):
        self.log_path = log_path
        self.log_every_k = max(1, int(log_every_k))
        self.flush_every_k = max(1, int(flush_every_k))
        self.records: list[dict] = []
        # previous-step state, keyed by position in the params list (consistent
        # across iterations so long as the optimizer iterates them in the same order).
        self._prev_grad: dict[int, torch.Tensor] = {}
        self._prev_param: dict[int, torch.Tensor] = {}
        self._records_since_flush = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    @staticmethod
    def _matrix_norms(X: torch.Tensor) -> dict:
        # X is 2D bf16/fp32; cast to fp32 for stable norms.
        Xf = X.detach().to(torch.float32)
        absXf = Xf.abs()
        # spectral via SVD top singular value
        try:
            sing = torch.linalg.svdvals(Xf)
        except Exception:
            sing = torch.tensor([0.0])
        s_top = float(sing[0]) if sing.numel() else 0.0
        s_sum = float(sing.sum())
        return {
            'inf': float(absXf.max()) if absXf.numel() else 0.0,
            '1': float(absXf.sum()),
            '2': s_top,
            'nuc': s_sum,
            'F': float(torch.linalg.norm(Xf, ord='fro')),
        }

    def record(
        self,
        step: int,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momenta: List[torch.Tensor],
        updates: List[torch.Tensor],
    ) -> None:
        if step % self.log_every_k != 0:
            return

        # Per-matrix ratios; aggregate as median.
        r_g_nuc_1: list[float] = []
        r_W_2_inf: list[float] = []
        r_U_2_inf: list[float] = []
        r_E_F_nuc: list[float] = []
        r_E_F_1: list[float] = []
        r_S_nuc_2: list[float] = []
        r_S_1_inf: list[float] = []

        for idx, (p, g, m, u) in enumerate(zip(params, grads, momenta, updates)):
            if p.ndim < 2:
                continue
            W2 = p if p.ndim == 2 else p.view(p.size(0), -1)
            G2 = g if g.ndim == 2 else g.view(g.size(0), -1)
            M2 = m if m.ndim == 2 else m.view(m.size(0), -1)
            U2 = u if u.ndim == 2 else u.view(u.size(0), -1)

            gn = self._matrix_norms(G2)
            wn = self._matrix_norms(W2)
            un = self._matrix_norms(U2)
            E = (G2 - M2)
            en = self._matrix_norms(E)

            r_g_nuc_1.append(_safe_div(gn['nuc'], gn['1']))
            r_W_2_inf.append(_safe_div(wn['2'], wn['inf']))
            r_U_2_inf.append(_safe_div(un['2'], un['inf']))
            r_E_F_nuc.append(_safe_div(en['F'], en['nuc']))
            r_E_F_1.append(_safe_div(en['F'], en['1']))

            # Smoothness proxy uses previous-step state, keyed by list position.
            if idx in self._prev_grad:
                dG = G2 - self._prev_grad[idx]
                dW = W2 - self._prev_param[idx]
                dGn = self._matrix_norms(dG)
                dWn = self._matrix_norms(dW)
                r_S_nuc_2.append(_safe_div(dGn['nuc'], dWn['2']))
                r_S_1_inf.append(_safe_div(dGn['1'], dWn['inf']))
            # Update cache for next step
            self._prev_grad[idx] = G2.detach().to(torch.float32).clone()
            self._prev_param[idx] = W2.detach().to(torch.float32).clone()

        def _med(xs):
            xs = [x for x in xs if isinstance(x, float) and x == x]  # drop nan
            return float(median(xs)) if xs else float('nan')

        rec = {
            'step': int(step),
            'grad_nuc_over_1': _med(r_g_nuc_1),
            'param_2_over_inf': _med(r_W_2_inf),
            'update_2_over_inf': _med(r_U_2_inf),
            'errmom_F_over_nuc': _med(r_E_F_nuc),
            'errmom_F_over_1': _med(r_E_F_1),
            'smooth_nuc_over_2': _med(r_S_nuc_2),
            'smooth_1_over_inf': _med(r_S_1_inf),
        }
        self.records.append(rec)
        self._records_since_flush += 1

    def maybe_flush(self) -> None:
        if self._records_since_flush < self.flush_every_k:
            return
        self.flush()

    def flush(self) -> None:
        tmp = self.log_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'records': self.records}, f)
        os.replace(tmp, self.log_path)
        self._records_since_flush = 0

