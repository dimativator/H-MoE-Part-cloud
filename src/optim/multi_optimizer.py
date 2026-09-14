"""MultiOptimizer: wraps two optimizers for separate param groups (proj vs non-proj).

Used by FRUGAL optimizers when ``--non_proj_opt`` is not ``adamw``, so that
projection params (2-D weight matrices) and non-projection params (embeddings,
layer norms, lm_head) can be trained with different algorithms.

The ``param_groups`` attribute is a flat list that contains the same dict
objects as the underlying optimizers' ``param_groups`` lists.  Because Python
dicts are shared by reference, learning-rate schedulers that mutate
``group["lr"]`` will automatically update the per-group LR inside both
underlying optimizers.
"""

from __future__ import annotations


class MultiOptimizer:
    """Thin wrapper that exposes two optimizers as a single optimizer interface.

    Parameters
    ----------
    proj_opt:
        Optimizer responsible for ``is_proj_params=True`` groups (the FRUGAL
        projection optimizer).
    non_proj_opt:
        Optimizer responsible for ``is_proj_params=False`` groups (embeddings,
        norms, lm_head).
    """

    def __init__(self, proj_opt, non_proj_opt):
        self.proj_opt = proj_opt
        self.non_proj_opt = non_proj_opt
        # Build a flat list of the same dict objects used by both optimizers.
        # Schedulers that mutate group["lr"] will propagate to both sub-opts.
        self.param_groups = list(proj_opt.param_groups) + list(non_proj_opt.param_groups)

    # ------------------------------------------------------------------
    # Core optimizer interface
    # ------------------------------------------------------------------

    def step(self, closure=None):
        loss = self.proj_opt.step(closure)
        state_comm = getattr(self.proj_opt, "_state_comm", None)
        reference = next(
            (
                p
                for group in self.non_proj_opt.param_groups
                for p in group["params"]
                if p.grad is not None
            ),
            None,
        )
        if state_comm is not None and reference is not None:
            with state_comm.phase("non_proj_optimizer", reference):
                self.non_proj_opt.step()
            # The projected optimizer finishes its profile before control returns
            # here. Flush the additional non-projected phase into the same profile.
            state_comm.finish_step()
        else:
            self.non_proj_opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.proj_opt.zero_grad(set_to_none=set_to_none)
        self.non_proj_opt.zero_grad(set_to_none=set_to_none)

    def get_last_comm_profile(self):
        profile = {}
        for optimizer in (self.proj_opt, self.non_proj_opt):
            getter = getattr(optimizer, "get_last_comm_profile", None)
            if getter is None:
                continue
            for key, value in getter().items():
                profile[key] = profile.get(key, 0.0) + float(value)
        return profile

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def state_dict(self):
        return {
            "proj": self.proj_opt.state_dict(),
            "non_proj": self.non_proj_opt.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.proj_opt.load_state_dict(state["proj"])
        self.non_proj_opt.load_state_dict(state["non_proj"])
        # Re-sync param_groups so schedulers keep correct references.
        self.param_groups = (
            list(self.proj_opt.param_groups) + list(self.non_proj_opt.param_groups)
        )


class MultiScheduler:
    """Wraps two LR schedulers (one per sub-optimizer in a MultiOptimizer)."""

    def __init__(self, proj_sched, non_proj_sched):
        self.proj_sched = proj_sched
        self.non_proj_sched = non_proj_sched

    def step(self):
        self.proj_sched.step()
        self.non_proj_sched.step()

    def state_dict(self):
        return {
            "proj": self.proj_sched.state_dict(),
            "non_proj": self.non_proj_sched.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.proj_sched.load_state_dict(state["proj"])
        self.non_proj_sched.load_state_dict(state["non_proj"])

    def get_last_lr(self):
        return self.proj_sched.get_last_lr() + self.non_proj_sched.get_last_lr()
