"""
AdEMAMix-Sign: a sign-based variant of AdEMAMix.

Like AdEMAMix, it mixes a fast and a slow EMA of the gradient, but instead of
normalizing the mixed momentum by the RMS of the gradient (exp_avg_sq), it
takes the sign of the mixed momentum, analogous to how SignSGD/Lion replace
RMS-normalization with sign().
"""

import torch

from .ademamix import linear_warmup_scheduler, linear_hl_warmup_scheduler


class AdEMAMixSign(torch.optim.Optimizer):
    r"""Implements the AdEMAMix-Sign algorithm.

    Mixes a fast EMA (beta_1) and a slow EMA (beta_3) of the gradient, then
    takes the sign of the combined momentum instead of dividing by the RMS of
    the gradient (exp_avg_sq is not tracked at all).

    update = -lr * sign(exp_avg_fast + alpha * exp_avg_slow)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            the fast and slow running averages of the gradient (beta_1, beta_3
            in AdEMAMix) (default: (0.9, 0.9999))
        alpha (float): AdEMAMix alpha coefficient mixing the slow and fast EMAs
            (default: 8.0)
        beta3_warmup (int, optional): number of warmup steps used to increase
            beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase
            alpha (default: None)
        weight_decay (float, optional): decoupled weight decay as in AdamW
            (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.9999),
        alpha=8.0,
        beta3_warmup=None,
        alpha_warmup=None,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
        )
        super(AdEMAMixSign, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMixSign, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            lmbda = group["weight_decay"]
            beta1, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMixSign does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if beta1 != 0.0:  # save memory in case beta1 is 0.0
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg_fast, exp_avg_slow = state["exp_avg_fast"], state["exp_avg_slow"]

                state["step"] += 1

                # Compute the effective alpha and beta3 in case warmup is used
                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(
                        state["step"],
                        alpha_end=alpha_final,
                        alpha_start=0,
                        warmup=alpha_warmup,
                    )
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(
                        state["step"],
                        beta_end=beta3_final,
                        beta_start=beta1,
                        warmup=beta3_warmup,
                    )
                else:
                    beta3 = beta3_final

                # Decay the fast and slow running averages of the gradient
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)

                # Take the sign of the mixed momentum instead of RMS-normalizing.
                update = exp_avg_fast.add(exp_avg_slow, alpha=alpha).sign_()

                # decoupled weight decay
                p.mul_(1 - lr * lmbda)

                p.add_(update, alpha=-lr)

        return loss
