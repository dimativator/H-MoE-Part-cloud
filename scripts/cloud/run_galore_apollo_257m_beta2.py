"""Four independent BF16 paper-configuration beta2 comparisons, no checkpoints."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = '/workspace-SR006.nfs3/dimativator/galore-apollo-257m-beta2-20261111'
GROUP = 'GaLore_APOLLO_257M_beta2_1xC_20261111'
TRIALS = {f'{name}_bf16_b2{beta}_1xc': (opt, 'bf16', lr, '0.1', 39250, 4)
          for name, opt, lr in (('galore', 'galore_adamw', '1e-3'),
                                ('apollo', 'apollo_adamw', '2e-3'))
          for beta in ('0.99', '0.999')}


def configure() -> object:
    if os.environ.get('RESULTS_DIR') != ROOT:
        raise RuntimeError('Unexpected beta2 comparison output root')
    import run_iclr500m_wd_precision as campaign
    campaign.TRIALS = TRIALS
    campaign.RUN_PREFIX = 'beta257m-'
    campaign.RUN_SUFFIX = '-20261111'
    original_arguments = campaign.arguments

    def arguments(trial: str, smoke: bool = False) -> list[str]:
        args = original_arguments(trial, smoke)
        values = {'--n-layer': '12', '--n-embd': '1024', '--n-head': '8',
                  '--beta2': trial.split('_b2')[1].split('_')[0]}
        if not smoke:
            values['--wandb-group'] = GROUP
            args[args.index('--wandb-tags') + 1] = '257M'
        for flag, value in values.items():
            args[args.index(flag) + 1] = value
        return args

    campaign.arguments = arguments
    return campaign


def factory_preflight(campaign: object) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from optim.optimization import get_optimizer
    for trial, (opt, _, lr, wd, _, _) in TRIALS.items():
        torch.manual_seed(0)
        parameter = torch.nn.Parameter(torch.randn(8, 8))
        beta = float(trial.split('_b2')[1].split('_')[0])
        args = SimpleNamespace(opt=opt, fp8_optim=False, non_proj_opt='adamw',
            beta1=0.9, beta2=beta, lr=float(lr), weight_decay=float(wd), eps=1e-7,
            n_embd=8, density=0.25, rank=2, update_gap=50, proj_params_lr_scale=1.0,
            reset_statistics=False, inactive_update_rule='sign_sgd', inactive_lr_scale=1.0,
            proj_side='std', proj_type='svd', apollo_proj='random',
            apollo_scale_type='channel', apollo_scale=1.0, apollo_scale_front=False)
        optimizer = get_optimizer([{'params': [parameter], 'is_proj_params': True}], args)
        for _ in range(2):
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        assert torch.isfinite(parameter).all()
        assert all(group['betas'] == (0.9, beta) for group in optimizer.param_groups)
        campaign.emit('BETA2_FACTORY_PASS', dict(trial=trial, beta2=beta, optimizer=opt))
    campaign.emit('BETA2_PREFLIGHT', dict(ready=True, trials=list(TRIALS), group=GROUP))


def main() -> None:
    campaign = configure()
    campaign.main()
    if os.environ.get('CAMPAIGN_STAGE') == 'preflight':
        factory_preflight(campaign)


if __name__ == '__main__':
    main()
