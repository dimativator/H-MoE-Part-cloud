"""Fresh SlimAdam-compressed matched NorMuon 257M, without checkpoints."""

import os

from run_matched_normuon import campaign, preflight


def main() -> None:
    preflight()
    lr = os.environ.get('LR', '5e-4')
    if lr not in campaign.LRS:
        raise ValueError('LR outside prepared grid')
    for phase in ('matched_smoke', '1xC'):
        campaign.train('slim_normuon_matched', lr, 257, phase, 39250, weight_decay='1e-1')


if __name__ == '__main__':
    main()
