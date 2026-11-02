"""Four-GPU fresh SlimMatchedNorMuon 500M 1xC, without checkpoints."""

import os

os.environ['MODE'] = 'matched'
os.environ['CLOUD_TRAIN_WORLD_SIZE'] = '4'

from run_matched_normuon import campaign, preflight


def main() -> None:
    preflight()
    for phase in ('matched_smoke', '1xC'):
        campaign.train('slim_normuon_matched', '5e-4', 500, phase, 75457, weight_decay='1e-4')


if __name__ == '__main__':
    main()
