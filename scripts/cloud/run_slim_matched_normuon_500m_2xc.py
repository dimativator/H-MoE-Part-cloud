"""Fresh four-GPU SlimMatchedNorMuon 500M 2xC, without checkpoints."""

import json
import os

os.environ['MODE'] = 'matched'
os.environ['CLOUD_TRAIN_WORLD_SIZE'] = '4'
os.environ['CLOUD_DISABLE_CHECKPOINTS'] = '1'

from run_matched_normuon import campaign, preflight


def main() -> None:
    preflight()
    metadata = json.loads((campaign.DATA / 'packed_metadata.json').read_text())
    if metadata['iterations'] < 150914:
        raise RuntimeError('Packed data does not cover the 2xC budget')
    for phase in ('matched_smoke', '2xC'):
        campaign.train('slim_normuon_matched', '5e-4', 500, phase, 150914, weight_decay='1e-4')


if __name__ == '__main__':
    main()
