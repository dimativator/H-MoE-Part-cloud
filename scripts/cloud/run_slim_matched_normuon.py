"""Fresh SlimAdam-compressed matched NorMuon 257M, without checkpoints."""

from run_matched_normuon import campaign, preflight


def main() -> None:
    preflight()
    for phase in ('matched_smoke', '1xC'):
        campaign.train('slim_normuon_matched', '5e-4', 257, phase, 39250, weight_decay='1e-1')


if __name__ == '__main__':
    main()
