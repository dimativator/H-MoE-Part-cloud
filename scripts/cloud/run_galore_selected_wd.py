"""Seven GaLore follow-ups after verified 500M BF16 WD selection."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = '/workspace-SR006.nfs3/dimativator/iclr500m-wd-precision-20261106/galore-selected-wd'
WDS = ('1e-1', '1e-2', '1e-3', '1e-4')
LAYOUT = [('bf16', 2)] + [(precision, budget) for precision in
          ('fp8_full', 'fp8_mixed', 'w16a16g32_fp8states') for budget in (1, 2)]


def configure() -> object:
    wd = os.environ.get('GALORE_SELECTED_WD')
    if wd not in WDS or os.environ.get('GALORE_SOURCE_FINAL_VERIFIED') != '1':
        raise RuntimeError('Verified source final and selected WD are required')
    if os.environ.get('RESULTS_DIR') != ROOT:
        raise RuntimeError('Unexpected follow-up output root')
    import run_iclr500m_wd_precision as campaign
    campaign.TRIALS = {
        f'galore_selected_wd{wd}_{precision}_{budget}xc':
            ('galore_adamw', precision, '1e-3', wd, 75457 * budget, 2 * budget)
        for precision, budget in LAYOUT
    }
    return campaign


def main() -> None:
    campaign = configure()
    campaign.main()
    if os.environ.get('CAMPAIGN_STAGE') == 'preflight':
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'}
        subprocess.run([sys.executable, str(Path(__file__).with_name('test_projected_optimizers_cpu.py'))],
                       env=env, check=True)
        campaign.emit('GALORE_FACTORY_PREFLIGHT', dict(ready=True, selected_wd=os.environ['GALORE_SELECTED_WD']))


if __name__ == '__main__':
    main()
