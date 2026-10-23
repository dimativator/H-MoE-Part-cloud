"""One two-GPU LR job: separate two-step smoke, then fresh 257M 1xC."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ['MODE'] = '257m_slim_normuon'
os.environ.setdefault('CAMPAIGN_DATE', '20261023')

import run_adaptive_muon as campaign


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if revision != os.environ['CODE_REVISION']:
        raise RuntimeError('Source revision differs from the submitted revision')
    campaign.ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(campaign.ROOT).free < 10**9:
        raise RuntimeError('Output filesystem needs at least 1GB free')
    mapping = repo / 'src/optim/memory_efficient/slim_adam/layer_map_llama_paper.json'
    print('SLIM_NORMUON_CONFIG=' + json.dumps(dict(
        code_revision=revision, compression_map_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
        compression_map=json.loads(mapping.read_text()), hidden_second_moment='post_ns_update_squared',
        auxiliary_second_moment='raw_gradient_squared', auxiliary_lr='same_as_matrix',
        weight_decay=0.1, auxiliary_weight_decay=0.0, ns_steps=5,
        matrix_momentum=0.95, matrix_beta2=0.95, adamw_betas=[0.9, 0.95],
        checkpoint_saving=False, resume=False, iterations=39250, tokens=5144576000)), flush=True)
    campaign.main()


if __name__ == '__main__':
    main()
