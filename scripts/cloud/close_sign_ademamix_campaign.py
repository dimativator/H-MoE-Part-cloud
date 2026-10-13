#!/usr/bin/env python3
"""Close the explicitly cancelled cooldown campaign and remove only its checkpoints."""
import json
import shutil
from pathlib import Path

roots = [Path("/home/jovyan/dimativator/sign-ademamix-20261007"), Path("/home/jovyan/dimativator/sign-ademamix-precision-20261010"), Path("/home/jovyan/dimativator/sign-ademamix-clip1-20261011")]
expected = {"500m_ademamix_sign_lr1e-4_full2xc": 2.718942642211914, "500m_ademamix_sign_lr1e-4_w8a8g8_fp32_full2xc": 2.7284302711486816, "500m_ademamix_sign_lr1e-4_w8a8g8_fp8_full2xc": 7.976907730102539, "500m_ademamix_sign_lr1e-4_w16a16g32_fp8_full2xc": 3.049394130706787}
verified = {}
for root in roots:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Invalid campaign root: {root}")
    for path in root.glob("*/*/metrics.jsonl"):
        name = path.parent.name
        if name not in expected:
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        final = [r for r in records if "final-val/loss" in r][-1]
        assert final["iter"] == 150914 and final["consumed_tokens"] == 19780599808
        assert final["final-val/loss"] == expected[name]
        assert any(r.get("event") == "lm_eval" and r.get("iter") == 150914 for r in records)
        verified[name] = final
assert set(verified) == set(expected), verified.keys()
removed = []
for root in roots:
    for path in sorted(root.glob("*/*/ckpts")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"Unsafe checkpoint path: {path}")
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path)
        removed.append({"path": str(path), "bytes": size})
    remaining = [str(p) for p in root.rglob("*.pt")]
    assert not remaining, remaining
result = {"status": "closed_by_user", "cooldowns": "cancelled_by_user", "verified": verified, "removed": removed, "checkpoints_remaining": []}
for root in roots:
    (root / "campaign_closed.json").write_text(json.dumps(result, indent=2))
print("SIGN_CAMPAIGN_CLOSED " + json.dumps(result), flush=True)
