#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY'
import runpy

tests = runpy.run_path("tests/test_scale_optimizer.py")
for name in sorted(tests):
    if name.startswith("test_scale_"):
        tests[name]()
        print(f"PASS {name}", flush=True)
PY
echo SCALE_UNIT_TEST_COMPLETE
