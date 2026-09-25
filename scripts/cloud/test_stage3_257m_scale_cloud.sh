#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q tests/test_scale_optimizer.py
echo SCALE_UNIT_TEST_COMPLETE
