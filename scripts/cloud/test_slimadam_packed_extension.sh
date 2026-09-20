#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -p test_fineweb_packed_third_extension.py
python -m unittest discover -s tests -p test_fineweb_replay.py
echo "SLIMADAM_PACKED_EXTENSION_TESTS_PASS"
