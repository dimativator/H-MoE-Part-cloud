"""Inspect follow-up storage and verify whether the complete sweep is ready."""

import json

import inspect_softmuon_aux
from softmuon_selection import select

try:
    result = select()
except (FileNotFoundError, ValueError) as exc:
    print('SOFTMUON_SWEEP_NOT_READY=' + json.dumps(dict(error=str(exc))), flush=True)
else:
    print('SOFTMUON_SWEEP_SELECTION=' + json.dumps(result), flush=True)
