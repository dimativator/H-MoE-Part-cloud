#!/usr/bin/env bash
set -euo pipefail

readonly wandb_root=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/wandb_offline/wandb

python - "${wandb_root}" <<'PY'
import json
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


root = Path(sys.argv[1])
for run_path in sorted(root.glob("offline-run-*/run-*.wandb")):
    datastore = DataStore()
    datastore.open_for_scan(str(run_path))
    name = ""
    project = ""
    exit_code: int | None = None
    history_count = 0
    min_iteration: int | None = None
    max_iteration: int | None = None
    while True:
        data = datastore.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField("run"):
            name = record.run.display_name
            project = record.run.project
        if record.HasField("exit"):
            exit_code = record.exit.exit_code
        if not record.HasField("history"):
            continue
        history_count += 1
        values: dict[str, object] = {}
        for item in record.history.item:
            try:
                values[item.key] = json.loads(item.value_json)
            except (TypeError, ValueError):
                continue
        for key in ("iteration", "iter", "itr", "_step"):
            value = values.get(key)
            if not isinstance(value, (int, float)):
                continue
            iteration = int(value)
            min_iteration = iteration if min_iteration is None else min(min_iteration, iteration)
            max_iteration = iteration if max_iteration is None else max(max_iteration, iteration)
    print(
        f"RUN path={run_path.parent} name={name!r} project={project!r} "
        f"histories={history_count} min_iteration={min_iteration} "
        f"max_iteration={max_iteration} exit={exit_code}"
    )
PY
