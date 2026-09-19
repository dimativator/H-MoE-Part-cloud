"""Coordinate mlsub MPI shell ranks without requiring mpi4py in the image."""

import os
import socket
import sys
import time
from pathlib import Path


def coordination_timeout_seconds(env: dict[str, str] | None = None) -> int:
    source = os.environ if env is None else env
    timeout = int(source.get("BENCHMARK_COORDINATION_TIMEOUT_SECONDS", "300"))
    if timeout < 1:
        raise ValueError("coordination timeout must be positive")
    return timeout


def _publish(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _wait_for(path: Path, deadline: float) -> str:
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text().strip()
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def main() -> None:
    action = sys.argv[1]
    root = Path(sys.argv[2]) / "coordination"
    rank = int(os.environ["BENCHMARK_WORLD_RANK"])
    world_size = int(os.environ["BENCHMARK_WORLD_SIZE"])
    deadline = time.monotonic() + coordination_timeout_seconds()

    if action == "address":
        path = root / "master_address"
        if rank == 0:
            _publish(path, socket.gethostbyname(socket.gethostname()))
        print(_wait_for(path, deadline), flush=True)
    elif action == "sync":
        phase = sys.argv[3]
        code = int(sys.argv[4])
        directory = root / phase
        _publish(directory / f"rank{rank}", str(code))
        result = directory / "result"
        if rank == 0:
            codes = [int(_wait_for(directory / f"rank{index}", deadline)) for index in range(world_size)]
            _publish(result, str(max(codes)))
        print(_wait_for(result, deadline), flush=True)
    else:
        raise ValueError(f"unknown coordination action: {action}")


if __name__ == "__main__":
    main()
