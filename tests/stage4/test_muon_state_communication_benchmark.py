import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import benchmark_muon_state_communication as bench
from scripts.benchmark_muon_state_communication import (
    _has_complete_samples,
    aggregate_repeats,
    aggregate_external,
    build_command,
    parse_output,
)


def test_pp2_dp2_command_uses_four_processes_without_megatron_distopt():
    command = build_command(
        Path("."),
        model_name="5.0b-wide",
        method="frugal_muon_muon_fp8_states",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        sequence_length=1024,
        micro_batch_size=1,
        global_batch_size=4,
        warmup_steps=2,
        measure_steps=3,
        density=0.25,
        update_gap=50,
    )

    assert command[command.index("--nproc-per-node") + 1] == "4"
    assert command[command.index("--pipeline-model-parallel-size") + 1] == "2"
    assert command[command.index("--optimizer") + 1] == "frugal_muon_muon"
    assert command[command.index("--optimizer-state-precision") + 1] == "fp8"
    assert "--use-distributed-optimizer" not in command
    assert command[command.index("--muon-fp8-bucket-bytes") + 1] == str(64 * 2**20)
    assert "--muon-fused-fp8-ns-input" in command


def test_external_mpi_command_does_not_start_nested_torchrun():
    command = build_command(
        Path("."),
        model_name="5.0b-wide",
        method="muon_bf16_states",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        sequence_length=1024,
        micro_batch_size=1,
        global_batch_size=4,
        warmup_steps=2,
        measure_steps=3,
        density=0.25,
        update_gap=50,
        external_distributed=True,
    )

    assert command[1] == "stage4/pretrain_gpt.py"
    assert "torch.distributed.run" not in command


def test_tp_pp_dp_mapping_is_forwarded_to_megatron():
    command = build_command(
        Path("."),
        model_name="9.9b-pp2",
        method="muon_fp8_states",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        data_parallel_size=8,
        sequence_length=1024,
        micro_batch_size=1,
        global_batch_size=8,
        warmup_steps=2,
        measure_steps=3,
        density=0.25,
        update_gap=50,
        use_tp_pp_dp_mapping=True,
    )

    assert "--use-tp-pp-dp-mapping" in command


def test_pp2_model_keeps_32_layers_per_pipeline_stage():
    model = bench.MODELS["9.9b-pp2"]

    assert model["layers"] == 64
    assert model["layers"] // 2 == bench.MODELS["4.9b"]["layers"]


def test_dp8_scaling_models_keep_4_9b_layer_shape():
    names = ("1b", "2b", "3b", "3.5b", "4b", "4.5b", "4.9b")
    assert [bench.MODELS[name]["layers"] for name in names] == [5, 12, 19, 22, 26, 29, 32]
    for name in names:
        assert {key: bench.MODELS[name][key] for key in ("hidden", "ffn", "heads")} == {
            "hidden": 3456,
            "ffn": 9216,
            "heads": 27,
        }


def test_dp8_scaling_launcher_preserves_measurement_settings():
    launcher = (Path(__file__).resolve().parents[2] / "cloud_benchmark_muon_state_communication_dp8_scaling.sh").read_text()
    assert "for model in 1b 2b 3b 3.5b 4b 4.5b 4.9b" in launcher
    assert "TP=1 PP=1 DP=8" in launcher
    assert "--global-batch-size 8" in launcher
    assert "--warmup-steps 10" in launcher
    assert "--measure-steps 50" in launcher
    assert "--repeats 3" in launcher
    assert "--fp8-bucket-bytes 67108864" in launcher
    assert "--fused-fp8-ns-input" in launcher
    assert "frugal_muon_muon_fp8_states" in launcher


def test_dp4_scaling_launcher_preserves_measurement_settings():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "cloud_benchmark_muon_state_communication_dp4_scaling.sh"
    ).read_text()
    assert "for model in 1b 2b 3b 3.5b 4b 4.5b 4.9b" in launcher
    assert "TP=1 PP=1 DP=4" in launcher
    assert "--global-batch-size 4" in launcher
    assert "--warmup-steps 10" in launcher
    assert "--measure-steps 50" in launcher
    assert "--repeats 3" in launcher
    assert "--fp8-bucket-bytes 67108864" in launcher
    assert "--fused-fp8-ns-input" in launcher
    assert "frugal_muon_muon_fp8_states" in launcher


def test_profiles_are_reduced_to_the_slowest_rank_per_step():
    output = "\n".join(
        (
            "iteration 3/ 5 | elapsed time per iteration (ms): 100.0",
            '[OPTIMIZER STATE COMM PROFILE] {"step": 2, "newton_schulz_ms": 10, "wire_encode_ms": 2, "state_all_gather_ms": 3, "wire_decode_ms": 4, "other_ms": 5, "optimizer_ms": 24, "stateful_wire_bytes": 100, "stateless_wire_bytes": 200, "received_wire_bytes": 75}',
            '[OPTIMIZER STATE COMM PROFILE] {"step": 2, "newton_schulz_ms": 12, "wire_encode_ms": 1, "state_all_gather_ms": 5, "wire_decode_ms": 3, "other_ms": 4, "optimizer_ms": 25, "stateful_wire_bytes": 100, "stateless_wire_bytes": 200, "received_wire_bytes": 75}',
        )
    )

    result = parse_output(output, warmup_steps=2, measure_steps=1)

    assert result["mean_step_ms"] == 100.0
    assert result["newton_schulz_ms"] == 12.0
    assert result["state_all_gather_ms"] == 5.0
    assert result["optimizer_ms"] == 25.0


def test_external_rank_results_use_critical_rank_per_step():
    rank0 = {
        "model": "5.0b-wide",
        "method": "muon_fp8_states",
        "rank": 0,
        "status": "ok",
        "peak_allocated_bytes": 10,
        "_step_times": [100.0],
        "_profiles": [
            {
                "step": 2,
                "newton_schulz_ms": 10,
                "wire_encode_ms": 2,
                "state_all_gather_ms": 3,
                "wire_decode_ms": 4,
                "other_ms": 5,
                "optimizer_ms": 24,
                "stateful_wire_bytes": 100,
                "stateless_wire_bytes": 0,
                "received_wire_bytes": 50,
            }
        ],
    }
    rank1 = {
        **rank0,
        "rank": 1,
        "peak_allocated_bytes": 12,
        "_step_times": [],
        "_profiles": [
            {
                **rank0["_profiles"][0],
                "state_all_gather_ms": 7,
                "optimizer_ms": 28,
                "other_ms": 5,
            }
        ],
    }

    result = aggregate_external([[rank0], [rank1]])[0]

    assert result["mean_step_ms"] == 100.0
    assert result["state_all_gather_ms"] == 7.0
    assert result["peak_allocated_bytes"] == 12


def test_write_results_adds_requested_derived_timings(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path)
    row = {
        "model": "9.9b-pp2",
        "method": "muon_fp8_states",
        "states": "fp8",
        "mean_step_ms": 100.0,
        "optimizer_ms": 60.0,
        "newton_schulz_ms": 30.0,
        "wire_encode_ms": 2.0,
        "state_all_gather_ms": 10.0,
        "wire_decode_ms": 3.0,
        "other_ms": 15.0,
        "peak_allocated_bytes": 2**30,
    }

    bench.write_results(args, [row])

    assert row["outside_optimizer_ms"] == 40.0
    assert row["newton_schulz_pipeline_ms"] == 33.0
    assert row["total_state_communication_ms"] == 12.0
    table = (tmp_path / "results.md").read_text()
    assert "Outside optimizer" in table
    assert "Newton-Schulz incl. preparation" in table
    assert "Total state communication" in table
    assert "Wire decode" not in table


def test_repeat_aggregation_reports_mean_and_sample_std():
    base = {
        "model": "5.0b-wide",
        "method": "muon_fp8_states",
        "states": "fp8",
        "status": "ok",
        "samples": 2,
        "peak_allocated_bytes": 10,
        "mean_step_ms": 100.0,
        "optimizer_ms": 60.0,
        "newton_schulz_ms": 30.0,
        "wire_encode_ms": 2.0,
        "state_all_gather_ms": 10.0,
        "wire_decode_ms": 3.0,
        "other_ms": 15.0,
    }
    rows = [base, {**base, "mean_step_ms": 104.0, "optimizer_ms": 62.0}]

    result = aggregate_repeats(rows)[0]

    assert result["repeats"] == 2
    assert result["samples"] == 4
    assert result["mean_step_ms"] == 102.0
    assert result["mean_step_ms_std"] == 2.8284
    assert result["newton_schulz_pipeline_ms"] == 33.0
    assert result["total_state_communication_ms"] == 12.0


def test_cloud_launcher_defaults_to_local_transformer_backend():
    launcher = Path("cloud_benchmark_muon_state_communication.sh").read_text()

    assert "transformer_impl=${TRANSFORMER_IMPL:-local}" in launcher
    assert '--transformer-impl "$transformer_impl"' in launcher
    assert 'make -C "$root/third_party/Megatron-LM/megatron/core/datasets"' in launcher


def test_16gpu_launcher_uses_two_nodes_and_cross_node_dp_mapping():
    launcher = Path("cloud_benchmark_muon_state_communication_16gpu.sh").read_text()

    assert 'if [[ "$nnodes" != 2 ]]' in launcher
    assert "--nnodes=2 --nproc-per-node=8" in launcher
    assert "--nnodes=2 --nproc-per-node=4" in launcher
    assert "--data-parallel-size 16" in launcher
    assert "--pipeline-parallel-size 2 --data-parallel-size 8" in launcher
    assert "--use-tp-pp-dp-mapping" in launcher
    assert "benchmark_multinode_collectives.py" in launcher


def test_16gpu_phase_barrier_outlives_benchmark_runtime():
    from scripts.benchmark_multinode_coordination import coordination_timeout_seconds

    launcher = Path("cloud_benchmark_muon_state_communication_16gpu.sh").read_text()
    assert "BENCHMARK_COORDINATION_TIMEOUT_SECONDS=18000" in launcher
    assert coordination_timeout_seconds({"BENCHMARK_COORDINATION_TIMEOUT_SECONDS": "18000"}) == 18000


def test_16gpu_launcher_resolves_two_nodes_from_sixteen_mpi_ranks():
    script = Path("cloud_benchmark_muon_state_communication_16gpu.sh")
    layouts = (
        (0, 0, 0, 1),
        (7, 7, 0, 0),
        (8, 0, 1, 1),
        (15, 7, 1, 0),
    )
    for world_rank, local_rank, node_rank, leader in layouts:
        env = os.environ | {
            "OMPI_COMM_WORLD_RANK": str(world_rank),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(local_rank),
            "OMPI_COMM_WORLD_SIZE": "16",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "8",
            "MLSUB_LAYOUT_ONLY": "1",
        }
        result = subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == (
            f"world_rank={world_rank} local_rank={local_rank} world_size=16 "
            f"local_size=8 node_rank={node_rank} nnodes=2 leader={leader}"
        )


def test_multinode_coordination_without_mpi4py(tmp_path):
    import ipaddress
    import sys

    helper = Path("scripts/benchmark_multinode_coordination.py")
    env = os.environ | {"BENCHMARK_WORLD_SIZE": "4"}
    addresses = [
        subprocess.Popen(
            [sys.executable, str(helper), "address", str(tmp_path)],
            env=env | {"BENCHMARK_WORLD_RANK": str(rank)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in reversed(range(4))
    ]
    resolved = []
    for process in addresses:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        resolved.append(stdout.strip())
    assert len(set(resolved)) == 1
    ipaddress.ip_address(resolved[0])

    processes = [
        subprocess.Popen(
            [sys.executable, str(helper), "sync", str(tmp_path), "smoke", str(rank % 3)],
            env=env | {"BENCHMARK_WORLD_RANK": str(rank)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(4)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "2"


def test_external_repeats_wait_for_all_ranks_and_propagate_failure(tmp_path):
    import sys

    code = (
        "from pathlib import Path; "
        "from scripts.benchmark_muon_state_communication import _sync_external_repeat; "
        "import os, time; "
        "rank = int(os.environ['RANK']); "
        "time.sleep(0.5 if rank == 1 else 0); "
        f"print(_sync_external_repeat(Path({str(tmp_path)!r}), '4.9b', "
        "'muon_bf16_states', 0, rank, 2, rank == 1))"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=os.environ | {"RANK": str(rank), "WORLD_SIZE": "2", "PYTHONPATH": "."},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "1"


def test_external_benchmark_does_not_overlap_successive_repeats(tmp_path):
    import sys

    code = f"""
import os
import time
from pathlib import Path
from types import SimpleNamespace
from scripts import benchmark_muon_state_communication as bench

output = Path({str(tmp_path)!r})
rank = int(os.environ['RANK'])
bench.parse_args = lambda: SimpleNamespace(
    models=['4.9b'], methods=['muon_bf16_states'], repeats=2,
    tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=2,
    output_dir=output,
)
def fake_run_one(root, args, model, method, repeat):
    if rank == 1 and repeat == 0:
        time.sleep(0.5)
        (output / 'rank1_finished_repeat0').write_text('yes')
    if rank == 0 and repeat == 1:
        assert (output / 'rank1_finished_repeat0').exists()
    return {{'status': 'ok', 'mean_step_ms': 1, 'state_all_gather_ms': 1, 'rank': rank}}
bench.run_one = fake_run_one
bench.aggregate_external = lambda rows: []
bench.write_results = lambda args, rows: None
raise SystemExit(bench.main())
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=os.environ | {"RANK": str(rank), "WORLD_SIZE": "2", "PYTHONPATH": "."},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, f"{stdout}\n{stderr}"


def test_pretrain_child_uses_fresh_store_for_each_repeat(tmp_path):
    from scripts.benchmark_muon_state_communication import _pretrain_environment

    parent = os.environ | {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29618",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
    }
    first = _pretrain_environment(parent, tmp_path, "4.9b", "muon_bf16_states", 0)
    second = _pretrain_environment(parent, tmp_path, "4.9b", "muon_bf16_states", 1)

    assert first["MASTER_PORT"] != parent["MASTER_PORT"]
    assert first["MASTER_PORT"] != second["MASTER_PORT"]
    assert "TORCHELASTIC_USE_AGENT_STORE" not in first
    assert first["MASTER_ADDR"] == parent.get("MASTER_ADDR")


def test_dataset_helper_can_use_torch_bundled_pybind11_headers():
    makefile = Path(
        "third_party/Megatron-LM/megatron/core/datasets/Makefile"
    ).read_text()

    assert "python3 -m pybind11 --includes" in makefile
    assert "torch.__path__[0] + '/include -I'" in makefile


def test_pipeline_rank_can_complete_from_optimizer_profiles_without_step_log():
    row = {
        "samples": 0,
        "_profiles": [{"step": 10}, {"step": 11}, {"step": 12}],
    }

    assert _has_complete_samples(row, measure_steps=3, external_distributed=True)
    assert not _has_complete_samples(row, measure_steps=3, external_distributed=False)
