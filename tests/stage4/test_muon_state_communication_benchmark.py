from pathlib import Path

from scripts.benchmark_muon_state_communication import (
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
