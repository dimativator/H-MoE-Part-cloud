from pathlib import Path

from scripts.benchmark_training_step import (
    MODELS,
    OPTIMIZERS,
    build_command,
    periodic_updates_in_window,
)


def test_benchmark_matrix_contains_requested_models_and_optimizers():
    assert MODELS == {
        "257m": {"layers": 12, "hidden": 1024, "ffn": 2816, "heads": 8},
        "500m": {"layers": 18, "hidden": 1280, "ffn": 3584, "heads": 20},
        "1b": {"layers": 16, "hidden": 2048, "ffn": 5504, "heads": 16},
        "3b": {"layers": 24, "hidden": 3072, "ffn": 8192, "heads": 24},
        "5b": {"layers": 32, "hidden": 3456, "ffn": 9216, "heads": 27},
    }
    assert set(OPTIMIZERS) == {
        "adam",
        "muon",
        "soap",
        "ademamix",
        "galore",
        "frugal",
        "frugal_muon_muon",
        "slim_adam",
        "apollo",
    }


def test_measurement_window_contains_one_gap_50_update():
    assert periodic_updates_in_window(10, 50, 50) == 1


def test_pipeline_parallelism_sets_world_size_and_periodic_flags():
    command = build_command(
        root=Path("."),
        model_name="5b",
        precision="bf16",
        optimizer="galore",
        global_batch=32,
        micro_batch=2,
        data_parallel_size=1,
        pipeline_parallel_size=4,
        warmup=10,
        measured=50,
        periodic_update_gap=50,
        projection_density=0.25,
    )

    assert command[command.index("--nproc-per-node") + 1] == "4"
    assert command[command.index("--pipeline-model-parallel-size") + 1] == "4"
    assert command[command.index("--frugal-update-gap") + 1] == "50"
    assert command[command.index("--soap-precondition-frequency") + 1] == "50"
    assert command[command.index("--frugal-density") + 1] == "0.25"


def test_muon_syrk_flag_is_forwarded_only_when_enabled():
    common = {
        "root": Path("."),
        "model_name": "500m",
        "precision": "fp8_act",
        "optimizer": "muon",
        "global_batch": 128,
        "micro_batch": 32,
        "data_parallel_size": 1,
        "warmup": 2,
        "measured": 2,
    }

    baseline = build_command(**common, muon_use_syrk=False)
    optimized = build_command(**common, muon_use_syrk=True)

    assert "--muon-use-syrk" not in baseline
    assert optimized.count("--muon-use-syrk") == 1


def test_muon_syrk_flag_is_not_forwarded_to_other_optimizers():
    command = build_command(
        root=Path("."),
        model_name="500m",
        precision="fp8_act",
        optimizer="adam",
        global_batch=128,
        micro_batch=32,
        data_parallel_size=1,
        warmup=2,
        measured=2,
        muon_use_syrk=True,
    )

    assert "--muon-use-syrk" not in command


def test_muon_batched_newton_schulz_flag_is_forwarded_only_when_enabled():
    common = {
        "root": Path("."),
        "model_name": "500m",
        "precision": "fp8_act",
        "optimizer": "muon",
        "global_batch": 128,
        "micro_batch": 32,
        "data_parallel_size": 1,
        "warmup": 2,
        "measured": 2,
    }

    baseline = build_command(**common, muon_batched_newton_schulz=False)
    optimized = build_command(**common, muon_batched_newton_schulz=True)

    assert "--muon-batched-newton-schulz" not in baseline
    assert optimized.count("--muon-batched-newton-schulz") == 1


def test_muon_batched_newton_schulz_flag_is_not_forwarded_to_other_optimizers():
    command = build_command(
        root=Path("."),
        model_name="500m",
        precision="fp8_act",
        optimizer="adam",
        global_batch=128,
        micro_batch=32,
        data_parallel_size=1,
        warmup=2,
        measured=2,
        muon_batched_newton_schulz=True,
    )

    assert "--muon-batched-newton-schulz" not in command
