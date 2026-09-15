import argparse
import atexit
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCORE_ROOT = ROOT / "third_party" / "Megatron-LM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MCORE_ROOT))


def take_state_precision():
    flag = "--optimizer-state-precision"
    index = sys.argv.index(flag)
    precision = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    if precision not in {"fp32", "bfloat16", "fp8"}:
        raise ValueError(f"unsupported optimizer-state precision: {precision}")
    return precision


precision = take_state_precision()

import megatron.core.optimizer.emerging_optimizers as mcore_eopt
import megatron.training.arguments as mcore_arguments
import megatron.core.optimizer.optimizer as mcore_optimizer_base
from stage4.ademamix import AdEMAMix
from stage4.state_sharded_muon import FrugalMuonMuon, StateShardedMuon


def _ademamix_config_to_kwargs(config, model_chunks, pg_collection):
    del model_chunks, pg_collection
    return {
        "lr": config.lr,
        "betas": (config.adam_beta1, config.adam_beta2, 0.9999),
        "alpha": 8.0,
        "eps": config.adam_eps,
        "weight_decay": config.weight_decay,
        "beta3_warmup_steps": None,
        "alpha_warmup_steps": None,
    }


mcore_eopt._EMERGING_OPTIMIZERS["ademamix"] = mcore_eopt.EmergingOptimizerEntry(
    optimizer_cls=AdEMAMix,
    config_to_kwargs=_ademamix_config_to_kwargs,
    default_param_overrides={},
)


def _state_sharded_muon_config_to_kwargs(config, model_chunks, pg_collection):
    kwargs = mcore_eopt._muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(
        {
            "state_precision": "fp8" if precision == "fp8" else "bfloat16",
            "distributed_state_sharding": config.muon_distributed_state_sharding,
            "profile_state_communication": config.muon_profile_state_communication,
            "fp8_bucket_bytes": config.muon_fp8_bucket_bytes,
            "fused_fp8_ns_input": config.muon_fused_fp8_ns_input,
            "frugal_density": config.frugal_density,
            "frugal_update_gap": config.frugal_update_gap,
            "frugal_coord_choice": config.frugal_coord_choice,
            "frugal_inactive_lr_scale": config.frugal_inactive_lr_scale,
        }
    )
    return kwargs


mcore_eopt._EMERGING_OPTIMIZERS["muon"].optimizer_cls = StateShardedMuon
mcore_eopt._EMERGING_OPTIMIZERS["muon"].config_to_kwargs = (
    _state_sharded_muon_config_to_kwargs
)
mcore_eopt._EMERGING_OPTIMIZERS["frugal_muon_muon"] = mcore_eopt.EmergingOptimizerEntry(
    optimizer_cls=FrugalMuonMuon,
    config_to_kwargs=_state_sharded_muon_config_to_kwargs,
    default_param_overrides={},
)


def _raw_state_sharded_muon(optimizer):
    raw = getattr(optimizer, "optimizer", optimizer)
    return raw if isinstance(raw, StateShardedMuon) else None


_chained_init = mcore_optimizer_base.ChainedOptimizer.__init__
_chained_step = mcore_optimizer_base.ChainedOptimizer._step


def _profiled_chained_init(self, chained_optimizers):
    _chained_init(self, chained_optimizers)
    for optimizer in chained_optimizers:
        raw = _raw_state_sharded_muon(optimizer)
        if raw is not None:
            raw._defer_profile_emit = True


def _profiled_chained_step(self):
    profiled = [
        raw
        for raw in (_raw_state_sharded_muon(opt) for opt in self.chained_optimizers)
        if raw is not None and raw.state_comm.profile_enabled
    ]
    if not profiled:
        return _chained_step(self)
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    success = _chained_step(self)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    for raw in profiled:
        raw.flush_state_comm_profile(elapsed_ms)
    return success


mcore_optimizer_base.ChainedOptimizer.__init__ = _profiled_chained_init
mcore_optimizer_base.ChainedOptimizer._step = _profiled_chained_step

# MCore registers SOAP through the generic fallback path, which leaves three gaps.
# Without the param overrides the tied 50,304x1,536 embedding is handed to SOAP,
# whose Kronecker factor and eigenbasis for that parameter are both 50,304x50,304
# (~10 GB each); SOAP's `eps` has no OptimizerConfig field, so `--adam-eps` never
# reaches it; and the three `soap_*` config fields have no command-line flags.


def _soap_config_to_kwargs(config, model_chunks, pg_collection):
    kwargs = mcore_eopt._default_adam_based_eopt_config_to_kwargs(
        "soap", config, model_chunks, pg_collection
    )
    kwargs["eps"] = config.adam_eps
    return kwargs


soap_entry = mcore_eopt._EMERGING_OPTIMIZERS["soap"]
soap_entry.config_to_kwargs = _soap_config_to_kwargs
soap_entry.default_param_overrides = mcore_eopt._default_param_overrides_factory()


def add_stage4_args(parser):
    for action in parser._actions:
        if action.dest == "optimizer":
            for optimizer in ("ademamix", "frugal_muon_muon"):
                if optimizer not in action.choices:
                    action.choices = [*action.choices, optimizer]
    group = parser.add_argument_group(title="stage4 soap")
    group.add_argument("--soap-shampoo-beta", type=float, default=0.95)
    group.add_argument("--soap-precondition-frequency", type=int, default=1)
    group.add_argument(
        "--soap-use-kl-shampoo", action=argparse.BooleanOptionalAction, default=True
    )
    group = parser.add_argument_group(title="stage4 distributed Muon")
    group.add_argument(
        "--muon-distributed-state-sharding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="row-shard Muon momentum across the DP group and all-gather before NS",
    )
    group.add_argument(
        "--muon-profile-state-communication",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    group.add_argument(
        "--muon-fp8-bucket-bytes",
        type=int,
        default=0,
        help="pack persistent FP8 state packets into communication buckets; 0 disables",
    )
    group.add_argument(
        "--muon-fused-fp8-ns-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fuse FP8 decode, Nesterov addition, normalization, and BF16 NS input",
    )
    group.add_argument("--frugal-density", type=float, default=0.25)
    group.add_argument("--frugal-update-gap", type=int, default=50)
    group.add_argument("--frugal-coord-choice", choices=("columns",), default="columns")
    group.add_argument("--frugal-inactive-lr-scale", type=float, default=1.0)
    return parser


mcore_parse_and_validate_args = mcore_arguments.parse_and_validate_args


def parse_and_validate_args(extra_args_provider=None, **kwargs):
    def provider(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return add_stage4_args(parser)

    return mcore_parse_and_validate_args(provider, **kwargs)


mcore_arguments.parse_and_validate_args = parse_and_validate_args

if precision == "fp8":
    import megatron.core.optimizer as mcore_optimizer
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS

    from stage4.fp8_optimizer_states import (
        make_fp8_ademamix,
        make_fp8_adamw,
        make_fp8_soap,
    )

    mcore_optimizer.Adam = make_fp8_adamw(mcore_optimizer.Adam)
    _EMERGING_OPTIMIZERS["ademamix"].optimizer_cls = make_fp8_ademamix(AdEMAMix)
    _EMERGING_OPTIMIZERS["soap"].optimizer_cls = make_fp8_soap(
        _EMERGING_OPTIMIZERS["soap"].optimizer_cls
    )

print(f"stage4 optimizer-state precision: {precision}", flush=True)

if os.getenv("STAGE4_REPORT_PEAK_MEMORY") == "1":

    def report_peak_memory():
        import torch

        print(
            "stage4 peak memory bytes: "
            f"allocated={torch.cuda.max_memory_allocated()} "
            f"reserved={torch.cuda.max_memory_reserved()}",
            flush=True,
        )

    atexit.register(report_peak_memory)

runpy.run_path(str(MCORE_ROOT / "pretrain_gpt.py"), run_name="__main__")
