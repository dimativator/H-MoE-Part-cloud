"""Independent normalization and probability ablations of dense LionMuon."""
import json
from pathlib import Path
import random

import torch

from third_party.lion_muon.lion_muon import LionMuon
from third_party.lion_muon.muon import zeropower_via_newtonschulz5


class LionMuonAblation(LionMuon):
    def __init__(self, *args, variant: str, total_steps: int,
                 neuron_beta: float = 0.95, probability_seed: int = 20261117, **kwargs):
        if variant not in ("rms", "neuron_half", "neuron_full", "probability_down"):
            raise ValueError("Unknown LionMuon ablation")
        if total_steps < 2 or not 0 <= neuron_beta < 1:
            raise ValueError("Invalid horizon or neuron variance decay")
        if kwargs.get("momentum_rank", 0) or kwargs.get("srank_alpha", 0):
            raise ValueError("Ablations require dense momentum and no adaptive schedule")
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.total_steps = total_steps
        self.neuron_beta = neuron_beta
        self.probability_rng = random.Random(probability_seed)
        self.muon_count = 0
        self.last_muon = False
        self.last_probability = 0.2

    def _is_muon_step(self):
        if self.variant == "probability_down":
            fraction = min(self._step_count / (self.total_steps - 1), 1.0)
            self.last_probability = 0.35 - 0.30 * fraction
            self.last_muon = self.probability_rng.random() < self.last_probability
        else:
            self.last_muon = super()._is_muon_step()
        self.muon_count += int(self.last_muon)
        return self.last_muon

    def _muon_update(self, update, state, group):
        if self.variant == "probability_down":
            return super()._muon_update(update, state, group)
        update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"]).float()
        alpha = {"rms": 0.0, "neuron_half": 0.5, "neuron_full": 1.0}[self.variant]
        if alpha:
            # Rows refer to output neurons in the original [out, in] matrix.
            statistic = update.square().mean(dim=1, keepdim=True)
            if "neuron_variance" not in state:
                state["neuron_variance"] = torch.zeros_like(statistic)
            variance = state["neuron_variance"]
            variance.lerp_(statistic, 1 - self.neuron_beta)
            update = update / (variance + 1e-8).pow(alpha / 2)
        return update * (0.2 / (update.square().mean().sqrt() + 1e-8))

    def get_step_metrics(self) -> dict:
        return {"lion_ablation/is_muon": int(self.last_muon),
                "lion_ablation/muon_probability": self.last_probability,
                "lion_ablation/muon_steps": self.muon_count,
                "lion_ablation/muon_fraction": self.muon_count / max(self._step_count, 1)}


SLIMNORM_TRIALS = ("neuron_no_rms", "neuron_rms_down", "neuron_no_rms_down",
                   "slim_no_rms", "slim_rms_down", "slim_no_rms_down")


class LionMuonSlimNorm(LionMuonAblation):
    """Alpha=1 ablations of final RMS, probability, and SlimAdam variance axes.

    Only hidden post-NS variance changes. Dense momentum and auxiliary AdamW
    remain identical to LionMuonAblation. Variance is updated on Muon steps.
    """

    def __init__(self, muon_params, *, variant: str, named_parameters: dict,
                 total_steps: int, **kwargs):
        if variant not in SLIMNORM_TRIALS:
            raise ValueError("Unknown SlimNorm trial")
        super().__init__(muon_params, variant="neuron_full", total_steps=total_steps, **kwargs)
        self.trial = variant
        self.final_rms = variant in ("neuron_rms_down", "slim_rms_down")
        self.probability_down = variant.endswith("_down")
        self.slim_variance = variant.startswith("slim_")
        layer_map = json.loads((Path(__file__).parent /
            "memory_efficient/slim_adam/layer_map_llama_paper.json").read_text())
        names = {id(p): name for name, p in named_parameters.items()}
        manifest = []
        for p in muon_params:
            name = names.get(id(p))
            kinds = [kind for kind, patterns in layer_map.items()
                     if name is not None and any(name.endswith(pattern) for pattern in patterns)]
            if len(kinds) != 1 or kinds[0] not in (
                "attention_query", "attention_key", "attention_value", "attention_output",
                "mlp_up", "mlp_gate", "mlp_down") or p.ndim != 2:
                raise ValueError(f"Unmapped hidden variance: {name}, {kinds}")
            axis = 1 if not self.slim_variance or kinds[0] in ("attention_query", "attention_key") else 0
            self.state[p]["variance_axis"] = axis
            shape = list(p.shape)
            shape[axis] = 1
            manifest.append(dict(name=name, axis=axis, variance_shape=shape))
        print("LION_SLIMNORM_COVERAGE=" + json.dumps(manifest), flush=True)

    def _is_muon_step(self):
        if self.probability_down:
            fraction = min(self._step_count / (self.total_steps - 1), 1.0)
            self.last_probability = 0.35 - 0.30 * fraction
            self.last_muon = self.probability_rng.random() < self.last_probability
        else:
            self.last_muon = LionMuon._is_muon_step(self)
        self.muon_count += int(self.last_muon)
        return self.last_muon

    def _muon_update(self, update, state, group):
        update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"]).float()
        statistic = update.square().mean(dim=state["variance_axis"], keepdim=True)
        if "neuron_variance" not in state:
            state["neuron_variance"] = torch.zeros_like(statistic)
        variance = state["neuron_variance"]
        variance.lerp_(statistic, 1 - self.neuron_beta)
        update = update / (variance + 1e-8).sqrt()
        if self.final_rms:
            update = update * (0.2 / (update.square().mean().sqrt() + 1e-8))
        return update
