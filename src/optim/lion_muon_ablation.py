"""Independent normalization and probability ablations of dense LionMuon."""
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
