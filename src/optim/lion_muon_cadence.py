"""LionMuon branch cadence and shared-momentum readout ablations."""
import math
import re

from optim.lion_muon_ablation import LionMuonAblation


TRIALS = ("regular", "constant_random", "muon_ema", "linear_zero", "half_cosine",
          "stagger_uniform", "stagger_early", "stagger_middle", "stagger_late")


class LionMuonCadence(LionMuonAblation):
    def __init__(self, matrices, *, variant, named_parameters, **kwargs):
        if variant not in TRIALS:
            raise ValueError(variant)
        super().__init__(matrices, variant="probability_down", **kwargs)
        self.trial = variant
        self.stagger = variant.startswith("stagger_")
        self.block = {}
        n = 1
        if self.stagger:
            names = {id(p): name for name, p in named_parameters.items()}
            for p in matrices:
                match = re.search(r"(?:^|\.)h\.(\d+)\.", names[id(p)])
                if match is None:
                    raise ValueError(f"Missing block for {names[id(p)]}")
                self.block[p] = int(match[1])
            n = max(self.block.values()) + 1
            if n % 3 or set(self.block.values()) != set(range(n)):
                raise ValueError("Expected contiguous blocks in three equal groups")
        self.weights = [1.0] * n
        if self.stagger and variant != "stagger_uniform":
            favored = ("stagger_early", "stagger_middle", "stagger_late").index(variant)
            self.weights = [1.5 if i // (n // 3) == favored else 0.75 for i in range(n)]
        self.phase = [i / n for i in range(n)]
        self.fires = [False] * n
        self.block_counts = [0] * n

    def _is_muon_step(self):
        x = min(self._step_count / (self.total_steps - 1), 1.0)
        p = 0.35 - 0.30 * x
        if self.trial == "constant_random":
            p = 0.20
        elif self.trial == "linear_zero":
            p = 0.40 * (1 - x)
        elif self.trial == "half_cosine":
            p = 0.20 + 0.15 * math.cos(math.pi * x)
        self.last_probability = p
        if self.stagger or self.trial == "regular":
            for i, weight in enumerate(self.weights):
                self.phase[i] += p * weight
                self.fires[i] = self.phase[i] >= 1
                self.phase[i] -= int(self.fires[i])
        else:
            self.fires[0] = self.probability_rng.random() < p
        for i, fired in enumerate(self.fires):
            self.block_counts[i] += int(fired)
        return self.fires[0]

    def _matrix_branch(self, param, global_branch):
        return self.fires[self.block[param]] if self.stagger else global_branch

    def _direction_beta(self, group, muon):
        if self.trial == "muon_ema" and muon:
            return group["beta2"]
        return group["beta1"]

    def get_step_metrics(self):
        return {"lion_cadence/base_probability": self.last_probability,
                "lion_cadence/block_fraction": sum(self.fires) / len(self.fires),
                **{f"lion_cadence/muon_count_{i}": count for i, count in enumerate(self.block_counts)}}
