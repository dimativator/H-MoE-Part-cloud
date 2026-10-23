"""NorMuon with SlimAdam's layer-specific second-moment reductions.

Hidden states track squared post-NS updates. Auxiliary states track squared
gradients with AdamW bias corrections. Full first moments are preserved.
"""

import json
import math
from pathlib import Path

from torch import Tensor

from .adaptive_muon import NorMuon
from third_party.normuon.kernels import zeropower_via_newtonschulz5


# Same static axes as SlimAdam's DEFAULT_COMPRESSION_RULES. Kept local to
# avoid importing the memory_efficient package and its unrelated dependencies.
COMPRESSION_RULES = dict(attention_query=(1,), attention_key=(1,),
                         attention_value=(0,), attention_output=(0,),
                         mlp_up=(0,), mlp_gate=(0,), mlp_down=(0,),
                         token_embedding=(1,), lm_head=(1,), norm=None)

LAYER_MAP = Path(__file__).parent / "memory_efficient/slim_adam/layer_map_llama_paper.json"


def compression_groups(groups: list[dict], named: dict[str, Tensor] | None) -> list[dict]:
    """Resolve every parameter explicitly; never silently use full variance."""
    if named is None:
        raise ValueError("SlimNorMuon requires named parameters for compression coverage.")
    layer_map = json.loads(LAYER_MAP.read_text())
    names = {id(p): name for name, p in named.items()}
    result = []
    seen = set()
    for group in groups:
        for p in group["params"]:
            name = names.get(id(p))
            matches = [kind for kind, patterns in layer_map.items()
                       if name is not None and any(name.endswith(pattern) for pattern in patterns)]
            if len(matches) != 1 or id(p) in seen:
                raise ValueError(f"Expected unique compression mapping and parameter: {name}, {matches}")
            seen.add(id(p))
            kind = matches[0]
            dims = COMPRESSION_RULES[kind]
            if (kind == "norm" and (p.ndim != 1 or group.get("is_proj_params", True))) or (
                kind != "norm" and p.ndim != 2
            ):
                raise ValueError(f"Unexpected parameter shape or optimizer role: {name}")
            auxiliary = kind in ("token_embedding", "lm_head", "norm")
            if auxiliary == group.get("is_proj_params", True):
                raise ValueError(f"Unexpected hidden/auxiliary assignment: {name}")
            result.append({**group, "params": [p], "compress_dims": dims, "param_name": name})
    manifest = []
    for group in result:
        shape = list(group["params"][0].shape)
        for dim in group["compress_dims"] or ():
            shape[dim] = 1
        manifest.append(dict(name=group["param_name"], compress_dims=group["compress_dims"], variance_shape=shape))
    print("SLIM_NORMUON_COVERAGE=" + json.dumps(manifest), flush=True)
    return result


class SlimNorMuon(NorMuon):
    """Keep native NorMuon normalization, changing only variance axes."""

    def _variance_template(self, g: Tensor, group: dict) -> Tensor:
        dims = group["compress_dims"]
        return g.mean(dim=dims, keepdim=True) if dims is not None else g

    def _squared_statistic(self, g: Tensor, group: dict) -> Tensor:
        dims = group["compress_dims"]
        squared = g.square()
        return squared.mean(dim=dims, keepdim=True) if dims is not None else squared

    def _matrix_update(self, g: Tensor, state: dict, group: dict) -> Tensor:
        state["momentum"].lerp_(g, 1 - group["momentum"])
        direction = g.lerp(state["momentum"], group["momentum"]) if group["nesterov"] else state["momentum"]
        update = zeropower_via_newtonschulz5(direction, steps=group["ns_steps"]).float()
        original_norm = update.norm(dim=(-2, -1), keepdim=True)
        state["variance"].lerp_(self._squared_statistic(update, group), 1 - group["beta2"])
        update.mul_(1 / (state["variance"].sqrt() + 1e-10))
        update.mul_(original_norm / (update.norm(dim=(-2, -1), keepdim=True) + 1e-10))
        return update * math.sqrt(max(1, g.size(0) / g.size(1)))
