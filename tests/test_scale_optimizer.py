import torch

from optim.memory_efficient.scale import SCALE


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(3, 2)
        self.transformer.attn = torch.nn.Module()
        self.transformer.attn.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.transformer.ln_f = torch.nn.LayerNorm(2)
        self.lm_head = torch.nn.Linear(2, 3, bias=False)


def test_scale_partition_and_update():
    model = TinyModel()
    optimizer = SCALE(model.named_parameters(), lr=0.1, weight_decay=0, momentum=0.9)
    names = dict(model.named_parameters())
    for param in names.values():
        param.grad = torch.ones_like(param)
    before = {name: param.detach().clone() for name, param in names.items()}
    optimizer.step()

    assert optimizer.state[names["transformer.wte.weight"]]["param_type"] == "main_param"
    assert optimizer.state[names["transformer.attn.q_proj.weight"]]["param_type"] == "main_param"
    assert optimizer.state[names["lm_head.weight"]]["param_type"] == "secondary_param"
    assert optimizer.state[names["transformer.ln_f.weight"]]["param_type"] == "oned_param"
    for name, param in names.items():
        torch.testing.assert_close(param, before[name] - 0.1)
    assert "moment1" not in optimizer.state[names["transformer.attn.q_proj.weight"]]
    assert "moment1" in optimizer.state[names["lm_head.weight"]]
    assert "moment2" in optimizer.state[names["transformer.ln_f.weight"]]


def test_scale_checkpoint_restores_momenta():
    model = TinyModel()
    optimizer = SCALE(model.named_parameters(), lr=0.01, weight_decay=0.1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    checkpoint = optimizer.state_dict()
    restored = SCALE(model.named_parameters(), lr=0.01, weight_decay=0.1)
    restored.load_state_dict(checkpoint)
    assert restored.state[model.lm_head.weight]["param_type"] == "secondary_param"
    torch.testing.assert_close(
        restored.state[model.lm_head.weight]["moment1"],
        optimizer.state[model.lm_head.weight]["moment1"],
    )
    restored.step()


def test_scale_embedding_uses_upstream_axis_zero():
    model = TinyModel()
    optimizer = SCALE(model.named_parameters(), lr=0.1, weight_decay=0)
    param = model.transformer.wte.weight
    param.grad = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    before = param.detach().clone()
    optimizer.step()
    expected = param.grad / param.grad.square().mean(dim=0, keepdim=True).sqrt()
    torch.testing.assert_close(param, before - 0.1 * expected)
