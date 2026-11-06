"""Activation checkpointing / offload leave the BF16 Llama numerics unchanged, and the
memory profiler attributes persistent bytes correctly. Runs on CPU; the FP8 (COAT)
path and the CUDA peak-memory numbers need a GPU and are not covered here."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import tiktoken  # noqa: F401  (imported by models; unused in these tests)
except ImportError:
    sys.modules["tiktoken"] = types.ModuleType("tiktoken")

from models.llama import Llama
from optim.mem_profile import MemProfiler, persistent_bytes


def config(**overrides):
    cfg = dict(
        vocab_size=97, sequence_length=16, n_layer=3, n_embd=32, n_head=4,
        multiple_of=16, dropout=0.0, rmsnorm_eps=1e-5, init_std=0.02,
        activation_checkpointing="none", activation_offload=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def loss_and_grads(cfg, idx, targets):
    torch.manual_seed(0)
    model = Llama(cfg).train()
    loss = model(idx, targets=targets)["loss"]
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}


class ActivationMemoryTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.idx = torch.randint(0, 97, (2, 16), generator=g)
        self.targets = torch.randint(0, 97, (2, 16), generator=g)
        self.ref_loss, self.ref_grads = loss_and_grads(config(), self.idx, self.targets)

    def assert_same(self, **overrides):
        loss, grads = loss_and_grads(config(**overrides), self.idx, self.targets)
        torch.testing.assert_close(loss, self.ref_loss, rtol=0, atol=0)
        for name, grad in grads.items():
            torch.testing.assert_close(grad, self.ref_grads[name], rtol=1e-6, atol=1e-7, msg=name)

    def test_mlp_checkpointing(self):
        self.assert_same(activation_checkpointing="mlp")

    def test_full_checkpointing(self):
        self.assert_same(activation_checkpointing="full")

    def test_offload(self):
        self.assert_same(activation_offload=True)

    def test_full_checkpointing_with_offload(self):
        self.assert_same(activation_checkpointing="full", activation_offload=True)

    def test_mlp_checkpointing_with_offload(self):
        self.assert_same(activation_checkpointing="mlp", activation_offload=True)

    def test_dropout_masks_are_replayed(self):
        cfg = dict(dropout=0.1)
        torch.manual_seed(3)
        ref = loss_and_grads(config(**cfg), self.idx, self.targets)
        torch.manual_seed(3)
        out = loss_and_grads(config(activation_checkpointing="full", **cfg), self.idx, self.targets)
        torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)

    def saved_bytes(self, **overrides):
        """Bytes autograd saves outside recomputed or offloaded regions."""
        total = 0

        def pack(t):
            nonlocal total
            total += t.numel() * t.element_size()
            return t

        model = Llama(config(**overrides)).train()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(self.idx, targets=self.targets)["loss"]
        return total

    def test_checkpointing_and_offload_engage(self):
        none = self.saved_bytes()
        mlp = self.saved_bytes(activation_checkpointing="mlp")
        full = self.saved_bytes(activation_checkpointing="full")
        self.assertLess(mlp, none)
        self.assertLess(full, mlp)
        # save_on_cpu is the innermost hook, so nothing reaches the counting hook.
        self.assertEqual(self.saved_bytes(activation_offload=True), 0)

    def test_eval_ignores_checkpointing(self):
        model = Llama(config(activation_checkpointing="full", activation_offload=True)).eval()
        with torch.no_grad():
            out = model(self.idx, targets=self.targets)
        self.assertTrue(torch.isfinite(out["loss"]))


class PersistentBytesTest(unittest.TestCase):
    def test_adamw_state_and_grads(self):
        model = Llama(config())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        before = persistent_bytes(model, opt)
        self.assertEqual(before["params"], n_bytes)
        self.assertEqual(before["grads"], 0)
        self.assertEqual(before["optimizer_state"], 0)

        model(torch.randint(0, 97, (2, 16)), targets=torch.randint(0, 97, (2, 16)))["loss"].backward()
        opt.step()
        after = persistent_bytes(model, opt)
        self.assertEqual(after["grads"], n_bytes)
        n_params = len(list(model.parameters()))
        step_bytes = sum(s["step"].numel() * s["step"].element_size() for s in opt.state.values())
        self.assertEqual(after["optimizer_state"], 2 * n_bytes + step_bytes)
        self.assertEqual(len(opt.state), n_params)

    def test_wrapped_optimizers_are_counted(self):
        model = Llama(config())
        params = list(model.parameters())
        inner = [torch.optim.AdamW(params[:2], lr=1e-3), torch.optim.SGD(params[2:], lr=1e-3, momentum=0.9)]
        wrapper = SimpleNamespace(optimizers=inner)
        model(torch.randint(0, 97, (2, 16)), targets=torch.randint(0, 97, (2, 16)))["loss"].backward()
        for o in inner:
            o.step()
        expected = sum(persistent_bytes(model, o)["optimizer_state"] for o in inner)
        self.assertEqual(persistent_bytes(model, wrapper)["optimizer_state"], expected)


class MemProfilerCPUTest(unittest.TestCase):
    def test_writes_json_after_n_steps(self):
        import json
        import tempfile

        model = Llama(config())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                mem_profile=f"{tmp}/mem.json", mem_profile_steps=2, mem_snapshot=None,
                device="cpu", opt="adamw", batch_size=2, acc_steps=1, sequence_length=16,
                n_layer=3, n_embd=32, activation_checkpointing="none",
            )
            prof = MemProfiler(cfg, model, opt)
            done = []
            for step in range(2):
                prof.step_begin(step)
                prof.before_forward(0)
                loss = model(torch.randint(0, 97, (2, 16)), targets=torch.randint(0, 97, (2, 16)))["loss"]
                prof.after_forward(0)
                loss.backward()
                prof.before_opt_step()
                opt.step()
                prof.after_opt_step()
                opt.zero_grad(set_to_none=True)
                done.append(prof.step_end())
            self.assertEqual(done, [False, True])
            data = json.loads(Path(cfg.mem_profile).read_text())
        self.assertEqual(len(data["steps"]), 2)
        self.assertGreater(data["summary"]["persistent_gb"]["optimizer_state"], 0)


if __name__ == "__main__":
    unittest.main()
