from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from optim.distributed_state_comm import DistributedStateCommunicator
from optim.fp8_state import (
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)


class QuantizedStateWireTest(TestCase):
    def _run_case(self, expansion: str) -> None:
        qargs = SimpleNamespace(
            qgroup_size=4,
            first_order_bit="E4M3",
            first_order_expansion=expansion,
            expand_min=16,
        )
        local_momentum = torch.tensor(
            [
                [0.25, -0.5, 0.75],
                [1.0, -1.25, 1.5],
                [1.75, -2.0, 2.25],
            ],
            dtype=torch.float32,
        )
        state = {}
        init_fp8_state(state, "momentum", local_momentum, qargs, order="first")
        quantize_fp8_state_(
            state,
            "momentum",
            local_momentum,
            qargs,
            signed=True,
        )
        stored_values = state["momentum"]
        full_gradient = torch.linspace(-1.0, 1.0, 15).reshape(5, 3)

        def fake_all_gather(output, local, group=None):
            del group
            output.view(2, -1).copy_(local.view(1, -1).expand(2, -1))

        communicator = DistributedStateCommunicator(
            enabled=True,
            wire_dtype="fp8",
            qargs=qargs,
            profile=True,
        )
        with (
            patch("optim.distributed_state_comm.dist.is_available", return_value=True),
            patch("optim.distributed_state_comm.dist.is_initialized", return_value=True),
            patch("optim.distributed_state_comm.dist.get_world_size", return_value=2),
            patch(
                "optim.distributed_state_comm.dist.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            communicator.start_step()
            actual = communicator.gather_quantized_state_rows(
                state,
                "momentum",
                original_rows=5,
                gradient=full_gradient,
                gradient_alpha=0.95,
            )
            communicator.finish_step()

        decoded_local = dequantize_fp8_state(
            state,
            "momentum",
            qargs,
            signed=True,
        )
        expected = torch.cat((decoded_local, decoded_local), dim=0)[:5]
        expected = expected + 0.95 * full_gradient
        torch.testing.assert_close(actual, expected)
        self.assertIs(state["momentum"], stored_values)

        profile = communicator.get_last_profile()
        self.assertEqual(profile["optimizer_state_collectives"], 1.0)
        self.assertGreater(profile["optimizer_state_wire_bytes"], 0.0)
        self.assertGreater(profile["optimizer_state_encode_ms"], 0.0)
        self.assertGreater(profile["optimizer_state_decode_ms"], 0.0)

    def test_reuses_plain_fp8_momentum(self) -> None:
        self._run_case("false")

    def test_reuses_expanded_fp8_momentum(self) -> None:
        self._run_case("expand")
