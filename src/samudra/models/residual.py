# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Residual prediction: the network predicts increments on the input state."""

from typing import Literal

import torch
from jaxtyping import Float

from samudra.constants import Prognostic

ResidualAnchor = Literal["same_slot", "chained"]


class ResidualHead(torch.nn.Module):
    """Turns a network's output into next states, as increments on its input.

    Each output channel is multiplied by ``scale`` and added to an anchor
    state. With ``same_slot`` anchoring, output step k is added to input step
    k, so with one output step it is added to the newest input. With
    ``chained`` anchoring, each output step is an increment on the previous
    output, starting from the newest input: ``z1 = x[t] + d1``,
    ``z2 = z1 + d2``, with gradients flowing through ``z1``.

    ``scale`` has one value per output channel. It comes from the data, like
    the normalization statistics, so it is not saved in checkpoints.
    """

    def __init__(
        self,
        *,
        input_steps: int,
        scale: Float[torch.Tensor, " out_channel"],
        anchor: ResidualAnchor = "same_slot",
    ) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.anchor = anchor
        self.scale: torch.Tensor
        self.register_buffer("scale", scale.float(), persistent=False)

    def forward(self, prognostic: Prognostic, decoding: Prognostic) -> Prognostic:
        increments = decoding * self.scale.view(1, -1, 1, 1)
        if self.anchor == "chained":
            n_var = prognostic.shape[1] // self.input_steps
            batch, _, *grid = increments.shape
            steps = increments.view(batch, -1, n_var, *grid).cumsum(dim=1)
            return (prognostic[:, -n_var:].unsqueeze(1) + steps).flatten(1, 2)
        return prognostic[:, -decoding.shape[1] :] + increments
