# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Self

import torch

from samudra.constants import Lat, Lon, PrognosticMask


@dataclasses.dataclass(frozen=True)
class GridContext:
    """Grid-level context for model forward passes and loss computation.

    Bundles spatial metadata that travels alongside input tensors during training:
    the ocean/land mask for excluding land cells from gradients, and the grid
    resolution for latitude-weighted loss calculations.

    Attributes:
        label_mask: Boolean mask indicating valid ocean cells for each prognostic
            variable. Shape: (num_prognostic_vars, lat, lon). Land cells are False.
        input_resolution_cpu: Tuple of (latitude, longitude) coordinate tensors defining
            the input spatial grid. Used by encoders for positional encoding.
        output_resolution_cpu: Tuple of (latitude, longitude) coordinate tensors defining
            the output spatial grid. Used by decoders to determine output shape and
            pixel-query positions. Defaults to input_resolution_cpu for single-scale use.
        input_mask: Optional channel-wise validity mask on the input grid. Decoders that
            render prognostic channels before changing resolution use it to exclude land
            values independently for each channel during interpolation.
    """

    label_mask: PrognosticMask
    input_resolution_cpu: tuple[Lat, Lon]
    output_resolution_cpu: tuple[Lat, Lon]
    input_mask: PrognosticMask | None = None

    def to(self, device: torch.device) -> Self:
        """Move the label mask to the specified device."""
        return dataclasses.replace(
            self,
            label_mask=self.label_mask.to(device, non_blocking=True),
            input_mask=(
                None
                if self.input_mask is None
                else self.input_mask.to(device, non_blocking=True)
            ),
        )
