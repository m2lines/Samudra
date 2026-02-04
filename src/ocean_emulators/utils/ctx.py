import dataclasses
from typing import Self

import torch

from ocean_emulators.constants import Lat, Lon, PrognosticMask


@dataclasses.dataclass(frozen=True)
class GridContext:
    """Grid-level context for model forward passes and loss computation.

    Bundles spatial metadata that travels alongside input tensors during training:
    the ocean/land mask for excluding land cells from gradients, and the grid
    resolution for latitude-weighted loss calculations.

    Attributes:
        label_mask: Boolean mask indicating valid ocean cells for each prognostic
            variable. Shape: (num_prognostic_vars, lat, lon). Land cells are False.
        input_resolution: Tuple of (latitude, longitude) coordinate tensors defining
            the spatial grid. Used for cosine-latitude weighting in loss functions.
    """

    label_mask: PrognosticMask
    input_resolution: tuple[Lat, Lon]

    def to(self, device: torch.device) -> Self:
        """Move the label mask to the specified device."""
        return dataclasses.replace(
            self, label_mask=self.label_mask.to(device, non_blocking=True)
        )
