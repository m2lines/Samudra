from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch import Tensor

from ocean_emulators.aggregator.metrics import area_weighted_sum
from ocean_emulators.constants import CP_SW, RHO_0, SECONDS_PER_5DAY, TensorMap
from ocean_emulators.utils.data import Normalize
from ocean_emulators.utils.device import get_device


class BaseCorrector(torch.nn.Module):
    """Base class for tensor correction modules."""

    def __init__(self, hist: int, tensor_map: TensorMap, normalize: Normalize):
        super().__init__()
        self.hist = hist
        self.tensor_map = tensor_map
        self.normalize = normalize
        self.num_prognostic_channels = len(self.tensor_map.prognostic_var_names)

    def _rearrange_fts(self, fts: Tensor) -> Tensor:
        return rearrange(fts, "n (hist c) h w -> (n hist) c h w", hist=self.hist + 1)

    def _rearrange_fts_with_boundary(self, fts: Tensor) -> Tuple[Tensor, Tensor]:
        fts_input = fts[:, : (self.hist + 1) * self.num_prognostic_channels]
        fts_input = rearrange(
            fts_input, "n (hist c) h w -> (n hist) c h w", hist=self.hist + 1
        )

        fts_boundary = fts[:, (self.hist + 1) * self.num_prognostic_channels :]
        fts_boundary = rearrange(
            fts_boundary, "n (hist c) h w -> (n hist) c h w", hist=self.hist + 1
        )
        return fts_input, fts_boundary

    def _inv_rearrange_fts(self, fts: Tensor) -> Tensor:
        return rearrange(fts, "(n hist) c h w -> n (hist c) h w", hist=self.hist + 1)

    def _unnormalize_fts_prognostic(self, fts: Tensor) -> Tensor:
        # Corrector is run in float64 to avoid precision loss
        fts = fts.to(torch.float64)
        return self.normalize.unnormalize_tensor_prognostic(fts, fill_value=0.0)

    def _normalize_fts_prognostic(self, fts: Tensor) -> Tensor:
        fts = self.normalize.normalize_tensor_prognostic(fts)
        return fts.to(torch.float32)

    def _unnormalize_fts_prognostic_with_boundary(
        self, fts: Tensor, fts_boundary: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Corrector is run in float64 to avoid precision loss
        fts = self._unnormalize_fts_prognostic(fts)
        fts_boundary = fts_boundary.to(torch.float64)
        fts_boundary = self.normalize.unnormalize_tensor_boundary(
            fts_boundary, fill_value=0.0
        )

        return fts, fts_boundary

    def forward(self, fts_input: Tensor, fts: Tensor) -> Tensor:
        """Apply correction to the input features.

        Args:
            fts_input: Input tensor to correct
            fts: Output tensor to correct

        Returns:
            Corrected output tensor
        """
        raise NotImplementedError


class ReLUCorrector(BaseCorrector):
    """
    Applies ReLU correction to specified tensor channels.
    """

    def __init__(
        self,
        non_negative_corrector_names: Optional[List[str]],
        hist: int,
        tensor_map: TensorMap,
        normalize: Normalize,
    ):
        super().__init__(hist, tensor_map, normalize)
        self.non_negative_corrector_names = non_negative_corrector_names
        if self.non_negative_corrector_names is not None:
            self.non_neg_indices = torch.cat(
                [
                    self.tensor_map.VAR_3D_IDX[name]
                    for name in self.non_negative_corrector_names
                ],
                dim=0,
            )
        else:
            self.non_neg_indices = torch.tensor(np.nan)

        self.non_neg_indices = self.non_neg_indices.to(get_device())

    def _apply_relu_correction(self, fts: Tensor) -> Tensor:
        """Applies ReLU to specified channels.

        Args:
            fts: tensor of shape (batch_size, channels, height, width)

        Returns:
            Corrected tensor of the same shape
        """
        unnormalized = self._unnormalize_fts_prognostic(fts)
        unnormalized[:, self.non_neg_indices, :, :] = torch.relu(
            unnormalized[:, self.non_neg_indices, :, :]
        )
        return self._normalize_fts_prognostic(unnormalized)

    def forward(self, fts_input: Tensor, fts: Tensor) -> Tensor:
        """Applies correction to the input features if needed.

        Args:
            fts_input: Input tensor of shape (batch_size, hist*channels, height, width)
            fts: Output tensor of shape (batch_size, hist*channels, height, width)

        Returns:
            Corrected output tensor of the same shape
        """
        if not torch.isnan(self.non_neg_indices).all():
            fts = self._rearrange_fts(fts)
            fts = self._apply_relu_correction(fts)
            fts = self._inv_rearrange_fts(fts)
        return fts


def compute_ocean_heat_content(
    T: Tensor, dz: Tensor, area_weighted_func: Callable
) -> Tuple[Tensor, Tensor]:
    """Compute the global heat content of the ocean.

    Args:
        T: Temperature tensor of shape (batch_size, depth, height, width)
        dz: Depth tensor of shape (depth,)
        area_weighted_func: Area weighted function

    Returns:
        Global heat content tensor of shape (batch_size,)
    """
    # Compute heat content per layer
    HC_t = RHO_0 * CP_SW * T * dz.view(1, -1, 1, 1)

    # Column integrated heat content
    total_HC_t = torch.sum(HC_t, dim=1)

    # Sum over depth to get total heat content
    global_HC_t = area_weighted_func(total_HC_t)  # (batch,) [J]

    return global_HC_t, HC_t


class OceanHeatCorrector(BaseCorrector):
    """
    Applies a correction to potential temperature to conserve
    ocean heat content.

    Following this document - https://www.overleaf.com/project/67ed705406995df4c185e6b6

    This class relies in input bounjdary conditions. Thus, we would
    need to supply the boundary conditions for each corresponding
    input step.
    """

    def __init__(
        self,
        hist: int,
        area_weights: torch.Tensor,
        tensor_map: TensorMap,
        normalize: Normalize,
        hfgeou_tensor: torch.Tensor,
        sea_surface_fraction_tensor: torch.Tensor,
    ):
        super().__init__(hist, tensor_map, normalize)
        # Area weights are not on the correct scale.
        self.area_weights = area_weights
        self.area_weighted_func = partial(
            area_weighted_sum, area_weights=self.area_weights
        )
        self.dz = self.tensor_map.dz

        self.thetao_idx = self.tensor_map.VAR_3D_IDX["thetao"]
        self.hfds_idx = self.tensor_map.INPT_BOUNDARY_IDX["hfds"]

        self.thetao_idx = self.thetao_idx.to(get_device())
        self.hfds_idx = self.hfds_idx.to(get_device())
        self.dz = self.dz.to(get_device())
        self.hfgeou_tensor = hfgeou_tensor.to(get_device())
        self.sea_surface_fraction_tensor = sea_surface_fraction_tensor.to(get_device())

        self.dHC_geothermal = (
            self.area_weighted_func(self.hfgeou_tensor) * SECONDS_PER_5DAY
        )

    def forward(self, fts_input_boundary: Tensor, fts: Tensor) -> Tensor:
        fts_input_boundary = fts_input_boundary.detach()

        fts = self._rearrange_fts(fts)
        fts = self._unnormalize_fts_prognostic(fts)

        fts_input, fts_boundary = self._rearrange_fts_with_boundary(fts_input_boundary)
        fts_input, fts_boundary = self._unnormalize_fts_prognostic_with_boundary(
            fts_input, fts_boundary
        )

        # The input and output mapping of the variables are the same
        T_input = fts_input[:, self.thetao_idx]  # (batch, depth, lat, lon)
        T_pred = fts[:, self.thetao_idx]

        # Extract the boundary variables
        surface_heat_flux = fts_boundary[:, self.hfds_idx].squeeze(1)

        global_HC_t0, HC_t0 = compute_ocean_heat_content(
            T_input, self.dz, self.area_weighted_func
        )
        global_HC_t1, HC_t1 = compute_ocean_heat_content(
            T_pred, self.dz, self.area_weighted_func
        )

        # Expected change in heat content from surface flux
        dHC_expected = (
            self.area_weighted_func(
                surface_heat_flux * self.sea_surface_fraction_tensor
            )
            * SECONDS_PER_5DAY
        )  # [J]

        # Apply geothermal heat flux
        dHC_expected += self.dHC_geothermal

        HC_correct_ratio = (global_HC_t0 + dHC_expected) / global_HC_t1

        # This seems redundant, so skipping it
        # HC_t1_corrected = HC_t1 * HC_correct_ratio.view(-1, 1, 1, 1)
        # T_corrected = HC_t1_corrected / (RHO_0 * CP_SW * self.dz.view(1, -1, 1, 1))

        T_corrected = T_pred * HC_correct_ratio.view(-1, 1, 1, 1)

        fts[:, self.thetao_idx] = T_corrected

        fts = self._normalize_fts_prognostic(fts)
        fts = self._inv_rearrange_fts(fts)

        return fts


class Corrector(torch.nn.Module):
    """Applies a sequence of corrections to input tensors based on configuration."""

    def __init__(
        self,
        config,
        hist: int,
        area_weights: torch.Tensor,
        static_data: xr.Dataset | None,
    ):
        """
        Corrector class that applies a sequence of corrections to input tensors based
        on configuration.

        Args:
            config: Configuration object containing corrector settings
            hist: History length for temporal data
            area_weights: Area weights for area weighting
            static_data: Static data for corrections
        """
        super().__init__()
        self.tensor_map: TensorMap = TensorMap.get_instance()
        self.normalize = Normalize.get_instance()

        correctors: List[BaseCorrector] = []

        # Initialize ReLU corrector if configured
        if (
            hasattr(config, "non_negative_corrector_names")
            and config.non_negative_corrector_names
        ):
            correctors.append(
                ReLUCorrector(
                    non_negative_corrector_names=config.non_negative_corrector_names,
                    hist=hist,
                    tensor_map=self.tensor_map,
                    normalize=self.normalize,
                )
            )

        if hasattr(config, "ocean_heat_corrector") and config.ocean_heat_corrector:
            assert static_data is not None, (
                "Static data is required for ocean heat corrector"
            )
            assert "hfgeou" in static_data.data_vars, (
                "hfgeou is required for ocean heat corrector"
            )
            assert "sea_surface_fraction" in static_data.data_vars, (
                "sea_surface_fraction is required for ocean heat corrector"
            )
            hfgeou = static_data["hfgeou"]
            sea_surface_fraction = static_data["sea_surface_fraction"]
            hfgeou_tensor = torch.from_numpy(hfgeou.to_numpy())
            sea_surface_fraction_tensor = torch.from_numpy(
                sea_surface_fraction.to_numpy()
            )
            correctors.append(
                OceanHeatCorrector(
                    hist=hist,
                    area_weights=area_weights,
                    tensor_map=self.tensor_map,
                    normalize=self.normalize,
                    hfgeou_tensor=hfgeou_tensor,
                    sea_surface_fraction_tensor=sea_surface_fraction_tensor,
                )
            )

        self.correctors = torch.nn.ModuleList(correctors)

    def forward(self, fts_input: Tensor, fts: Tensor) -> Tensor:
        """Applies all corrections sequentially to the input features.

        Args:
            fts_input: Input tensor
            fts: Output tensor to correct

        Returns:
            Corrected output tensor after applying all corrections
        """
        for corrector in self.correctors:
            fts = corrector(fts_input, fts)
        return fts
