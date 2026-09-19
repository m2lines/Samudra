# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch import Tensor

from samudra.aggregator.metrics import area_weighted_sum
from samudra.constants import (
    Boundary,
    DataLayout,
    HistBatched,
    HistChanneled,
    Input,
    Prognostic,
)
from samudra.derived_variables import compute_global_ocean_heat_content
from samudra.utils.data import BatchPreprocessor
from samudra.utils.device import get_device


class BaseCorrector(torch.nn.Module):
    """Base class for tensor correction modules."""

    def __init__(
        self, input_steps: int, data_layout: DataLayout, normalize: BatchPreprocessor
    ):
        super().__init__()
        self.input_steps = input_steps
        self.data_layout = data_layout
        self.normalize = normalize
        self.num_prognostic_channels = len(self.data_layout.prognostic_var_names)

    def _flatten_hist(self, fts: HistChanneled) -> HistBatched:
        return rearrange(
            fts, "n (hist c) h w -> (n hist) c h w", hist=self.input_steps
        )

    def _flatten_input(self, fts: Input) -> tuple[HistBatched, HistBatched]:
        fts_input = fts[:, : self.input_steps * self.num_prognostic_channels]
        fts_input = self._flatten_hist(fts_input)

        fts_boundary = fts[:, self.input_steps * self.num_prognostic_channels :]
        fts_boundary = self._flatten_hist(fts_boundary)
        return fts_input, fts_boundary

    def _unflatten_hist(self, fts: HistBatched) -> HistChanneled:
        return rearrange(
            fts, "(n hist) c h w -> n (hist c) h w", hist=self.input_steps
        )

    def _unnormalize_fts_prognostic(self, fts: Prognostic) -> Prognostic:
        # Corrector is run in float64 to avoid precision loss
        fts = fts.to(torch.float64)
        return self.normalize.unnormalize_tensor_prognostic(fts, fill_value=0.0)

    def _normalize_fts_prognostic(self, fts: Prognostic) -> Prognostic:
        fts = self.normalize.normalize_tensor_prognostic(fts)
        return fts.to(torch.float32)

    def _unnormalize_fts_input(
        self, fts: Prognostic, fts_boundary: Boundary
    ) -> tuple[Prognostic, Boundary]:
        # Corrector is run in float64 to avoid precision loss
        fts = self._unnormalize_fts_prognostic(fts)
        fts_boundary = fts_boundary.to(torch.float64)
        fts_boundary = self.normalize.unnormalize_tensor_boundary(
            fts_boundary, fill_value=0.0
        )

        return fts, fts_boundary

    def forward(self, fts_input: Input, fts: Prognostic) -> Prognostic:
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
        non_negative_corrector_names: list[str] | None,
        input_steps: int,
        data_layout: DataLayout,
        normalize: BatchPreprocessor,
    ):
        super().__init__(input_steps, data_layout, normalize)
        self.non_negative_corrector_names = non_negative_corrector_names
        if self.non_negative_corrector_names is not None:
            self.non_neg_indices = torch.cat(
                [
                    self.data_layout.variable_indices[name]
                    for name in self.non_negative_corrector_names
                ],
                dim=0,
            )
        else:
            self.non_neg_indices = torch.tensor(np.nan)

        self.non_neg_indices = self.non_neg_indices.to(get_device())

    def _apply_relu_correction(self, fts: Prognostic) -> Prognostic:
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

    def forward(self, fts_input: Input, fts: Prognostic) -> Prognostic:
        """Applies correction to the input features if needed.

        Args:
            fts_input: Input tensor of shape (batch_size, hist*channels, height, width)
            fts: Output tensor of shape (batch_size, hist*channels, height, width)

        Returns:
            Corrected output tensor of the same shape
        """
        if not torch.isnan(self.non_neg_indices).all():
            fts = self._flatten_hist(fts)
            fts = self._apply_relu_correction(fts)
            fts = self._unflatten_hist(fts)
        return fts


def compute_expected_heat_content_change(
    surface_heat_flux: Tensor,
    geothermal_heat_flux: Tensor,
    sea_surface_fraction_tensor: Tensor,
    area_weighted_func: Callable,
    seconds_per_time_step: int,
) -> Tensor:
    # Expected change in heat content from surface flux
    dHC_expected = (
        area_weighted_func(surface_heat_flux * sea_surface_fraction_tensor)
        * seconds_per_time_step
    )  # [J]

    # Apply geothermal heat flux
    dHC_expected += geothermal_heat_flux

    return dHC_expected


class OceanHeatCorrector(BaseCorrector):
    """
    Applies a correction to potential temperature to conserve
    ocean heat content.

    Following this document - https://www.overleaf.com/project/67ed705406995df4c185e6b6

    This class relies on input boundary conditions, namely hfds and hfgeou.
    """

    def __init__(
        self,
        input_steps: int,
        area_weights: torch.Tensor,
        data_layout: DataLayout,
        normalize: BatchPreprocessor,
        hfgeou_tensor: torch.Tensor,
        sea_surface_fraction_tensor: torch.Tensor,
        seconds_per_time_step: int,
    ):
        super().__init__(input_steps, data_layout, normalize)
        self.seconds_per_time_step = seconds_per_time_step
        # Area weights are not on the correct scale.
        self.area_weights = area_weights
        self.area_weighted_func = partial(
            area_weighted_sum, area_weights=self.area_weights
        )
        self.dz = self.data_layout.dz

        self.thetao_idx = self.data_layout.variable_indices[
            data_layout.ocean_heat_temperature_var
        ]
        self.hfds_idx = torch.tensor(
            [self.data_layout.boundary_var_names.index(data_layout.surface_heat_flux_var)]
        )

        self.thetao_idx = self.thetao_idx.to(get_device())
        self.hfds_idx = self.hfds_idx.to(get_device())
        self.dz = self.dz.to(get_device())
        self.hfgeou_tensor = hfgeou_tensor.to(get_device())
        self.sea_surface_fraction_tensor = sea_surface_fraction_tensor.to(get_device())

        self.dHC_geothermal = (
            self.area_weighted_func(
                self.hfgeou_tensor * self.sea_surface_fraction_tensor
            )
            * seconds_per_time_step
        )

        # Squared fractional imbalance of the raw (pre-correction) prediction,
        # set each forward call. Exposed so training can penalize it directly
        # instead of relying solely on supervising the pre-correction output --
        # see https://arxiv.org/abs/2607.18416 (a soft penalty alone, without
        # also moving the supervision point, was not sufficient in their
        # ablation).
        self.last_imbalance_sq: Tensor | None = None

    def forward(self, fts_input: Input, fts: Prognostic) -> Prognostic:
        fts_input = fts_input.detach()

        fts = self._flatten_hist(fts)
        fts = self._unnormalize_fts_prognostic(fts)

        fts_input, fts_boundary = self._flatten_input(fts_input)
        fts_input, fts_boundary = self._unnormalize_fts_input(fts_input, fts_boundary)

        # The input and output mapping of the variables are the same
        T_input = fts_input[:, self.thetao_idx]  # (batch, depth, lat, lon)
        T_pred = fts[:, self.thetao_idx]

        # Extract the boundary variables
        surface_heat_flux = fts_boundary[:, self.hfds_idx].squeeze(1)

        global_HC_t0 = compute_global_ocean_heat_content(
            T_input, self.dz, self.area_weighted_func
        )
        global_HC_t1 = compute_global_ocean_heat_content(
            T_pred, self.dz, self.area_weighted_func
        )
        dHC_expected = compute_expected_heat_content_change(
            surface_heat_flux,
            self.dHC_geothermal,
            self.sea_surface_fraction_tensor,
            self.area_weighted_func,
            self.seconds_per_time_step,
        )

        HC_correct_ratio = (global_HC_t0 + dHC_expected) / (global_HC_t1 + 1e-8)
        self.last_imbalance_sq = (HC_correct_ratio - 1.0).pow(2)

        T_corrected = T_pred * HC_correct_ratio.view(-1, 1, 1, 1)

        fts[:, self.thetao_idx] = T_corrected

        fts = self._normalize_fts_prognostic(fts)
        fts = self._unflatten_hist(fts)

        return fts


class Correctors(torch.nn.Module):
    """Applies a sequence of corrections to input tensors based on configuration."""

    def __init__(
        self,
        non_negative_corrector_names: list[str] | None,
        ocean_heat_corrector: bool,
        input_steps: int,
        area_weights: torch.Tensor,
        static_data: xr.Dataset | None,
        data_layout: DataLayout,
        normalize: BatchPreprocessor,
        imbalance_penalty_weight: float = 0.0,
    ):
        """
        Correctors class that applies a sequence of corrections to input tensors based
        on configuration.

        Args:
            non_negative_corrector_names (list[str]): list of names of non-negative correctors (None turns feature off).
            ocean_heat_corrector (bool): whether to apply ocean heat corrections (turns this feature on or off)
            input_steps: Number of raw timesteps stacked in each model call.
            area_weights: Area weights for area weighting
            static_data: Static data for corrections
            data_layout: Canonical channel layout, physical metadata, and tensor index mappings.
            normalize: Normalization/unnormalization helper shared with the train/eval pipeline.
            imbalance_penalty_weight: Weight on the soft penalty applied to any
                sub-corrector's own raw-budget imbalance (e.g. `(HC_correct_ratio
                - 1) ** 2` for the ocean heat corrector). Added to the training
                loss by `BaseModel.forward` alongside supervising the
                pre-correction prediction. 0.0 disables the penalty.
        """
        super().__init__()
        self.data_layout = data_layout
        self.normalize = normalize
        self.imbalance_penalty_weight = imbalance_penalty_weight
        self.last_imbalance_penalty: Tensor | None = None

        correctors: list[BaseCorrector] = []

        # Initialize ReLU corrector if configured
        if non_negative_corrector_names is not None:
            correctors.append(
                ReLUCorrector(
                    non_negative_corrector_names=non_negative_corrector_names,
                    input_steps=input_steps,
                    data_layout=self.data_layout,
                    normalize=self.normalize,
                )
            )

        if ocean_heat_corrector:
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
                    input_steps=input_steps,
                    area_weights=area_weights,
                    data_layout=self.data_layout,
                    normalize=self.normalize,
                    hfgeou_tensor=hfgeou_tensor,
                    sea_surface_fraction_tensor=sea_surface_fraction_tensor,
                    seconds_per_time_step=self.data_layout.seconds_per_time_step,
                )
            )

        self.correctors = torch.nn.ModuleList(correctors)

    def forward(self, fts_input: Input, fts: Prognostic) -> Prognostic:
        """Applies all corrections sequentially to the input features.

        Args:
            fts_input: Input tensor
            fts: Output tensor to correct

        Returns:
            Corrected output tensor after applying all corrections
        """
        self.last_imbalance_penalty = None
        for corrector in self.correctors:
            fts = corrector(fts_input, fts)
            imbalance_sq = getattr(corrector, "last_imbalance_sq", None)
            if imbalance_sq is not None:
                penalty = imbalance_sq.mean()
                self.last_imbalance_penalty = (
                    penalty
                    if self.last_imbalance_penalty is None
                    else self.last_imbalance_penalty + penalty
                )
        return fts
