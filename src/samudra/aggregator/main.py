# SPDX-FileCopyrightText: 2024 Allen Institute for Artificial Intelligence
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""
The code in this directory is directly inspired by the ACE project by AI2.

See the original repository at: https://github.com/ai2cm/ace/tree/39133c18524cda85965486ecdc8cb64aac06f4c3/fme/fme/ace/aggregator
"""

import torch

from samudra.aggregator.inference import InferenceEvaluatorAggregator
from samudra.aggregator.train import TrainAggregator
from samudra.aggregator.validate import MultiScaleValidateAggregator, ValidateAggregator
from samudra.aggregator.validate.map import MapAggregator
from samudra.aggregator.validate.reduced import MeanAggregator
from samudra.aggregator.validate.snapshot import SnapshotAggregator
from samudra.aggregator.validate.spatial import NormalizedSpatialDiagnosticsAggregator
from samudra.aggregator.validate.sub_aggregator import ValidateSubAggregator
from samudra.constants import BoundaryVarNames, PrognosticVarNames, TensorMap
from samudra.utils.data import CanonicalDataset, Normalize


class Aggregator:
    @staticmethod
    def get_train_aggregator(tensor_map: TensorMap) -> TrainAggregator:
        return TrainAggregator(tensor_map)

    @staticmethod
    def get_validation_aggregator(
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        *,
        include_image_aggregators: bool = True,
        patch_size: tuple[int, int] | None = None,
    ) -> ValidateAggregator:
        val_aggregators: dict[str, ValidateSubAggregator] = {
            "reduced": MeanAggregator(area_weights, hist),
        }
        if include_image_aggregators:
            val_aggregators.update(
                {
                    "snapshot": SnapshotAggregator(metadata, hist),
                    "mean_map": MapAggregator(metadata, hist),
                }
            )
            if patch_size is not None:
                val_aggregators["spatial"] = NormalizedSpatialDiagnosticsAggregator(
                    tensor_map, hist, patch_size
                )

        return ValidateAggregator(
            val_aggregators,
            hist=hist,
            num_prognostic_channels=num_prognostic_channels,
            tensor_map=tensor_map,
            normalize=normalize,
            record_baselines=True,
        )

    @staticmethod
    def get_multiscale_validation_aggregator(
        sources: list[CanonicalDataset],
        hist: int,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        *,
        include_image_aggregators: bool = True,
        patch_extent: tuple[float, float] | None = None,
    ) -> MultiScaleValidateAggregator:
        """Build independent validation diagnostics for each output grid."""
        aggregators: dict[tuple[int, int], tuple[str, ValidateAggregator]] = {}
        for source in sources:
            grid = source.grid_size
            if grid in aggregators:
                raise ValueError(f"Duplicate validation grid {grid}.")
            scale_label = f"{grid[0]}x{grid[1]}"
            normalize = Normalize(
                source,
                prognostic_var_names=prognostic_var_names,
                boundary_var_names=boundary_var_names,
            )
            patch_size = None
            if patch_extent is not None:
                patch_size = (
                    round(patch_extent[0] * grid[0] / 180.0),
                    round(patch_extent[1] * grid[1] / 360.0),
                )
            aggregators[grid] = (
                scale_label,
                Aggregator.get_validation_aggregator(
                    source.metadata,
                    hist,
                    source.spherical_area_weights,
                    num_prognostic_channels,
                    tensor_map,
                    normalize,
                    include_image_aggregators=include_image_aggregators,
                    patch_size=patch_size,
                ),
            )
        return MultiScaleValidateAggregator(aggregators)

    @staticmethod
    def get_inline_inference_aggregator(
        n_timesteps: int,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        channel_mean_names: list[str] | None = None,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
            normalize=normalize,
            tensor_map=tensor_map,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=False,
            log_global_mean_norm_time_series=False,
            channel_mean_names=channel_mean_names,
        )

    @staticmethod
    def get_standalone_inference_aggregator(
        n_timesteps: int,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        channel_mean_names: list[str] | None = None,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
            normalize=normalize,
            tensor_map=tensor_map,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=True,
            log_global_mean_norm_time_series=True,
            channel_mean_names=channel_mean_names,
        )
