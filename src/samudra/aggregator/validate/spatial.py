# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Single-step spectral and patch-boundary validation diagnostics."""

import math

import matplotlib.pyplot as plt
import torch

from samudra.aggregator.validate.sub_aggregator import ValidateSubAggregator
from samudra.constants import TensorMap
from samudra.utils.distributed import all_reduce_mean, is_main_process
from samudra.utils.wandb import Metrics, MetricsDict, WandBLogger


def zonal_power_spectrum(data: torch.Tensor) -> torch.Tensor:
    """Return mean zonal power with shape ``[channel, wavenumber]``.

    Input has shape ``[batch, channel, latitude, longitude]``. Land NaNs are
    replaced with zero, and each latitude row is centered before the FFT so the
    diagnostic focuses on spatial variability rather than the zonal mean.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(data.shape)}")
    data = torch.nan_to_num(data)
    data = data - data.mean(dim=-1, keepdim=True)
    coefficients = torch.fft.rfft(data, dim=-1, norm="ortho")
    return coefficients.abs().square().mean(dim=(0, 2))


def high_wavenumber_power_ratio(
    generated_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    *,
    cutoff_fraction: float = 0.5,
) -> torch.Tensor:
    """Return generated/target power in the upper resolved wavenumber band."""
    if generated_spectrum.shape != target_spectrum.shape:
        raise ValueError("Generated and target spectra must have identical shapes.")
    if generated_spectrum.ndim != 2:
        raise ValueError("Spectra must have shape [channel, wavenumber].")
    if not 0.0 < cutoff_fraction < 1.0:
        raise ValueError("cutoff_fraction must be between zero and one.")
    n_nonzero_modes = generated_spectrum.shape[-1] - 1
    cutoff = 1 + math.floor(n_nonzero_modes * cutoff_fraction)
    generated_power = generated_spectrum[:, cutoff:].sum(dim=-1)
    target_power = target_spectrum[:, cutoff:].sum(dim=-1)
    return generated_power / target_power.clamp_min(torch.finfo(target_power.dtype).eps)


def patch_seam_jump_ratio(
    error: torch.Tensor,
    patch_size: tuple[int, int],
) -> torch.Tensor:
    """Compare error jumps at patch boundaries with within-patch jumps.

    A value near one means patch boundaries are no less smooth than ordinary
    neighboring pixels. Values above one identify boundary-aligned artifacts.
    The returned tensor has one value per channel.
    """
    if error.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(error.shape)}")
    patch_h, patch_w = patch_size
    if patch_h < 1 or patch_w < 1:
        raise ValueError("Patch dimensions must be positive.")
    _, channels, height, width = error.shape
    if patch_h >= height or patch_w >= width:
        raise ValueError(
            f"Patch size {patch_size} must be smaller than grid {(height, width)}."
        )

    error = torch.nan_to_num(error)
    vertical_jumps = (error[:, :, 1:, :] - error[:, :, :-1, :]).abs()
    horizontal_jumps = (error[:, :, :, 1:] - error[:, :, :, :-1]).abs()
    vertical_boundary = (
        torch.arange(1, height, device=error.device).remainder(patch_h) == 0
    )
    horizontal_boundary = (
        torch.arange(1, width, device=error.device).remainder(patch_w) == 0
    )

    seam_values = torch.cat(
        (
            vertical_jumps[:, :, vertical_boundary, :]
            .permute(0, 2, 3, 1)
            .reshape(-1, channels),
            horizontal_jumps[:, :, :, horizontal_boundary]
            .permute(0, 2, 3, 1)
            .reshape(-1, channels),
        ),
        dim=0,
    )
    interior_values = torch.cat(
        (
            vertical_jumps[:, :, ~vertical_boundary, :]
            .permute(0, 2, 3, 1)
            .reshape(-1, channels),
            horizontal_jumps[:, :, :, ~horizontal_boundary]
            .permute(0, 2, 3, 1)
            .reshape(-1, channels),
        ),
        dim=0,
    )
    seam_mean = seam_values.mean(dim=0)
    interior_mean = interior_values.mean(dim=0)
    return seam_mean / interior_mean.clamp_min(torch.finfo(error.dtype).eps)


class NormalizedSpatialDiagnosticsAggregator(ValidateSubAggregator):
    """Aggregate normalized spectra and patch-seam metrics for one grid."""

    def __init__(
        self,
        tensor_map: TensorMap,
        hist: int,
        patch_size: tuple[int, int],
    ) -> None:
        self.tensor_map = tensor_map
        self.hist = hist
        self.patch_size = patch_size
        self._n_samples = 0
        self._target_spectrum: torch.Tensor | None = None
        self._generated_spectrum: torch.Tensor | None = None
        self._seam_ratio: torch.Tensor | None = None

    @torch.no_grad()
    def record_batch(
        self,
        *,
        loss,
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ) -> None:
        del loss, target_data, gen_data, input_data, input_data_norm
        names = self.tensor_map.prognostic_var_names
        target = torch.stack(
            [target_data_norm[name][:, self.hist] for name in names], dim=1
        )
        generated = torch.stack(
            [gen_data_norm[name][:, self.hist] for name in names], dim=1
        )
        batch_size = target.shape[0]
        target_spectrum = zonal_power_spectrum(target) * batch_size
        generated_spectrum = zonal_power_spectrum(generated) * batch_size
        seam_ratio = (
            patch_seam_jump_ratio(generated - target, self.patch_size) * batch_size
        )

        if self._target_spectrum is None:
            self._target_spectrum = target_spectrum
            self._generated_spectrum = generated_spectrum
            self._seam_ratio = seam_ratio
        else:
            self._target_spectrum += target_spectrum
            assert self._generated_spectrum is not None
            assert self._seam_ratio is not None
            self._generated_spectrum += generated_spectrum
            self._seam_ratio += seam_ratio
        self._n_samples += batch_size

    def _group_logs(self, name: str, values: torch.Tensor) -> MetricsDict:
        logs: MetricsDict = {}
        for index, channel in enumerate(self.tensor_map.prognostic_var_names):
            logs[f"{name}/channel/{channel}"] = float(values[index].cpu())
        for variable, indices in self.tensor_map.VAR_3D_IDX.items():
            logs[f"{name}/variable/{variable}"] = float(
                values[indices.long()].mean().cpu()
            )
        for depth, indices in self.tensor_map.DP_3D_IDX.items():
            logs[f"{name}/depth/{depth}"] = float(values[indices.long()].mean().cpu())
        return logs

    def _spectrum_image(
        self,
        target_spectrum: torch.Tensor,
        generated_spectrum: torch.Tensor,
    ):
        variables = self.tensor_map.VAR_SET
        figure, axes = plt.subplots(
            len(variables), 1, figsize=(7, 2.5 * len(variables)), squeeze=False
        )
        for axis, variable in zip(axes[:, 0], variables, strict=True):
            indices = self.tensor_map.VAR_3D_IDX[variable].long()
            target = target_spectrum[indices].mean(dim=0).cpu()
            generated = generated_spectrum[indices].mean(dim=0).cpu()
            wavenumber = torch.arange(target.shape[0])
            axis.loglog(wavenumber[1:], target[1:], label="target")
            axis.loglog(wavenumber[1:], generated[1:], label="generated")
            axis.set_title(variable)
            axis.set_ylabel("normalized power")
            axis.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("zonal wavenumber")
        axes[0, 0].legend()
        figure.tight_layout()
        image = WandBLogger.get_instance().Image(
            figure,
            caption="One-step zonal power spectra averaged by variable group.",
        )
        plt.close(figure)
        return image

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        if (
            self._n_samples == 0
            or self._target_spectrum is None
            or self._generated_spectrum is None
            or self._seam_ratio is None
        ):
            raise ValueError("No batches have been recorded.")
        target_spectrum = all_reduce_mean(self._target_spectrum / self._n_samples)
        generated_spectrum = all_reduce_mean(self._generated_spectrum / self._n_samples)
        seam_ratio = all_reduce_mean(self._seam_ratio / self._n_samples)
        high_frequency_ratio = high_wavenumber_power_ratio(
            generated_spectrum, target_spectrum
        )

        logs: MetricsDict = {}
        logs.update(
            self._group_logs(
                f"{label}/high_wavenumber_power_ratio", high_frequency_ratio
            )
        )
        logs.update(self._group_logs(f"{label}/patch_seam_jump_ratio", seam_ratio))
        if is_main_process():
            logs[f"{label}/zonal_power_spectrum"] = self._spectrum_image(
                target_spectrum, generated_spectrum
            )
        return logs
