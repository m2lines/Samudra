import dataclasses
import logging
import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
import wandb

from ocean_emulators.utils import spectrum as spectrum_utils
from ocean_emulators.utils.data import DataSource
from ocean_emulators.utils.wandb import MetricsDict, WandBLogger

logger = logging.getLogger(__name__)

type SpectraLocation = tuple[str, tuple[float, float], tuple[float, float]]

_KM_PER_DEGREE = 111.32


def precompute_spatial_temporal_means(
    source: DataSource,
    prognostic_var_names: Iterable[str],
) -> dict[str, torch.Tensor]:
    """
    Precompute mean fields over the full available time axis for spectra anomalies.

    This is intentionally scoped to the currently-used spatial spectra algorithm.
    Temporal trend/cycle precomputes are not needed until temporal PSD logging is added.
    """
    available_var_names = [
        name for name in prognostic_var_names if name in source.data.variables
    ]
    if not available_var_names:
        return {}

    logger.info(
        "Precomputing spectra temporal means for %d variables from %s",
        len(available_var_names),
        source.name,
    )
    means_ds = source.data[available_var_names].mean(dim="time").compute()
    return {
        name: torch.as_tensor(means_ds[name].values, dtype=torch.float32)
        for name in available_var_names
    }


@dataclasses.dataclass(frozen=True)
class _LocationSelection:
    name: str
    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    lon_mask: torch.Tensor
    lat_mask: torch.Tensor
    dx_km: float
    dy_km: float


class SpectraLogger:
    """Computes and plots target/prediction spectra for configured lat/lon boxes."""

    def __init__(
        self,
        *,
        lat: torch.Tensor | None,
        lon: torch.Tensor | None,
        locations: list[SpectraLocation] | None,
        prognostic_var_names: list[str] | None,
        metadata: dict[str, dict[str, str]] | None = None,
        temporal_means: dict[str, torch.Tensor] | None = None,
    ):
        self._metadata = metadata or {}
        self._prognostic_var_names = list(prognostic_var_names or [])
        self._temporal_means = {
            name: value.detach().cpu().float()
            for name, value in (temporal_means or {}).items()
        }
        self._locations = self._build_locations(lat, lon, locations or [])

    @property
    def enabled(self) -> bool:
        return bool(self._locations) and bool(self._prognostic_var_names)

    @staticmethod
    def _normalize_lon_bounds(
        lon: torch.Tensor, lon_bounds: tuple[float, float]
    ) -> tuple[float, float]:
        lo, hi = lon_bounds
        lon_min = float(torch.min(lon).item())
        lon_max = float(torch.max(lon).item())
        if lon_min >= 0 and (lo < 0 or hi < 0):
            return (lo % 360.0, hi % 360.0)
        if lon_max <= 180 and (lo > 180 or hi > 180):

            def to_signed(x: float) -> float:
                return ((x + 180.0) % 360.0) - 180.0

            return (to_signed(lo), to_signed(hi))
        return lo, hi

    @staticmethod
    def _deg_lon_to_km(delta_deg: float, lat_deg: float) -> float:
        return abs(delta_deg) * _KM_PER_DEGREE * max(
            abs(math.cos(math.radians(lat_deg))), 1e-6
        )

    @staticmethod
    def _deg_lat_to_km(delta_deg: float) -> float:
        return abs(delta_deg) * _KM_PER_DEGREE

    def _build_locations(
        self,
        lat: torch.Tensor | None,
        lon: torch.Tensor | None,
        locations: list[SpectraLocation],
    ) -> list[_LocationSelection]:
        if lat is None or lon is None:
            return []

        lat_cpu = lat.detach().cpu().float()
        lon_cpu = lon.detach().cpu().float()
        selections: list[_LocationSelection] = []
        for name, lon_bounds_raw, lat_bounds_raw in locations:
            lat_lo, lat_hi = sorted(
                (float(lat_bounds_raw[0]), float(lat_bounds_raw[1]))
            )
            lon_lo, lon_hi = self._normalize_lon_bounds(lon_cpu, lon_bounds_raw)

            lat_mask = (lat_cpu >= lat_lo) & (lat_cpu <= lat_hi)
            if lon_lo <= lon_hi:
                lon_mask = (lon_cpu >= lon_lo) & (lon_cpu <= lon_hi)
            else:
                # Allow wrap-around boxes, e.g. (350, 20).
                lon_mask = (lon_cpu >= lon_lo) | (lon_cpu <= lon_hi)

            lat_sel = lat_cpu[lat_mask]
            lon_sel = lon_cpu[lon_mask]
            if lat_sel.numel() < 4 or lon_sel.numel() < 4:
                logger.warning(
                    "Skipping spectra location '%s': selected box is too small.",
                    name,
                )
                continue

            lat_diff = torch.diff(lat_sel).abs()
            lon_diff = torch.diff(lon_sel).abs()
            if lat_diff.numel() == 0 or lon_diff.numel() == 0:
                logger.warning(
                    "Skipping spectra location '%s': zero grid spacing in selected box.",
                    name,
                )
                continue

            dy_km = self._deg_lat_to_km(float(lat_diff.median().item()))
            lat_mid = float(lat_sel.mean().item())
            dx_km = self._deg_lon_to_km(float(lon_diff.median().item()), lat_mid)
            if dx_km == 0.0 or dy_km == 0.0:
                logger.warning(
                    "Skipping spectra location '%s': zero km spacing in selected box.",
                    name,
                )
                continue

            selections.append(
                _LocationSelection(
                    name=name,
                    lon_bounds=(lon_lo, lon_hi),
                    lat_bounds=(lat_lo, lat_hi),
                    lon_mask=lon_mask,
                    lat_mask=lat_mask,
                    dx_km=dx_km,
                    dy_km=dy_km,
                )
            )
        return selections

    def _compute_spectrum(
        self,
        data: torch.Tensor,
        var_name: str,
        selection: _LocationSelection,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        patch = data.detach().cpu().float()
        patch = patch[selection.lat_mask][:, selection.lon_mask]
        temporal_mean = self._temporal_means.get(var_name)
        if temporal_mean is not None and temporal_mean.ndim == 2:
            mean_patch = temporal_mean[selection.lat_mask][:, selection.lon_mask]
            patch = patch - mean_patch

        finite = torch.isfinite(patch)
        if finite.sum() < 4:
            return None
        patch = torch.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        k, spectrum = spectrum_utils.compute_isotropic_spectrum_torch(
            patch,
            dx=selection.dx_km,
            dy=selection.dy_km,
            n_factor=2,
            detrend="linear",
            window="hann",
            truncate=True,
        )
        valid = torch.isfinite(k) & torch.isfinite(spectrum) & (k > 0) & (spectrum > 0)
        if valid.sum() < 1:
            return None
        k_rad_per_km = k[valid] * (2.0 * torch.pi)
        return k_rad_per_km, spectrum[valid]

    def _get_caption(
        self, *, var_name: str, selection: _LocationSelection, forecast_step: int | None
    ) -> str:
        if var_name in self._metadata:
            caption_name = self._metadata[var_name]["long_name"]
            units = self._metadata[var_name]["units"]
        else:
            caption_name = var_name
            units = "unknown_units"
        caption = (
            f"{caption_name} spectra in {selection.name} "
            f"(lon={selection.lon_bounds}, lat={selection.lat_bounds}) [{units}]"
        )
        if forecast_step is not None:
            caption += f" forecast_step={forecast_step}"
        return caption

    def _make_image(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        var_name: str,
        selection: _LocationSelection,
        forecast_step: int | None,
    ):
        target_spec = self._compute_spectrum(target, var_name, selection)
        pred_spec = self._compute_spectrum(prediction, var_name, selection)
        if target_spec is None or pred_spec is None:
            return None

        k_target, s_target = target_spec
        k_pred, s_pred = pred_spec

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(k_target.numpy(), s_target.numpy(), label="target", linewidth=2)
        ax.loglog(k_pred.numpy(), s_pred.numpy(), label="prediction", linewidth=2)
        ax.set_xlabel("angular wavenumber [rad/km]")
        ax.set_ylabel("k * P(k)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.set_title(var_name)
        caption = self._get_caption(
            var_name=var_name, selection=selection, forecast_step=forecast_step
        )
        try:
            image = WandBLogger.get_instance().Image(fig, caption=caption)
        except ValueError:
            # Tests and some local utilities may call this without initializing wandb.
            image = wandb.Image(fig, caption=caption)
        plt.close(fig)
        return image

    def _var_names(self, data: dict[str, torch.Tensor]) -> list[str]:
        return [name for name in self._prognostic_var_names if name in data]

    def get_logs_for_single_step(
        self,
        *,
        target_data: dict[str, torch.Tensor],
        gen_data: dict[str, torch.Tensor],
        time_index: int,
        sample_index: int,
        key_prefix: str = "",
        forecast_step: int | None = None,
    ) -> MetricsDict:
        logs: MetricsDict = {}
        if not self.enabled:
            return logs

        for var_name in self._var_names(gen_data):
            target = target_data[var_name][sample_index, time_index]
            prediction = gen_data[var_name][sample_index, time_index]
            for selection in self._locations:
                image = self._make_image(
                    target=target,
                    prediction=prediction,
                    var_name=var_name,
                    selection=selection,
                    forecast_step=forecast_step,
                )
                if image is None:
                    continue
                logs[f"{key_prefix}spectra_{selection.name}/{var_name}"] = image
        return logs

    def get_logs_for_all_steps(
        self,
        *,
        target_data: dict[str, torch.Tensor],
        gen_data: dict[str, torch.Tensor],
        sample_index: int,
        key_prefix: str = "",
        forecast_step_offset: int = 0,
    ) -> list[MetricsDict]:
        if not self.enabled:
            return []
        if not gen_data:
            return []
        first_name = next(iter(gen_data))
        n_steps = int(gen_data[first_name].shape[1])

        step_logs: list[MetricsDict] = []
        for i_step in range(n_steps):
            step_logs.append(
                self.get_logs_for_single_step(
                    target_data=target_data,
                    gen_data=gen_data,
                    time_index=i_step,
                    sample_index=sample_index,
                    key_prefix=key_prefix,
                    forecast_step=forecast_step_offset + i_step,
                )
            )
        return step_logs
