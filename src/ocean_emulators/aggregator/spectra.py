import dataclasses
import logging

import matplotlib.pyplot as plt
import torch
import wandb

from ocean_emulators.utils.wandb import MetricsDict, WandBLogger

logger = logging.getLogger(__name__)

type SpectraLocation = tuple[str, tuple[float, float], tuple[float, float]]


def _detrend_linear_torch(data: torch.Tensor) -> torch.Tensor:
    """Remove a best-fit linear plane from 4D tensors shaped (B, C, H, W)."""
    b, c, h, w = data.shape
    device = data.device
    dtype = data.dtype

    y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    a = torch.stack(
        [x_grid.flatten(), y_grid.flatten(), torch.ones_like(x_grid).flatten()], dim=1
    )
    bc = b * c
    data_flat = data.reshape(bc, h * w)
    coeffs, _, _, _ = torch.linalg.lstsq(a, data_flat.T)
    plane = (a @ coeffs.permute(1, 0).unsqueeze(-1)).reshape(bc, h, w)
    detrended = data.reshape(bc, h, w) - plane
    return detrended.reshape(b, c, h, w)


def compute_isotropic_spectrum_torch(
    data: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    num_bins: int | None = None,
    n_factor: int = 4,
    remove_mean: bool = True,
    detrend: str | None = None,
    window: str | None = "Hann",
    truncate: bool = True,
    cutoff_before_bins: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute isotropic 1D spectrum from 2D/3D/4D fields."""
    device = data.device
    dtype = data.dtype
    orig_dim = data.dim()

    if orig_dim == 2:
        data = data.reshape(1, 1, *data.shape)
    elif orig_dim == 3:
        data = data.unsqueeze(1)
    elif orig_dim != 4:
        raise ValueError("Input data must be 2D, 3D, or 4D (B, C, H, W)")

    b, c, h, w = data.shape
    bc = b * c
    lx = w * dx
    ly = h * dy

    if num_bins is None:
        num_bins = min(h, w) // n_factor

    if detrend == "linear":
        data = _detrend_linear_torch(data)
    elif detrend == "constant" or remove_mean:
        data = data - torch.mean(data, dim=(-2, -1), keepdim=True)

    if window and window.lower() == "hann":
        win_y = torch.hann_window(h, device=device, dtype=dtype).unsqueeze(1)
        win_x = torch.hann_window(w, device=device, dtype=dtype).unsqueeze(0)
        win_2d = (win_y * win_x).reshape(1, 1, h, w)
        window_correction = torch.mean(win_2d**2).item()
        data = data * win_2d
    else:
        window_correction = 1.0

    fft_2d = torch.fft.rfft2(data, norm="forward")
    power_2d = torch.abs(fft_2d) ** 2
    power_2d = power_2d / window_correction
    psd_2d = power_2d * (lx * ly)

    k_x = torch.fft.rfftfreq(w, d=dx, device=device, dtype=dtype)
    k_y = torch.fft.fftfreq(h, d=dy, device=device, dtype=dtype)
    k_x_nyq = 1.0 / (2.0 * dx)
    k_y_nyq = 1.0 / (2.0 * dy)

    k_y_grid, k_x_grid = torch.meshgrid(k_y, k_x, indexing="ij")
    k_mag = torch.sqrt(k_x_grid**2 + k_y_grid**2)
    k_max_domain = float(k_mag.max().item())

    if truncate and cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
    else:
        k_max = k_max_domain

    k_bins = torch.linspace(0.0, k_max, num_bins + 1, device=device, dtype=dtype)
    if truncate and not cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
        k_bins = k_bins[k_bins < k_max_cutoff]
        num_bins = k_bins.numel() - 1
    k_bins_centers = (k_bins[:-1] + k_bins[1:]) / 2

    k_mag_flat = k_mag.flatten()
    bin_edges = k_bins[1:-1]
    bin_indices = torch.bucketize(k_mag_flat, bin_edges, right=True)

    n_flat = k_mag_flat.shape[0]
    psd_flat_batched = psd_2d.reshape(bc, n_flat)
    bin_indices_batched = bin_indices.expand(bc, -1)
    binned_psd_sum = torch.zeros(bc, num_bins, device=device, dtype=dtype)
    binned_psd_sum.scatter_add_(dim=1, index=bin_indices_batched, src=psd_flat_batched)

    binned_counts = torch.bincount(bin_indices, minlength=num_bins).float()
    binned_counts[binned_counts == 0] = torch.nan

    iso_psd_binned = binned_psd_sum / binned_counts.unsqueeze(0)
    iso_spectrum = iso_psd_binned * k_bins_centers.unsqueeze(0)
    iso_spectrum = iso_spectrum.reshape(b, c, num_bins)
    iso_spectrum[..., 0] = torch.nan

    if orig_dim == 2:
        iso_spectrum = iso_spectrum.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        iso_spectrum = iso_spectrum.squeeze(1)

    return k_bins_centers, iso_spectrum


@dataclasses.dataclass(frozen=True)
class _LocationSelection:
    name: str
    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    lon_mask: torch.Tensor
    lat_mask: torch.Tensor
    dx: float
    dy: float


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
    ):
        self._metadata = metadata or {}
        self._prognostic_var_names = list(prognostic_var_names or [])
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

            dy = float(torch.diff(lat_sel).abs().mean().item())
            dx = float(torch.diff(lon_sel).abs().mean().item())
            if dx == 0.0 or dy == 0.0:
                logger.warning(
                    "Skipping spectra location '%s': zero grid spacing in selected box.",
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
                    dx=dx,
                    dy=dy,
                )
            )
        return selections

    @staticmethod
    def _prep_field_for_fft(data: torch.Tensor) -> torch.Tensor | None:
        finite = torch.isfinite(data)
        if finite.sum() < 4:
            return None
        centered = data - torch.nanmean(data)
        return torch.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_spectrum(
        self, data: torch.Tensor, selection: _LocationSelection
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        patch = data.detach().cpu()
        patch = patch[selection.lat_mask][:, selection.lon_mask]
        prepared_patch = self._prep_field_for_fft(patch)
        if prepared_patch is None:
            return None
        patch = prepared_patch
        k, spectrum = compute_isotropic_spectrum_torch(
            patch,
            dx=selection.dx,
            dy=selection.dy,
            remove_mean=False,
            detrend="constant",
            window="hann",
            truncate=True,
        )
        valid = torch.isfinite(k) & torch.isfinite(spectrum) & (k > 0) & (spectrum > 0)
        if valid.sum() < 1:
            return None
        return k[valid], spectrum[valid]

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
        target_spec = self._compute_spectrum(target, selection)
        pred_spec = self._compute_spectrum(prediction, selection)
        if target_spec is None or pred_spec is None:
            return None

        k_target, s_target = target_spec
        k_pred, s_pred = pred_spec

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(k_target.numpy(), s_target.numpy(), label="target", linewidth=2)
        ax.loglog(k_pred.numpy(), s_pred.numpy(), label="prediction", linewidth=2)
        ax.set_xlabel("wavenumber [cycles/degree]")
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
