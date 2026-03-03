"""
Spectrum Analysis Toolkit for Ocean Emulator Evaluation
========================================================

This module provides functions for computing and visualizing power spectra
of ocean model variables, including:

1. **Spatial (Isotropic) Spectra** — 2D wavenumber power spectra for scalar
   fields (e.g. temperature, salinity) and kinetic energy (KE = 0.5*(u²+v²)).

2. **Temporal Spectra** — 1D frequency power spectra for scalar fields and KE,
   with seasonal-cycle removal and linear detrending.

Each spectrum type has a ``plot_*_comparison_together`` interface that compares
emulators where each carries its own truth dataset (useful for multi-resolution
comparisons).


Preprocessing Pipelines
-----------------------

The two spectrum types apply **different** preprocessing steps before FFT.
Below is a summary:

**A. Spatial (Isotropic) Spectrum** (``compute_isotropic_spectrum_torch``)

  Applied to each 2-D spatial snapshot independently:

  1. **Remove time-mean (anomaly)**
     Before calling the core spectrum function, the plot helpers subtract the
     temporal mean at each grid point:  ``anom = field - field.mean("time")``.
     This removes the climatological mean state so we analyse *anomalies* only.

  2. **Linear spatial detrend** (``detrend='linear'``)
     A least-squares plane ``a·x + b·y + c`` is fitted to each (H, W) slice
     and subtracted.  This removes large-scale spatial gradients that would
     otherwise dominate the low-wavenumber bins.
     *(Alternative: ``detrend='constant'`` only removes the spatial mean.)*

  3. **Hann (Hanning) window** (``window='hann'``)
     A 2-D Hann taper is applied to suppress spectral leakage caused by the
     non-periodic domain boundaries.  The PSD is corrected by dividing by
     ``mean(window²)`` so that the total variance is preserved.

  4. **2-D rFFT → radial binning**
     The 2-D real FFT is computed, converted to power spectral density (PSD),
     and then azimuthally averaged into isotropic wavenumber bins.
     The final output is the variance-preserving form ``k · P(k)``.

  Pipeline diagram::

      raw field
        │
        ▼
      subtract time-mean  →  anomaly field
        │
        ▼
      linear spatial detrend (remove plane a·x + b·y + c)
        │
        ▼
      apply 2-D Hann window
        │
        ▼
      2-D rFFT  →  |FFT|²  →  PSD(kx, ky)
        │
        ▼
      radial (isotropic) binning  →  k · P(k)


**B. Temporal Spectrum** (``_compute_temporal_psd``)

  Applied to the time series at *each* spatial grid point, then spatially
  averaged:

  1. **NaN → 0 fill**
     Land / missing values are replaced with zero before further processing.

  2. **Linear temporal detrend** (``scipy.signal.detrend``, type='linear')
     Removes both the mean and any linear trend (a·t + b) from each grid-point
     time series.  This prevents low-frequency spectral leakage from a
     long-term drift.

  3. **Remove seasonal cycle (climatology)**
     A repeating annual climatology is computed by reshaping the time series
     into complete years, averaging across years to get a "mean year", and
     then tiling and subtracting it.  This removes the dominant annual and
     semi-annual peaks so that the remaining spectrum highlights inter-annual
     and sub-seasonal variability.

  4. **Hann (Hanning) window** (``scipy.signal.windows.hann``)
     A 1-D Hann window is applied along the time axis to reduce spectral
     leakage.  The PSD is corrected by ``mean(window²)``.

  5. **1-D rFFT → one-sided PSD**
     The one-sided PSD is computed.  The DC component and (for even-length
     series) the Nyquist component are *not* doubled, following standard
     convention.

  Pipeline diagram::

      raw time series  (time, y, x)
        │
        ▼
      fill NaN → 0
        │
        ▼
      linear temporal detrend  (remove a·t + b)
        │
        ▼
      subtract seasonal climatology  (mean annual cycle)
        │
        ▼
      apply 1-D Hann window along time
        │
        ▼
      1-D rFFT → |FFT|² → one-sided PSD
        │
        ▼
      average PSD over all spatial grid points


**C. Kinetic Energy Spectra**

  For KE variants (``plot_ke_*``), the same pipelines above are applied to
  u-velocity and v-velocity *separately*, and the final KE spectrum is:

      KE_spectrum = 0.5 × (spectrum_u + spectrum_v)


Dependencies
------------
numpy, torch, matplotlib, xarray, scipy

Quick Start
-----------
>>> import xarray as xr
>>> from spectrum import plot_isotropic_spectrum_comparison_together
>>>
>>> emulators = {
...     "Model-A": {
...         "ds": xr.open_zarr("model_a.zarr"),
...         "color": "tab:blue",
...         "truth": xr.open_zarr("truth_a.zarr"),
...         "truth_label": "OM4-1deg",
...     },
...     "Model-B": {
...         "ds": xr.open_zarr("model_b.zarr"),
...         "color": "tab:orange",
...         "truth": xr.open_zarr("truth_b.zarr"),
...         "truth_label": "OM4-halfdeg",
...     },
... }
>>> plot_isotropic_spectrum_comparison_together(emulators, var_to_eval="thetao")
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, LogFormatterMathtext
from scipy import signal


# ============================================================================
# §1  Core Computation — Detrending & Isotropic Spectrum
# ============================================================================

def _detrend_linear_torch(data: torch.Tensor) -> torch.Tensor:
    """Remove a linear plane of best fit from 4-D ``(B, C, H, W)`` data.

    A plane ``a·x + b·y + c`` is fitted (least-squares) to each ``(H, W)``
    slice independently and subtracted.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Detrended tensor with the same shape as *data*.
    """
    B, C, H, W = data.shape
    device = data.device
    dtype = data.dtype

    y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    A = torch.stack([X.flatten(), Y.flatten(), torch.ones_like(X).flatten()], dim=1)

    B_prime = B * C
    data_flat = data.reshape(B_prime, H * W)

    coeffs, _, _, _ = torch.linalg.lstsq(A, data_flat.T)

    plane = (A @ coeffs.permute(1, 0).unsqueeze(-1)).reshape(B_prime, H, W)

    detrended_data = data.reshape(B_prime, H, W) - plane

    return detrended_data.reshape(B, C, H, W)


def compute_isotropic_spectrum_torch(
    data: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    num_bins: int | None = None,
    n_factor: int = 4,
    remove_mean: bool = True,
    detrend: str | None = None,
    window: str = "Hann",
    truncate: bool = True,
    cutoff_before_bins: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the isotropic 1-D power spectrum from 2-D / 3-D / 4-D data.

    The result matches ``xrft.isotropic_power_spectrum(scaling="density")``.

    Parameters
    ----------
    data : torch.Tensor
        Input data — ``(H, W)``, ``(B, H, W)`` or ``(B, C, H, W)``.
    dx, dy : float
        Grid spacing in the x / y dimensions (metres).
    num_bins : int, optional
        Number of radial wavenumber bins.  Defaults to ``min(H, W) // n_factor``.
    n_factor : int
        Denominator used when *num_bins* is ``None``.
    remove_mean : bool
        Subtract the spatial mean before computing the FFT.  Overridden by
        *detrend*.
    detrend : ``"linear"`` | ``"constant"`` | ``None``
        Detrending mode applied before FFT.
    window : ``"hann"`` | ``None``
        Tapering window.
    truncate : bool
        Truncate spectrum at the smallest Nyquist frequency.
    cutoff_before_bins : bool
        When *truncate* is ``True``, set the bin upper limit to the Nyquist
        cutoff **before** computing bin edges (matches *xrft* behaviour).

    Returns
    -------
    k_bins_centers : torch.Tensor
        1-D tensor of radial wavenumber bin centres.  Shape ``(num_bins,)``.
    iso_spectrum : torch.Tensor
        Variance-preserving spectrum ``k · P(k)``.
        Shape matches the input batch/channel dimensions.
    """
    device = data.device
    dtype = data.dtype
    orig_dim = data.dim()

    if orig_dim == 2:
        data = data.reshape(1, 1, *data.shape)
    elif orig_dim == 3:
        data = data.unsqueeze(1)
    elif orig_dim != 4:
        raise ValueError("Input data must be 2D, 3D, or 4D (B, C, H, W)")

    B, C, H, W = data.shape
    B_prime = B * C
    Lx = W * dx
    Ly = H * dy

    if num_bins is None:
        num_bins = min(H, W) // n_factor

    # --- Preprocessing ---------------------------------------------------
    if detrend == "linear":
        data = _detrend_linear_torch(data)
    elif detrend == "constant" or remove_mean:
        data = data - torch.mean(data, dim=(-2, -1), keepdim=True)

    if window and window.lower() == "hann":
        win_y = torch.hann_window(H, device=device, dtype=dtype).unsqueeze(1)
        win_x = torch.hann_window(W, device=device, dtype=dtype).unsqueeze(0)
        win_2d = (win_y * win_x).reshape(1, 1, H, W)

        window_correction = torch.mean(win_2d**2).item()
        data = data * win_2d
    else:
        window_correction = 1.0

    # --- 2-D FFT & PSD ---------------------------------------------------
    fft_2d = torch.fft.rfft2(data, norm="forward")

    power_2d = torch.abs(fft_2d) ** 2
    power_2d = power_2d / window_correction

    psd_2d = power_2d * (Lx * Ly)

    # --- Wavenumber grid --------------------------------------------------
    k_x = torch.fft.rfftfreq(W, d=dx, device=device, dtype=dtype)
    k_y = torch.fft.fftfreq(H, d=dy, device=device, dtype=dtype)

    k_x_nyq = 1.0 / (2.0 * dx)
    k_y_nyq = 1.0 / (2.0 * dy)

    k_Y, k_X = torch.meshgrid(k_y, k_x, indexing="ij")
    k_mag = torch.sqrt(k_X**2 + k_Y**2)

    k_max_domain = k_mag.max()

    if truncate and cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
    else:
        k_max = k_max_domain

    k_bins = torch.linspace(0, k_max, num_bins + 1, device=device, dtype=dtype)
    if truncate and not cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
        k_bins = k_bins[k_bins < k_max_cutoff]
        num_bins = k_bins.numel() - 1
    k_bins_centers = (k_bins[:-1] + k_bins[1:]) / 2

    # --- Radial binning ---------------------------------------------------
    k_mag_flat = k_mag.flatten()
    bin_edges = k_bins[1:-1]

    bin_indices = torch.bucketize(k_mag_flat, bin_edges, right=True)

    N_flat = k_mag_flat.shape[0]
    psd_flat_batched = psd_2d.reshape(B_prime, N_flat)

    bin_indices_batched = bin_indices.expand(B_prime, -1)
    binned_psd_sum = torch.zeros(B_prime, num_bins, device=device, dtype=dtype)
    binned_psd_sum.scatter_add_(dim=1, index=bin_indices_batched, src=psd_flat_batched)
    binned_counts = torch.bincount(bin_indices, minlength=num_bins)

    binned_counts_safe = binned_counts.float()
    binned_counts_safe[binned_counts_safe == 0] = torch.nan

    iso_psd_binned = binned_psd_sum / binned_counts_safe.unsqueeze(0)
    iso_spectrum = iso_psd_binned * k_bins_centers.unsqueeze(0)
    iso_spectrum = iso_spectrum.reshape(B, C, num_bins)
    iso_spectrum[..., 0] = torch.nan

    if orig_dim == 2:
        iso_spectrum = iso_spectrum.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        iso_spectrum = iso_spectrum.squeeze(1)
    return k_bins_centers, iso_spectrum


# ============================================================================
# §2  Temporal Spectrum Helpers (numpy / scipy)
# ============================================================================

def _infer_dt_days(ds) -> float:
    """Infer the time-step size in days from an xarray Dataset."""
    try:
        time_vals = ds.time.values
        if len(time_vals) > 1:
            delta_t = time_vals[1] - time_vals[0]
            if isinstance(delta_t, np.timedelta64):
                return float(delta_t) / np.timedelta64(1, "D")
            else:
                return delta_t.days + delta_t.seconds / 86400.0
    except Exception:
        pass
    return 1.0


def _compute_temporal_psd(data_np: np.ndarray, dt_days: float):
    """Temporal PSD at each spatial point, then spatially averaged.

    Parameters
    ----------
    data_np : np.ndarray
        Shape ``(time, …)``; NaNs will be replaced with 0.
    dt_days : float
        Time-step in days.

    Returns
    -------
    freqs : np.ndarray   — frequency array in cycles / year.
    psd   : np.ndarray   — spatially-averaged PSD.
    """
    n_time = data_np.shape[0]
    spatial_shape = data_np.shape[1:]
    dt_years = dt_days / 365.25
    fs = 1.0 / dt_years  # cycles / year

    data_np = np.nan_to_num(data_np, nan=0.0)

    # Linear detrend along time axis
    data_detrended = signal.detrend(data_np, axis=0, type="linear")

    # Remove seasonal cycle (monthly climatology)
    steps_per_year = int(round(365.25 / dt_days))
    n_full = (n_time // steps_per_year) * steps_per_year
    if n_full >= steps_per_year:
        climatology = (
            data_detrended[:n_full]
            .reshape(-1, steps_per_year, *spatial_shape)
            .mean(axis=0)
        )
        seasonal = np.tile(climatology, (n_time // steps_per_year + 1,) + (1,) * len(spatial_shape))[:n_time]
        data_detrended = data_detrended - seasonal

    # Hanning window
    win = signal.windows.hann(n_time)
    data_windowed = data_detrended * win[(...,) + (np.newaxis,) * len(spatial_shape)]
    window_correction = np.mean(win**2)

    # FFT
    fft_result = np.fft.rfft(data_windowed, axis=0)
    power_spectrum = np.abs(fft_result) ** 2

    # One-sided PSD
    psd_spatial = 2.0 * power_spectrum / (fs * n_time * window_correction)
    psd_spatial[0] /= 2.0  # DC component
    if n_time % 2 == 0:
        psd_spatial[-1] /= 2.0  # Nyquist

    psd = np.nanmean(psd_spatial, axis=tuple(range(1, psd_spatial.ndim)))
    freqs = np.fft.rfftfreq(n_time, d=dt_years)
    return freqs, psd


# ============================================================================
# §3  Plot Formatting Helpers
# ============================================================================

def _format_lon_lat_strings(lon_slice, lat_slice):
    """Return human-readable lon/lat range strings for titles."""
    lon_start = lon_slice.start if lon_slice.start is not None else 0
    lon_stop = lon_slice.stop if lon_slice.stop is not None else 360
    lon_start_display = f"{lon_start}°E"
    lon_stop_display = f"{lon_stop - 360}°W" if lon_stop > 180 else f"{lon_stop}°E"

    lat_start = lat_slice.start if lat_slice.start is not None else -90
    lat_stop = lat_slice.stop if lat_slice.stop is not None else 90
    lat_start_display = f"{abs(lat_start)}°S" if lat_start < 0 else f"{lat_start}°N"
    lat_stop_display = f"{abs(lat_stop)}°S" if lat_stop < 0 else f"{lat_stop}°N"

    return lon_start_display, lon_stop_display, lat_start_display, lat_stop_display


# ============================================================================
# §4  Spatial (Isotropic) Spectrum
# ============================================================================

def plot_isotropic_spectrum_comparison_together(
    emulators_dict: dict,
    var_to_eval: str = "thetao",
    lev_idx: int = 0,
    time_window: int = 100,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers: list[float] | None = None,
    show_ratios: bool = True,
):
    """Plot isotropic spectrum where each emulator carries its own truth.

    Parameters
    ----------
    emulators_dict : dict
        ``{label: {"ds": xr.Dataset, "color": str,
                   "truth": xr.Dataset, "truth_label": str}, …}``
    var_to_eval, lev_idx, time_window, lon_slice, lat_slice : same as above.
    target_wavenumbers : list[float], optional
    show_ratios : bool
    """
    if target_wavenumbers is None:
        target_wavenumbers = [0.01, 0.02]

    def _get_spectrum(ds_target):
        dx_mean = float(ds_target.dx.sel({"x": lon_slice, "y": lat_slice}).mean().values)
        dy_mean = float(ds_target.dy.sel({"x": lon_slice, "y": lat_slice}).mean().values)
        raw = ds_target[var_to_eval].isel(lev=lev_idx, time=slice(None, time_window))
        raw = raw.transpose("time", ...).sel({"x": lon_slice, "y": lat_slice})
        anom = raw - raw.mean(dim="time")
        k, spec = compute_isotropic_spectrum_torch(
            torch.as_tensor(anom.fillna(0).values),
            dx=dx_mean, dy=dy_mean, n_factor=2,
            detrend="linear", window="hann", cutoff_before_bins=False,
        )
        return k.cpu().numpy() * 1000 * 2 * np.pi, spec.cpu().numpy()

    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        k_truth, spec_truth = _get_spectrum(emu_data["truth"])
        k_emu, spec_emu = _get_spectrum(emu_data["ds"])
        all_spectra[label] = {
            "k_truth": k_truth, "spec_truth": spec_truth,
            "k_emu": k_emu, "spec_emu": spec_emu,
            "truth_label": emu_data.get("truth_label", "Truth"),
        }

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted_truths: set[str] = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        tl = data["truth_label"]
        if tl not in plotted_truths:
            ax.loglog(data["k_truth"], data["spec_truth"].mean(0), "--",
                      label=f"{tl} ({var_to_eval} Anomalies)",
                      linewidth=2, color=color, alpha=0.7)
            plotted_truths.add(tl)
        ax.loglog(data["k_emu"], data["spec_emu"].mean(0), "-",
                  label=f"{label} ({var_to_eval} Anomalies)", linewidth=2, color=color)

    if show_ratios:
        for target_k in target_wavenumbers:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_t = np.argmin(np.abs(data["k_truth"] - target_k))
                idx_e = np.argmin(np.abs(data["k_emu"] - target_k))
                actual_k = data["k_truth"][idx_t]
                val_t = data["spec_truth"].mean(0)[idx_t]
                val_e = data["spec_emu"].mean(0)[idx_e]
                ratio = val_e / val_t if val_t > 0 else 0
                color = emulators_dict[label]["color"]
                ax.text(actual_k * 1.1, val_e * (1.5**i),
                        f'{label}/{data["truth_label"]}={ratio:.3f}',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color))
            ax.axvline(x=actual_k, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=False))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
    ax.set_ylabel("Power Spectral Density")

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    lon_s, lon_e, lat_s, lat_e = _format_lon_lat_strings(lon_slice, lat_slice)
    ax.set_title(
        f"Isotropic Power Spectrum of {var_to_eval} ({depth_value}m)\n"
        f"Region: {lon_s}–{lon_e}, {lat_s}–{lat_e}",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================================
# §5  KE Spatial Spectrum
# ============================================================================

def plot_ke_spectrum_comparison_together(
    emulators_dict: dict,
    var_u: str = "uo",
    var_v: str = "vo",
    lev_idx: int = 0,
    time_window: int = 100,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers: list[float] | None = None,
    show_ratios: bool = True,
):
    """Plot KE isotropic spectrum — each emulator with its own truth.

    Parameters
    ----------
    emulators_dict : dict
        ``{label: {"ds": xr.Dataset, "color": str,
                   "truth": xr.Dataset, "truth_label": str}, …}``
    """
    if target_wavenumbers is None:
        target_wavenumbers = [0.01, 0.02]

    def _get_ke_spectrum(ds_target):
        dx_mean = float(ds_target.dx.sel({"x": lon_slice, "y": lat_slice}).mean().values)
        dy_mean = float(ds_target.dy.sel({"x": lon_slice, "y": lat_slice}).mean().values)

        def _process(var_name):
            d = ds_target[var_name].isel(lev=lev_idx, time=slice(None, time_window))
            d = d.transpose("time", ...).sel({"x": lon_slice, "y": lat_slice})
            return torch.as_tensor((d - d.mean(dim="time")).fillna(0).values)

        u_t = _process(var_u)
        v_t = _process(var_v)
        k_c, su = compute_isotropic_spectrum_torch(u_t, dx=dx_mean, dy=dy_mean, n_factor=2,
                                                    detrend="linear", window="hann")
        _, sv = compute_isotropic_spectrum_torch(v_t, dx=dx_mean, dy=dy_mean, n_factor=2,
                                                  detrend="linear", window="hann")
        return k_c.cpu().numpy() * 1000 * 2 * np.pi, (0.5 * (su + sv)).cpu().numpy()

    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        k_truth, spec_truth = _get_ke_spectrum(emu_data["truth"])
        k_emu, spec_emu = _get_ke_spectrum(emu_data["ds"])
        all_spectra[label] = {
            "k_truth": k_truth, "spec_truth": spec_truth,
            "k_emu": k_emu, "spec_emu": spec_emu,
            "truth_label": emu_data.get("truth_label", "Truth"),
        }

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted_truths: set[str] = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        tl = data["truth_label"]
        if tl not in plotted_truths:
            ax.loglog(data["k_truth"], data["spec_truth"].mean(0), "--",
                      label=f"{tl} (KE Anomalies)", linewidth=2, color=color, alpha=0.7)
            plotted_truths.add(tl)
        ax.loglog(data["k_emu"], data["spec_emu"].mean(0), "-",
                  label=f"{label} (KE Anomalies)", linewidth=2, color=color)

    if show_ratios:
        for target_k in target_wavenumbers:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_t = np.argmin(np.abs(data["k_truth"] - target_k))
                idx_e = np.argmin(np.abs(data["k_emu"] - target_k))
                actual_k = data["k_truth"][idx_t]
                val_t = data["spec_truth"].mean(0)[idx_t]
                val_e = data["spec_emu"].mean(0)[idx_e]
                ratio = val_e / val_t if val_t > 0 else 0
                color = emulators_dict[label]["color"]
                ax.text(actual_k * 1.1, val_e * (1.5**i),
                        f'{label}/{data["truth_label"]}={ratio:.3f}',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color))
            ax.axvline(x=actual_k, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=8))
    ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
    ax.set_ylabel("Kinetic Energy Density")

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    lon_s, lon_e, lat_s, lat_e = _format_lon_lat_strings(lon_slice, lat_slice)
    ax.set_title(
        f"Kinetic Energy Spectrum (Anomalies) at {depth_value}m\n"
        f"Region: {lon_s}–{lon_e}, {lat_s}–{lat_e}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================================
# §6  Temporal Spectrum (Scalar)
# ============================================================================

def plot_temporal_spectrum_comparison_together(
    emulators_dict: dict,
    var_to_eval: str = "thetao",
    lev_idx: int = 0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies: list[float] | None = None,
    show_ratios: bool = True,
):
    """Plot temporal spectrum — each emulator with its own truth.

    Parameters
    ----------
    emulators_dict : dict
        ``{label: {"ds": xr.Dataset, "color": str,
                   "truth": xr.Dataset, "truth_label": str}, …}``
    """
    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]

    def _get_temporal(ds_target):
        data = ds_target[var_to_eval].isel(lev=lev_idx).sel({"x": lon_slice, "y": lat_slice})
        dt_days = _infer_dt_days(ds_target)
        return _compute_temporal_psd(data.values, dt_days)

    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        freq_truth, psd_truth = _get_temporal(emu_data["truth"])
        freq_emu, psd_emu = _get_temporal(emu_data["ds"])
        all_spectra[label] = {
            "freq_truth": freq_truth, "psd_truth": psd_truth,
            "freq_emu": freq_emu, "psd_emu": psd_emu,
            "truth_label": emu_data.get("truth_label", "Truth"),
        }

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted_truths: set[str] = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        tl = data["truth_label"]
        if tl not in plotted_truths:
            ax.loglog(data["freq_truth"][1:], data["psd_truth"][1:], "--",
                      label=f"{tl} ({var_to_eval} Anomalies)", linewidth=2, color=color, alpha=0.7)
            plotted_truths.add(tl)
        ax.loglog(data["freq_emu"][1:], data["psd_emu"][1:], "-",
                  label=f"{label} ({var_to_eval} Anomalies)", linewidth=2, color=color)

    if show_ratios and target_frequencies:
        for target_f in target_frequencies:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_t = np.argmin(np.abs(data["freq_truth"] - target_f))
                idx_e = np.argmin(np.abs(data["freq_emu"] - target_f))
                actual_f = data["freq_truth"][idx_t]
                val_t = data["psd_truth"][idx_t]
                val_e = data["psd_emu"][idx_e]
                ratio = val_e / val_t if val_t > 0 else 0
                color = emulators_dict[label]["color"]
                ax.text(actual_f * 1.2, val_e * (1.5**i),
                        f'{label}/{data["truth_label"]}={ratio:.3f}',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color))
            ax.axvline(x=actual_f, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Frequency (cycles/year)")
    ax.set_ylabel("Power Spectral Density")

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    lon_s, lon_e, lat_s, lat_e = _format_lon_lat_strings(lon_slice, lat_slice)
    ax.set_title(
        f"Temporal Power Spectrum of {var_to_eval} at {depth_value}m\n"
        f"Region: {lon_s}–{lon_e}, {lat_s}–{lat_e}",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================================
# §7  Temporal KE Spectrum
# ============================================================================

def plot_ke_temporal_spectrum_comparison_together(
    emulators_dict: dict,
    var_u: str = "uo",
    var_v: str = "vo",
    lev_idx: int = 0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies: list[float] | None = None,
    show_ratios: bool = True,
):
    """Plot temporal KE spectrum — each emulator with its own truth.

    Parameters
    ----------
    emulators_dict : dict
        ``{label: {"ds": xr.Dataset, "color": str,
                   "truth": xr.Dataset, "truth_label": str}, …}``
    """
    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]

    def _get_ke_temporal(ds_target):
        u = ds_target[var_u].isel(lev=lev_idx).sel({"x": lon_slice, "y": lat_slice})
        v = ds_target[var_v].isel(lev=lev_idx).sel({"x": lon_slice, "y": lat_slice})
        dt_days = _infer_dt_days(ds_target)

        u_np = np.nan_to_num(u.values, nan=0.0)
        v_np = np.nan_to_num(v.values, nan=0.0)
        n_time = u_np.shape[0]
        spatial_shape = u_np.shape[1:]
        dt_years = dt_days / 365.25
        fs = 1.0 / dt_years

        u_det = signal.detrend(u_np, axis=0, type="linear")
        v_det = signal.detrend(v_np, axis=0, type="linear")

        steps_per_year = int(round(365.25 / dt_days))
        n_full = (n_time // steps_per_year) * steps_per_year
        if n_full >= steps_per_year:
            clim_u = u_det[:n_full].reshape(-1, steps_per_year, *spatial_shape).mean(axis=0)
            clim_v = v_det[:n_full].reshape(-1, steps_per_year, *spatial_shape).mean(axis=0)
            seasonal_u = np.tile(clim_u, (n_time // steps_per_year + 1,) + (1,) * len(spatial_shape))[:n_time]
            seasonal_v = np.tile(clim_v, (n_time // steps_per_year + 1,) + (1,) * len(spatial_shape))[:n_time]
            u_det -= seasonal_u
            v_det -= seasonal_v

        win = signal.windows.hann(n_time)
        expand = (...,) + (np.newaxis,) * len(spatial_shape)
        u_win = u_det * win[expand]
        v_win = v_det * win[expand]
        w_corr = np.mean(win**2)

        fft_u = np.fft.rfft(u_win, axis=0)
        fft_v = np.fft.rfft(v_win, axis=0)
        power_ke = 0.5 * (np.abs(fft_u)**2 + np.abs(fft_v)**2)

        psd_spatial = 2.0 * power_ke / (fs * n_time * w_corr)
        psd_spatial[0] /= 2.0
        if n_time % 2 == 0:
            psd_spatial[-1] /= 2.0

        psd = np.nanmean(psd_spatial, axis=tuple(range(1, psd_spatial.ndim)))
        freqs = np.fft.rfftfreq(n_time, d=dt_years)
        return freqs, psd

    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        freq_truth, psd_truth = _get_ke_temporal(emu_data["truth"])
        freq_emu, psd_emu = _get_ke_temporal(emu_data["ds"])
        all_spectra[label] = {
            "freq_truth": freq_truth, "psd_truth": psd_truth,
            "freq_emu": freq_emu, "psd_emu": psd_emu,
            "truth_label": emu_data.get("truth_label", "Truth"),
        }

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted_truths: set[str] = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        tl = data["truth_label"]
        if tl not in plotted_truths:
            ax.loglog(data["freq_truth"][1:], data["psd_truth"][1:], "--",
                      label=f"{tl} (KE Anomalies)", linewidth=2, color=color, alpha=0.7)
            plotted_truths.add(tl)
        ax.loglog(data["freq_emu"][1:], data["psd_emu"][1:], "-",
                  label=f"{label} (KE Anomalies)", linewidth=2, color=color)

    if show_ratios and target_frequencies:
        for target_f in target_frequencies:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_t = np.argmin(np.abs(data["freq_truth"] - target_f))
                idx_e = np.argmin(np.abs(data["freq_emu"] - target_f))
                actual_f = data["freq_truth"][idx_t]
                val_t = data["psd_truth"][idx_t]
                val_e = data["psd_emu"][idx_e]
                ratio = val_e / val_t if val_t > 0 else 0
                color = emulators_dict[label]["color"]
                ax.text(actual_f * 1.2, val_e * (1.5**i),
                        f'{label}/{data["truth_label"]}={ratio:.3f}',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color))
            ax.axvline(x=actual_f, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Frequency (cycles/year)")
    ax.set_ylabel("Kinetic Energy Power Spectral Density")

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    lon_s, lon_e, lat_s, lat_e = _format_lon_lat_strings(lon_slice, lat_slice)
    ax.set_title(
        f"Temporal KE Power Spectrum at {depth_value}m\n"
        f"Region: {lon_s}–{lon_e}, {lat_s}–{lat_e}",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
