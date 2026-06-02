import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import xarray as xr

# ── Global font size defaults ──
_FONT_SIZE = 16
_SUPTITLE_SIZE = 20
plt.rcParams.update(
    {
        "font.size": _FONT_SIZE,
        "axes.titlesize": _FONT_SIZE,
        "axes.labelsize": _FONT_SIZE,
        "xtick.labelsize": _FONT_SIZE,
        "ytick.labelsize": _FONT_SIZE,
        "legend.fontsize": _FONT_SIZE,
        "figure.titlesize": _SUPTITLE_SIZE,
    }
)


def _fix_lognorm_ticks(cb, vmin, vmax, min_ticks=3):
    """Ensure a LogNorm colorbar has at least `min_ticks` ticks.

    When the data range spans less than ~2 orders of magnitude, matplotlib's
    default LogLocator may only place 2 ticks.  This helper adds geometrically
    spaced intermediate ticks so the reader can always orient themselves.
    """
    existing = [t for t in cb.get_ticks() if vmin <= t <= vmax]
    if len(existing) >= min_ticks:
        return
    ticks = np.geomspace(vmin, vmax, min_ticks)
    cb.set_ticks(ticks)
    cb.ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())


def _scale_suffix(emulators_dict):
    """Previously returned ' (multi-resolution)' for multi-scale runs; now disabled."""
    return ""


# 5. Variable Mappings
OCEAN_FIELD_NAME_PREFIXES = {
    "thetao": ["thetao_"],
    "so": ["so_"],
    "uo": ["uo_"],
    "vo": ["vo_"],
    "zos": ["zos"],
    "sea_surface_temperature": ["sst"],
    "lev": ["idepth_"],
}

# 6. Plotting Constants
BASINS_TO_PLOT = ["Atlantic", "Pacific", "Indian", "Southern"]

DEPTH_SLICES = [
    {"min": 0, "max": 700, "title": "Surface (0-700m)"},
    {"min": 700, "max": 2000, "title": "Intermediate (700-2000m)"},
    {"min": 2000, "max": 7000, "title": "Deep (2000-7000m)"},
]
VAR_CONFIG = {
    "thetao": {
        "cmap": cmocean.cm.thermal,
        "units": "$^\\circ$C",
        "label": "Temperature",
        "centered": False,
    },
    "so": {
        "cmap": cmocean.cm.haline,
        "units": "psu",
        "label": "Salinity",
        "centered": False,
    },
    "ohc": {
        "cmap": cmocean.cm.thermal,
        "units": "J",
        "label": "Total Heat Content",
        "centered": False,
    },
    "zos": {"cmap": cmocean.cm.balance, "units": "m", "label": "SSH", "centered": True},
    "uo": {
        "cmap": cmocean.cm.balance,
        "units": "m/s",
        "label": "U-Velocity",
        "centered": True,
    },
    "vo": {
        "cmap": cmocean.cm.balance,
        "units": "m/s",
        "label": "V-Velocity",
        "centered": True,
    },
}


def _display_name(variable_name):
    """Map CMIP variable names to human-readable labels for titles/legends."""
    return VAR_CONFIG.get(variable_name, {}).get("label", variable_name)


def reorder_legend_paired(handles, labels):
    """Reorder legend: truths in first row, emulators in second row, ncol=n_pairs.

    Matplotlib legend with ncol fills column-major, so to get:
        Row 1: T1, T2, T3
        Row 2: E1, E2, E3
    we must interleave: [T1, E1, T2, E2, T3, E3] with ncol=n_pairs.
    """
    truth_h, truth_l, emu_h, emu_l = [], [], [], []
    for h, l in zip(handles, labels):
        if l.startswith("OM4"):
            truth_h.append(h)
            truth_l.append(l)
        else:
            emu_h.append(h)
            emu_l.append(l)
    n_pairs = max(len(truth_h), len(emu_h))
    # Interleave: T1, E1, T2, E2, ... so column-major layout gives rows
    out_h, out_l = [], []
    for i in range(n_pairs):
        if i < len(truth_h):
            out_h.append(truth_h[i])
            out_l.append(truth_l[i])
        if i < len(emu_h):
            out_h.append(emu_h[i])
            out_l.append(emu_l[i])
    return out_h, out_l, n_pairs


# Constants
RHO = 1035  # Reference density (kg/m^3)
CP = 3850  # Specific heat capacity (J/kg/K)
SCALE_FACTOR = 1e21  # Scale to ZJ (10^21 J)

DEPTH_LAYERS = [
    {"title": "2.5m", "index": 0},
    {"title": "700m", "index": 10},
    {"title": "2000m", "index": 14},
    # {'title':'3000m','index':15}
]

SPECTRUM_REGIONS = [
    ("Gulf Stream", slice(300, 320), slice(25, 45)),
    ("Kuroshio", slice(150, 170), slice(25, 45)),
    ("Agulhas", slice(40, 60), slice(-50, -30)),
    ("Malvinas", slice(311, 331), slice(-51, -31)),
    ("Niño 3.4", slice(190, 240), slice(-5, 5)),
    ("Tropical Pacific", slice(130, 290), slice(-30, 30)),
]


def concat_levels(
    ds: xr.Dataset, prefix: str, xg_name: str | None = None, new_coord: None = None
) -> xr.DataArray:
    """Concatenates the levels of a 3D variable into a single DataArray."""
    levels = []
    names = []
    level_pattern = re.compile(rf"{prefix}_(\d+)$")
    for name in ds.data_vars:
        match = level_pattern.search(name)
        if match:
            levels.append(int(match.group(1)))
            names.append(name)
    names = sorted(names, key=lambda name: levels[names.index(name)])
    da = xr.concat([ds[name] for name in names], dim="lev")
    da.name = prefix
    if xg_name is not None:
        da.attrs["xg_name"] = xg_name
    if new_coord is not None:
        new_coord_name, new_coord = new_coord
        da[new_coord_name] = new_coord
        da = da.swap_dims({"lev": new_coord_name})
    return da


def extract_basin_masks(ds_with_basin_var):
    """Extracts boolean masks from the integer 'basin' variable."""
    basin_code = ds_with_basin_var["basin"]
    return {
        "Atlantic": xr.where(basin_code == 2, 1.0, np.nan),
        "Pacific": xr.where(basin_code == 3, 1.0, np.nan),
        "Indian": xr.where(basin_code == 5, 1.0, np.nan),
        "Southern": xr.where(basin_code == 1, 1.0, np.nan),
    }


def _detrend_linear_torch(data):
    """
    Removes a linear plane of best fit from 4D (B, C, H, W) data.
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
    data,
    dx=1.0,
    dy=1.0,
    num_bins=None,
    n_factor=4,
    remove_mean=True,
    detrend=None,
    window="Hann",
    truncate=True,
    cutoff_before_bins: bool = True,
):
    """
    Computes the isotropic 1D power spectrum from 2D (H,W), 3D (B,H,W),
    or 4D (B,C,H,W) data. Matches `xrft.isotropic_power_spectrum(scaling="density")`.

    The output spectrum is computed for each batch and channel element.

    Parameters:
    ----------
    data : torch.Tensor
        Input data tensor. Can be 2D, 3D, or 4D.
    dx : float, optional
        Grid spacing in the x-dimension.
    dy : float, optional
        Grid spacing in the y-dimension.
    num_bins : int, optional
        Number of bins. If None, defaults to min(H, W) // n_factor.
    n_factor : int, optional
        Factor to determine number of bins.
    remove_mean : bool, optional
        If True, removes spatial mean. Overridden by `detrend`.
    detrend : str, optional
        'linear' or 'constant'.
    window : str, optional
        'hann' or 'Hann'.
    truncate : bool, optional
        If True, truncates spectrum at the smallest Nyquist frequency.
    cutoff_before_bins: bool, optional
        If True, truncates the spectrum after already computing the bin locations. Matches xrft
    Returns:
    -------
    k_bins_centers : torch.Tensor
        1D tensor of bin center wavenumbers. Shape: (num_bins,)
    iso_spectrum : torch.Tensor
        1D tensor of the (k * P(k)) spectrum.
        Shape: (B, C, num_bins), (B, num_bins), or (num_bins,)
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

    fft_2d = torch.fft.rfft2(data, norm="forward")

    power_2d = torch.abs(fft_2d) ** 2
    power_2d = power_2d / window_correction

    psd_2d = power_2d * (Lx * Ly)

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


def rmse_spatial(truth_map, pred_map, area):
    """Area-weighted RMSE for 2D spatial maps."""
    sq_diff = (truth_map - pred_map) ** 2

    # Mask weights where data is missing
    weights = area.where(sq_diff.notnull())

    numerator = (sq_diff * weights).sum(["y", "x"])
    denominator = weights.sum(["y", "x"])

    return np.sqrt(numerator / denominator).values.item()


def calculate_ohc_2d(ds):
    """
    Calculates Ocean Heat Content (J/m^2) by integrating thetao over depth.
    Formula: Integral(rho * cp * thetao * dz)
    """
    # Ensure dz is present and correct shape
    # Sum over 'lev' dimension
    ohc = ds["thetao"] * RHO * CP * ds["dz"]
    ohc.attrs["units"] = "J/m^2"
    return ohc


def calculate_zonal_mean_2d(field_2d, mask, dx, wet_2d):
    """
    Zonal Mean for a 2D field (e.g. OHC).
    """
    numerator = (field_2d * mask * dx).sum("x")
    denominator = (mask * wet_2d * dx).sum("x")
    return numerator / denominator


def calculate_vertical_mean_simple(field, dz, wet_mask, min_depth, max_depth):
    """
    Calculates the vertical mean map using Truth's geometry.
    Formula: Sum(Field * dz * wet) / Sum(dz * wet) over depth slice.
    """
    # Select depth slices
    f_slice = field.sel(lev=slice(min_depth, max_depth))
    dz_slice = dz.sel(lev=slice(min_depth, max_depth))
    wet_slice = wet_mask.sel(lev=slice(min_depth, max_depth))

    numerator = (f_slice * dz_slice * wet_slice).sum("lev")
    denominator = (dz_slice * wet_slice).sum("lev")

    return numerator / denominator.where(denominator != 0)


def calculate_ohc_map(dataset, dz, wet_mask, min_depth, max_depth):
    """
    Calculates the Time-Mean OHC map (Vertically Integrated).
    Formula: Sum(rho * cp * thetao * dz) / Scale
    Units: GJ/m^2
    """
    # Select slices
    thetao_slice = dataset["thetao"].sel(lev=slice(min_depth, max_depth))
    dz_slice = dz.sel(lev=slice(min_depth, max_depth))
    wet_slice = wet_mask.sel(lev=slice(min_depth, max_depth))

    # 1. Integrate Vertically (Sum over lev)
    # Result is (time, y, x)
    # ohc_integrated = (RHO * CP * thetao_slice * dz_slice *dataset['areacello']* wet_slice).sum('lev')

    ohc_integrated = (RHO * CP * thetao_slice * dz_slice * wet_slice).sum("lev")

    # 2. Time Average
    ohc_mean = ohc_integrated.mean("time")

    # 3. Scale
    return ohc_mean / SCALE_FACTOR


def calculate_temperature_map_layer(
    dataset, var_name, dz, wet_mask, depth_index, time_idx=None
):
    """
    Computes a 2D map for a specific variable at a specific time index.
    - 3D vars: Computes a specific depth.
    - 2D vars: Returns surface field masked by Truth.
    """
    # Select Time Step
    # 2. Time Average
    ds_t = dataset.mean(dim="time")
    # ds_t = dataset.isel(time=time_idx)
    field = ds_t[var_name]

    # Select Depth Slice
    f_slice = field.isel(lev=depth_index)
    dz_slice = dz.isel(lev=depth_index)
    wet_slice = wet_mask.isel(lev=depth_index)

    # Depth Average
    numerator = f_slice * dz_slice * wet_slice
    denominator = dz_slice * wet_slice
    result = numerator / denominator

    return result.compute()


def calculate_zonal_mean(field_mean, mask, dx, wet):
    """
    Simple Zonal Mean using your exact method.
    Numerator: Sum(Field * Mask * dx)
    Denominator: Sum(Mask * Wet_Truth * dx)
    """
    numerator = (field_mean * mask * dx).sum("x")
    denominator = (mask * wet * dx).sum("x")
    return numerator / denominator


def calculate_snapshot_map_SpecificLayer(
    dataset, var_name, dz, wet_mask, depth_index, time_idx
):
    """
    Computes a 2D map for a specific variable at a specific time index.
    - 3D vars: Computes depth-weighted average.
    - 2D vars: Returns surface field masked by Truth.
    """
    # Select Time Step
    ds_t = dataset.isel(time=time_idx)
    field = ds_t[var_name]

    # Select Depth Slice
    f_slice = field.isel(lev=depth_index)
    dz_slice = dz.isel(lev=depth_index)
    wet_slice = wet_mask.isel(lev=depth_index)

    # Depth Average
    numerator = f_slice * dz_slice * wet_slice
    denominator = dz_slice * wet_slice
    result = numerator / denominator

    return result.compute()


def calculate_ohc_snapshot_map(dataset, dz, wet_mask, min_depth, max_depth, time_index):
    """
    Calculates the Time-Mean OHC map (Vertically Integrated).
    Formula: Sum(rho * cp * thetao * dz) / Scale
    Units: GJ/m^2
    """
    # Select slices
    thetao_slice = dataset["thetao"].sel(lev=slice(min_depth, max_depth))
    dz_slice = dz.sel(lev=slice(min_depth, max_depth))
    wet_slice = wet_mask.sel(lev=slice(min_depth, max_depth))

    # 1. Integrate Vertically (Sum over lev)
    # Result is (time, y, x)
    # ohc_integrated = (RHO * CP * thetao_slice * dz_slice *dataset['areacello']* wet_slice).sum('lev')

    ohc_integrated = (RHO * CP * thetao_slice * dz_slice * wet_slice).sum("lev")

    # 2. Time snapshot
    ohc_mean = ohc_integrated.isel(time=time_index)

    # 3. Scale
    return ohc_mean / SCALE_FACTOR


def process_dataset(ds_input, truth_levs, dz_arr, grid_metrics_dict):
    """Encapsulates the logic to turn raw rollout/truth into a stacked dataset with grid metrics."""
    # Initialize basic coordinate structure
    ds_stacked = xr.Dataset(
        coords={
            "time": (["time"], ds_input.time.values),
            "y": (["y"], ds_input.y.values),
            "x": (["x"], ds_input.x.values),
            "lev": (["lev"], truth_levs),
            "dz": (["lev"], dz_arr),
        }
    )

    # Process 3D vars
    for var in ["uo", "vo", "thetao", "so"]:
        if var in OCEAN_FIELD_NAME_PREFIXES:
            prefix = OCEAN_FIELD_NAME_PREFIXES[var][0][:-1]
            try:
                data_processed = (
                    concat_levels(ds_input, prefix, var)
                    .squeeze()
                    .transpose("time", "y", "x", "lev")
                    .data
                )
                ds_stacked[var] = (["time", "y", "x", "lev"], data_processed)
            except Exception:
                pass  # Variable might not exist

    # Process 2D vars
    if "zos" in ds_input:
        ds_stacked["zos"] = (["time", "y", "x"], ds_input["zos"].squeeze().data)

    # Apply Dummy Mask to dz if velocity exists to ensure consistent masking
    if "vo" in ds_stacked:
        ds_stacked["dz"] = (
            ds_stacked["dz"] * (ds_stacked["vo"].isel(time=0) * 0 + 1)
        ).compute()

    # Add Grid Metrics
    for metric, data in grid_metrics_dict.items():
        ds_stacked[metric] = data
        ds_stacked = ds_stacked.set_coords(metric)

    return ds_stacked

    # save_path = os.path.join(output_dir, f"Spatial_Bias_{label.replace(' ', '_')}_{variable_name}.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")

    # save_path = os.path.join(output_dir, f"Snapshots_{var_name}_{depth_config['min']}_{depth_config['max']}.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")

    # Save Figure
    # save_path = os.path.join(output_dir, f"TimeSeries_Comparison_{var_label}.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")

    # if output_dir:
    #     import os
    #     save_path = os.path.join(output_dir, f"Variance_Comparison_{variable_name}.png")
    #     plt.savefig(save_path, bbox_inches='tight')
    #     print(f"Saved: {save_path}")
    #     plt.close(fig)
    # else:
    #     plt.show()

    # save_path = os.path.join(output_dir, f"Zonal_Bias_{label.replace(' ', '_')}_{variable_name}.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Saved: {save_path}")


# =============================================================================
# "_together" versions: Each emulator entry contains 'truth' and 'truth_label'
# emulators_dict format: {label: {'ds': ds, 'color': color, 'truth': ds_truth, 'truth_label': label}}
# =============================================================================


def plot_mean_timeseries_comparison_together(
    emulators_dict, variable_name, var_units, var_label, output_dir, overlay=False
):
    """
    Time series comparison where each emulator uses its corresponding truth.
    Multiscale: rows=depth slices, cols=scales (one truth+emulator pair per column).
    Single truth: rows=depth slices, cols=emulators (or overlay=True for all on one panel).
    """
    print(f"Generating Time Series for {var_label}...")
    num_depths = len(DEPTH_SLICES)

    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)
    multiscale = len(unique_truth_labels) > 1

    def _compute_ts(ds, variable_name, depth_slice, area, wet_mask):
        field = ds[variable_name].sel(lev=depth_slice)
        dz = ds["dz"].sel(lev=depth_slice)
        vol_weight = dz * area
        ts = (field * vol_weight).sum(["x", "y", "lev"]).compute() / (
            vol_weight * wet_mask.sel(lev=depth_slice)
        ).sum(["x", "y", "lev"]).compute()
        return ts

    def _to_years(time_vals):
        return np.array([t.year + (t.dayofyr - 1) / 365.25 for t in time_vals])

    if multiscale:
        # Rows: depth slices, Cols: one per emulator (each with its own truth)
        ncols = len(emulators_dict)
        fig, axes = plt.subplots(
            num_depths,
            ncols,
            figsize=(5 * ncols, 3.5 * num_depths),
            constrained_layout=True,
            sharey="row",
        )
        if num_depths == 1:
            axes = axes.reshape(1, -1)
        if ncols == 1:
            axes = axes.reshape(-1, 1)

        for col_j, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            area = ds_truth["areacello"]
            wet_mask = xr.where(
                np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0
            )

            for row_i, Slices in enumerate(DEPTH_SLICES):
                ax = axes[row_i, col_j]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(axis="both", which="both", direction="out", length=4)
                depth_slice = slice(Slices["min"], Slices["max"])

                ts_truth = _compute_ts(
                    ds_truth, variable_name, depth_slice, area, wet_mask
                )
                ts_emu = _compute_ts(ds_emu, variable_name, depth_slice, area, wet_mask)
                time_truth = _to_years(ts_truth.time.values)
                time_emu = _to_years(ts_emu.time.values)

                ax.plot(
                    time_truth,
                    ts_truth.values,
                    label=truth_label,
                    color="black",
                    lw=2,
                    linestyle="--",
                    alpha=0.7,
                )
                ax.plot(time_emu, ts_emu.values, label=label, color=color, lw=2)

                min_len = min(len(ts_truth), len(ts_emu))
                y_true = ts_truth.values[:min_len]
                y_pred = ts_emu.values[:min_len]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1.0 - ss_res / ss_tot

                if row_i == 0:
                    ax.set_title(f"{label}", fontweight="bold")
                ax.set_xlabel("Year")
                if col_j == 0:
                    ax.set_ylabel(f"{Slices['title']}\n{var_label} [{var_units}]")
                else:
                    ax.set_ylabel("")
                ax.text(
                    0.02,
                    0.95,
                    f"R²={r2:.2f}",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # Shared legend per column (bottom of each column, two rows)
        for col_j in range(ncols):
            handles, labels = axes[0, col_j].get_legend_handles_labels()
            axes[num_depths - 1, col_j].legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.70),
                ncol=1,
                frameon=False,
            )

        fig.suptitle(
            f"Global Mean {var_label} Time Series{_scale_suffix(emulators_dict)}",
            y=1.05,
        )
    elif overlay:
        # Single truth, overlay: all emulators on one panel per depth slice
        fig, axes = plt.subplots(num_depths, 1, figsize=(12, 4 * num_depths))
        if num_depths == 1:
            axes = [axes]

        truth_plotted = False
        stats_text_per_depth = {i: [] for i in range(num_depths)}

        for label, emu_data in emulators_dict.items():
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            area = ds_truth["areacello"]
            wet_mask = xr.where(
                np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0
            )

            for row_i, Slices in enumerate(DEPTH_SLICES):
                ax = axes[row_i]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(axis="both", which="both", direction="out", length=4)
                depth_slice = slice(Slices["min"], Slices["max"])

                ts_truth = _compute_ts(
                    ds_truth, variable_name, depth_slice, area, wet_mask
                )
                ts_emu = _compute_ts(ds_emu, variable_name, depth_slice, area, wet_mask)
                time_truth = _to_years(ts_truth.time.values)
                time_emu = _to_years(ts_emu.time.values)

                if not truth_plotted:
                    ax.plot(
                        time_truth,
                        ts_truth.values,
                        label=truth_label,
                        color="black",
                        lw=2,
                        linestyle="--",
                        alpha=0.7,
                    )
                ax.plot(time_emu, ts_emu.values, label=label, color=color, lw=2)

                min_len = min(len(ts_truth), len(ts_emu))
                y_true = ts_truth.values[:min_len]
                y_pred = ts_emu.values[:min_len]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1.0 - ss_res / ss_tot
                stats_text_per_depth[row_i].append(f"{label} R²={r2:.2f}")

                ax.set_xlabel("Year")
                ax.set_ylabel(f"{Slices['title']}\n{var_label} [{var_units}]")

            truth_plotted = True

        for row_i in range(num_depths):
            ax = axes[row_i]
            stats_block = ",  ".join(stats_text_per_depth[row_i])
            ax.text(
                0.02,
                0.95,
                stats_block,
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        all_handles, all_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=len(all_labels),
            frameon=False,
        )
        fig.suptitle(
            f"Global Mean {var_label} Time Series{_scale_suffix(emulators_dict)}",
            y=1.02,
        )
    else:
        # Single truth: grid layout (rows=depths, cols=emulators)
        ncols = len(emulators_dict)
        fig, axes = plt.subplots(
            num_depths,
            ncols,
            figsize=(5 * ncols, 3.5 * num_depths),
            constrained_layout=True,
            sharey="row",
        )
        if num_depths == 1:
            axes = axes.reshape(1, -1)
        if ncols == 1:
            axes = axes.reshape(-1, 1)

        for col_j, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            area = ds_truth["areacello"]
            wet_mask = xr.where(
                np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0
            )

            for row_i, Slices in enumerate(DEPTH_SLICES):
                ax = axes[row_i, col_j]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(axis="both", which="both", direction="out", length=4)
                depth_slice = slice(Slices["min"], Slices["max"])

                ts_truth = _compute_ts(
                    ds_truth, variable_name, depth_slice, area, wet_mask
                )
                ts_emu = _compute_ts(ds_emu, variable_name, depth_slice, area, wet_mask)
                time_truth = _to_years(ts_truth.time.values)
                time_emu = _to_years(ts_emu.time.values)

                ax.plot(
                    time_truth,
                    ts_truth.values,
                    label=truth_label,
                    color="black",
                    lw=2,
                    linestyle="--",
                    alpha=0.7,
                )
                ax.plot(time_emu, ts_emu.values, label=label, color=color, lw=2)

                min_len = min(len(ts_truth), len(ts_emu))
                y_true = ts_truth.values[:min_len]
                y_pred = ts_emu.values[:min_len]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1.0 - ss_res / ss_tot

                if row_i == 0:
                    ax.set_title(f"{label}", fontweight="bold")
                ax.set_xlabel("Year")
                if col_j == 0:
                    ax.set_ylabel(f"{Slices['title']}\n{var_label} [{var_units}]")
                else:
                    ax.set_ylabel("")
                ax.text(
                    0.02,
                    0.95,
                    f"R²={r2:.2f}",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # Shared legend for all columns — collect from all columns to include all emulators
        all_handles, all_labels = [], []
        seen = set()
        for col_j in range(ncols):
            h, l = axes[0, col_j].get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in seen:
                    all_handles.append(hi)
                    all_labels.append(li)
                    seen.add(li)
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(all_labels),
            frameon=False,
        )
        fig.suptitle(
            f"Global Mean {var_label} Time Series{_scale_suffix(emulators_dict)}",
            y=1.05,
        )
    plt.show()


def _detrend_1d(ts):
    """Remove linear trend from a 1-D numpy array. Returns detrended array."""
    n = len(ts)
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    ts_mean = ts.mean()
    slope = np.sum((t - t_mean) * (ts - ts_mean)) / np.sum((t - t_mean) ** 2)
    return ts - (slope * (t - t_mean) + ts_mean)


def plot_mean_timeseries_detrended_comparison_together(
    emulators_dict, variable_name, var_units, var_label, output_dir, overlay=False
):
    """
    Detrended time series comparison.
    Multiscale: rows=depth slices, cols=scales (one truth+emulator pair per column).
    Single truth: rows=depth slices, cols=emulators (or overlay=True for all on one panel).
    """
    print(f"Generating Detrended Time Series for {var_label}...")
    num_depths = len(DEPTH_SLICES)

    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)
    multiscale = len(unique_truth_labels) > 1

    def _compute_ts(ds, variable_name, depth_slice, area, wet_mask):
        field = ds[variable_name].sel(lev=depth_slice)
        dz = ds["dz"].sel(lev=depth_slice)
        vol_weight = dz * area
        ts = (field * vol_weight).sum(["x", "y", "lev"]).compute() / (
            vol_weight * wet_mask.sel(lev=depth_slice)
        ).sum(["x", "y", "lev"]).compute()
        return ts

    def _to_years(time_vals):
        return np.array([t.year + (t.dayofyr - 1) / 365.25 for t in time_vals])

    if multiscale:
        ncols = len(emulators_dict)
        fig, axes = plt.subplots(
            num_depths,
            ncols,
            figsize=(5 * ncols, 3.5 * num_depths),
            constrained_layout=True,
            sharey="row",
        )
        if num_depths == 1:
            axes = axes.reshape(1, -1)
        if ncols == 1:
            axes = axes.reshape(-1, 1)

        for col_j, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            area = ds_truth["areacello"]
            wet_mask = xr.where(
                np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0
            )

            for row_i, Slices in enumerate(DEPTH_SLICES):
                ax = axes[row_i, col_j]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(axis="both", which="both", direction="out", length=4)
                depth_slice = slice(Slices["min"], Slices["max"])

                ts_truth = _compute_ts(
                    ds_truth, variable_name, depth_slice, area, wet_mask
                )
                ts_emu = _compute_ts(ds_emu, variable_name, depth_slice, area, wet_mask)
                ts_truth_dt = _detrend_1d(ts_truth.values)
                ts_emu_dt = _detrend_1d(ts_emu.values)
                time_truth = _to_years(ts_truth.time.values)
                time_emu = _to_years(ts_emu.time.values)

                ax.plot(
                    time_truth,
                    ts_truth_dt,
                    label=truth_label,
                    color="black",
                    lw=2,
                    linestyle="--",
                    alpha=0.7,
                )
                ax.plot(time_emu, ts_emu_dt, label=label, color=color, lw=2)

                min_len = min(len(ts_truth_dt), len(ts_emu_dt))
                y_true = ts_truth_dt[:min_len]
                y_pred = ts_emu_dt[:min_len]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1.0 - ss_res / ss_tot

                if row_i == 0:
                    ax.set_title(f"{label}", fontweight="bold")
                ax.set_xlabel("Year")
                if col_j == 0:
                    ax.set_ylabel(f"{Slices['title']}\n{var_label} [{var_units}]")
                else:
                    ax.set_ylabel("")
                ax.text(
                    0.02,
                    0.95,
                    f"R²={r2:.2f}",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # Shared legend per column (bottom of each column)
        for col_j in range(ncols):
            handles, labels = axes[0, col_j].get_legend_handles_labels()
            axes[num_depths - 1, col_j].legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.70),
                ncol=1,
                frameon=False,
            )

        fig.suptitle(
            f"Detrended Global Mean {var_label} Time Series{_scale_suffix(emulators_dict)}",
            y=1.05,
        )
    elif overlay:
        # Single truth, overlay: all emulators on one panel per depth slice
        fig, axes = plt.subplots(num_depths, 1, figsize=(12, 4 * num_depths))
        if num_depths == 1:
            axes = [axes]

        truth_plotted = False
        stats_text_per_depth = {i: [] for i in range(num_depths)}

        for label, emu_data in emulators_dict.items():
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            area = ds_truth["areacello"]
            wet_mask = xr.where(
                np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0
            )

            for row_i, Slices in enumerate(DEPTH_SLICES):
                ax = axes[row_i]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(axis="both", which="both", direction="out", length=4)
                depth_slice = slice(Slices["min"], Slices["max"])

                ts_truth = _compute_ts(
                    ds_truth, variable_name, depth_slice, area, wet_mask
                )
                ts_emu = _compute_ts(ds_emu, variable_name, depth_slice, area, wet_mask)
                ts_truth_dt = _detrend_1d(ts_truth.values)
                ts_emu_dt = _detrend_1d(ts_emu.values)
                time_truth = _to_years(ts_truth.time.values)
                time_emu = _to_years(ts_emu.time.values)

                if not truth_plotted:
                    ax.plot(
                        time_truth,
                        ts_truth_dt,
                        label=truth_label,
                        color="black",
                        lw=2,
                        linestyle="--",
                        alpha=0.7,
                    )
                ax.plot(time_emu, ts_emu_dt, label=label, color=color, lw=2)

                min_len = min(len(ts_truth_dt), len(ts_emu_dt))
                y_true = ts_truth_dt[:min_len]
                y_pred = ts_emu_dt[:min_len]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1.0 - ss_res / ss_tot
                stats_text_per_depth[row_i].append(f"{label} R²={r2:.2f}")

                ax.set_xlabel("Year")
                ax.set_ylabel(f"{Slices['title']}\n{var_label} [{var_units}]")

            truth_plotted = True

        for row_i in range(num_depths):
            ax = axes[row_i]
            stats_block = ",  ".join(stats_text_per_depth[row_i])
            ax.text(
                0.02,
                0.95,
                stats_block,
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        all_handles, all_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=len(all_labels),
            frameon=False,
        )
        fig.suptitle(
            f"Detrended Global Mean {var_label} Time Series{_scale_suffix(emulators_dict)}",
            y=1.02,
        )
    else:
        # Single truth: grid layout (rows=depths, cols=emulators)
        ncols = len(emulators_dict)
        fig, axes = plt.subplots(
            num_depths,
            ncols,
            figsize=(5 * ncols, 3.5 * num_depths),
            constrained_layout=True,
            sharey="row",
        )
        if num_depths == 1:
            axes = axes.reshape(1, -1)
        if ncols == 1:
            axes = axes.reshape(-1, 1)

        for col_j, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            area = ds_truth["areacello"]
            wet_mask = xr.where(
                np.isnan(ds_truth[variable_name].isel(time=0)), 0.0, 1.0
            )

            for row_i, Slices in enumerate(DEPTH_SLICES):
                ax = axes[row_i, col_j]
                ax.set_facecolor("white")
                ax.grid(False)
                ax.tick_params(axis="both", which="both", direction="out", length=4)
                depth_slice = slice(Slices["min"], Slices["max"])

                ts_truth = _compute_ts(
                    ds_truth, variable_name, depth_slice, area, wet_mask
                )
                ts_emu = _compute_ts(ds_emu, variable_name, depth_slice, area, wet_mask)
                ts_truth_dt = _detrend_1d(ts_truth.values)
                ts_emu_dt = _detrend_1d(ts_emu.values)
                time_truth = _to_years(ts_truth.time.values)
                time_emu = _to_years(ts_emu.time.values)

                ax.plot(
                    time_truth,
                    ts_truth_dt,
                    label=truth_label,
                    color="black",
                    lw=2,
                    linestyle="--",
                    alpha=0.7,
                )
                ax.plot(time_emu, ts_emu_dt, label=label, color=color, lw=2)

                min_len = min(len(ts_truth_dt), len(ts_emu_dt))
                y_true = ts_truth_dt[:min_len]
                y_pred = ts_emu_dt[:min_len]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1.0 - ss_res / ss_tot

                if row_i == 0:
                    ax.set_title(f"{label}", fontweight="bold")
                ax.set_xlabel("Year")
                if col_j == 0:
                    ax.set_ylabel(f"{Slices['title']}\n{var_label} [{var_units}]")
                else:
                    ax.set_ylabel("")
                ax.text(
                    0.02,
                    0.95,
                    f"R²={r2:.2f}",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # Shared legend for all columns — collect from all columns to include all emulators
        all_handles, all_labels = [], []
        seen = set()
        for col_j in range(ncols):
            h, l = axes[0, col_j].get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in seen:
                    all_handles.append(hi)
                    all_labels.append(li)
                    seen.add(li)
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(all_labels),
            frameon=False,
        )
        fig.suptitle(
            f"Detrended Global Mean {var_label} Time Series{_scale_suffix(emulators_dict)}",
            y=1.05,
        )
    plt.show()


def plot_ke_variance_comparison_together(
    emulators_dict,
    var_u="uo",
    var_v="vo",
    units="m$^2$/s$^2$",
    output_dir=None,
    colorbar_fix=False,
    n_levels=None,
):
    """EKE comparison where each emulator uses its corresponding truth.
    Shows one row per unique ground truth, then one row per emulator.
    When n_levels is set, uses a discrete colorbar with that many levels.
    """
    print(f"\nGenerating EKE Comparison...")

    datasets = list(emulators_dict.keys())
    ncols = len(DEPTH_SLICES)

    def compute_eke(ds, var_u_name, var_v_name):
        u_prime = ds[var_u_name] - ds[var_u_name].mean("time")
        v_prime = ds[var_v_name] - ds[var_v_name].mean("time")
        eke = 0.5 * (u_prime**2 + v_prime**2)
        return eke.mean("time")

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_label = list(emulators_dict.keys())[0]
    ref_data = emulators_dict[ref_label]
    ref_tl = ref_data.get("truth_label", "Truth")

    n_truths = 1
    nrows = n_truths + len(datasets)

    # Compute ground truth EKE
    plot_data_truth = {}
    ds_truth = ref_data["truth"]
    dz_truth = ds_truth["dz"]
    wet_mask = (ds_truth[var_u].isel(time=0) * 0 + 1).compute()
    truth_eke_3d = compute_eke(ds_truth, var_u, var_v)
    plot_data_truth[ref_tl] = {}
    for Slices in DEPTH_SLICES:
        plot_data_truth[ref_tl][Slices["title"]] = calculate_vertical_mean_simple(
            truth_eke_3d, dz_truth, wet_mask, Slices["min"], Slices["max"]
        ).compute()

    # Emulator EKE
    plot_data = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        dz_truth = ds_truth["dz"]
        wet_mask = (ds_truth[var_u].isel(time=0) * 0 + 1).compute()
        emu_eke_3d = compute_eke(emu_data["ds"], var_u, var_v)

        plot_data[label] = {}
        for Slices in DEPTH_SLICES:
            plot_data[label][Slices["title"]] = calculate_vertical_mean_simple(
                emu_eke_3d, dz_truth, wet_mask, Slices["min"], Slices["max"]
            ).compute()

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(4 * ncols, 2.0 * nrows + 0.8))
    gs = GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        height_ratios=[1] * nrows + [0.04],
        hspace=0.02,
        wspace=0.05,
        top=0.93,
        bottom=0.03,
        left=0.12,
        right=0.97,
    )

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
    cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
    for cax in cbar_axes:
        pos = cax.get_position()
        new_width = pos.width * 0.7
        offset = (pos.width - new_width) / 2
        cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices["title"]
        all_data = [plot_data_truth[ref_tl][key].values.flatten()]
        all_data += [plot_data[ds][key].values.flatten() for ds in datasets]
        flat_data = np.concatenate(all_data)
        flat_data = flat_data[~np.isnan(flat_data)]
        flat_pos = flat_data[flat_data > 0]
        vmin = np.percentile(flat_pos, 5) if len(flat_pos) > 0 else 1e-10
        vmax = np.percentile(flat_pos, 99) if len(flat_pos) > 0 else 1.0

        if n_levels is not None:
            levels = np.geomspace(vmin, vmax, n_levels + 1)
            cmap_use = cmocean.cm.thermal.resampled(n_levels)
            norm = colors.BoundaryNorm(levels, ncolors=cmap_use.N, clip=False)
        else:
            cmap_use = cmocean.cm.thermal
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        # Truth row
        ax = axes[0, j]
        p_mesh = plot_data_truth[ref_tl][key].plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="x",
            y="y",
            cmap=cmap_use,
            norm=norm,
            add_colorbar=False,
            rasterized=True,
        )
        ax.add_feature(
            cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
        )
        ax.coastlines(zorder=11)
        row_label = ref_tl.replace(" (", "\n(")
        if j == 0:
            ax.text(
                -0.07,
                0.55,
                row_label,
                va="center",
                ha="right",
                rotation=90,
                transform=ax.transAxes,
                fontweight="bold",
                multialignment="center",
            )
        ax.set_title(key, fontweight="bold")

        # Emulator rows
        truth_vals = plot_data_truth[ref_tl][key].values.flatten()
        for i, ds_name in enumerate(datasets):
            ax = axes[n_truths + i, j]
            p_mesh = plot_data[ds_name][key].plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmap_use,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)
            row_label = ds_name.replace(" (", "\n(")
            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    row_label,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                    multialignment="center",
                )
            ax.set_title("")

            # Global EKE pattern correlation & RMSE vs truth
            emu_vals = plot_data[ds_name][key].values.flatten()
            valid = ~(np.isnan(emu_vals) | np.isnan(truth_vals))
            if valid.sum() > 1:
                corr = np.corrcoef(emu_vals[valid], truth_vals[valid])[0, 1]
                rmse = np.sqrt(np.mean((emu_vals[valid] - truth_vals[valid]) ** 2))
                ax.text(
                    0.02,
                    0.95,
                    f"Corr: {corr:.3f}\nRMSE: {rmse:.2e}",
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        zorder=20,
                    ),
                    verticalalignment="top",
                    linespacing=1.2,
                    zorder=20,
                )

        _ext = "both" if n_levels is not None else "neither"
        cb = fig.colorbar(
            p_mesh,
            cax=cbar_axes[j],
            orientation="horizontal",
            label=f"EKE ({units})",
            extend=_ext,
        )
        if n_levels is not None:
            # Log-spaced levels: pick ~4 evenly spaced ticks from the geomspace boundaries
            n_ticks = min(3, len(levels))
            tick_idx = np.linspace(0, len(levels) - 1, n_ticks, dtype=int)
            cb.set_ticks(levels[tick_idx])
            cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
        else:
            _fix_lognorm_ticks(cb, vmin, vmax)

    fig.suptitle(f"Eddy Kinetic Energy{_scale_suffix(emulators_dict)}", y=1.0)
    plt.show()


def plot_variance_comparison_together(
    emulators_dict,
    variable_name,
    units,
    output_dir=None,
    colorbar_fix=False,
    n_levels=None,
):
    """Variance comparison where each emulator uses its corresponding truth.
    Shows one row per unique ground truth, then one row per emulator.
    When n_levels is set, uses a discrete colorbar with that many levels.
    """
    print(f"\nGenerating Variance Comparison for {variable_name}...")

    datasets = list(emulators_dict.keys())
    ncols = len(DEPTH_SLICES)

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_label = list(emulators_dict.keys())[0]
    ref_data = emulators_dict[ref_label]
    ref_tl = ref_data.get("truth_label", "Truth")

    n_truths = 1
    nrows = n_truths + len(datasets)

    # Compute ground truth variance
    plot_data_truth = {}
    ds_truth = ref_data["truth"]
    dz_truth = ds_truth["dz"]
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
    truth_var_3d = ds_truth[variable_name].var("time")
    plot_data_truth[ref_tl] = {}
    for Slices in DEPTH_SLICES:
        plot_data_truth[ref_tl][Slices["title"]] = calculate_vertical_mean_simple(
            truth_var_3d, dz_truth, wet_mask, Slices["min"], Slices["max"]
        ).compute()

    # Compute emulator variance
    plot_data = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        dz_truth = ds_truth["dz"]
        wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
        emu_var_3d = emu_data["ds"][variable_name].var("time")

        plot_data[label] = {}
        for Slices in DEPTH_SLICES:
            plot_data[label][Slices["title"]] = calculate_vertical_mean_simple(
                emu_var_3d, dz_truth, wet_mask, Slices["min"], Slices["max"]
            ).compute()

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(4 * ncols, 2.0 * nrows + 0.8))
    gs = GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        height_ratios=[1] * nrows + [0.04],
        hspace=0.02,
        wspace=0.05,
        top=0.93,
        bottom=0.03,
        left=0.12,
        right=0.97,
    )

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
    cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
    for cax in cbar_axes:
        pos = cax.get_position()
        new_width = pos.width * 0.7
        offset = (pos.width - new_width) / 2
        cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices["title"]
        # Include all truth data in flat_data for normalization
        all_data = [plot_data_truth[ref_tl][key].values.flatten()]
        all_data += [plot_data[ds][key].values.flatten() for ds in datasets]
        flat_data = np.concatenate(all_data)
        flat_data = flat_data[~np.isnan(flat_data)]
        vmax = np.percentile(flat_data, 95) if len(flat_data) > 0 else 1.0

        if n_levels is not None:
            levels = np.linspace(0, vmax, n_levels + 1)
            cmap_use = cmocean.cm.thermal.resampled(n_levels)
            norm = colors.BoundaryNorm(levels, ncolors=cmap_use.N, clip=False)
        else:
            cmap_use = cmocean.cm.thermal
            norm = colors.Normalize(vmin=0, vmax=vmax)

        # Truth row
        ax = axes[0, j]
        p_mesh = plot_data_truth[ref_tl][key].plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="x",
            y="y",
            cmap=cmap_use,
            norm=norm,
            add_colorbar=False,
            rasterized=True,
        )
        ax.add_feature(
            cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
        )
        ax.coastlines(zorder=11)
        row_label = ref_tl.replace(" (", "\n(")
        if j == 0:
            ax.text(
                -0.07,
                0.55,
                row_label,
                va="center",
                ha="right",
                rotation=90,
                transform=ax.transAxes,
                fontweight="bold",
                multialignment="center",
            )
        ax.set_title(key, fontweight="bold")

        # Emulator rows
        for i, ds_name in enumerate(datasets):
            ax = axes[n_truths + i, j]
            p_mesh = plot_data[ds_name][key].plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmap_use,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)
            row_label = ds_name.replace(" (", "\n(")
            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    row_label,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                    multialignment="center",
                )
            ax.set_title("")

        _ext = "both" if n_levels is not None else "neither"
        cb = fig.colorbar(
            p_mesh,
            cax=cbar_axes[j],
            orientation="horizontal",
            label=f"Variance ({units})",
            extend=_ext,
        )
        if n_levels is not None:
            cb.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    fig.suptitle(
        f"Temporal Variance: {_display_name(variable_name)}{_scale_suffix(emulators_dict)}",
        y=1.0,
    )
    plt.show()


def plot_ohc_zonal_bias_ocean_comparison_together(
    emulators_dict, masks_list, colorbar_fix=False
):
    """OHC Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    units = "J/m$^2$"
    num_emulators = len(emulators_dict)

    all_bias_profiles = {}
    all_rmse_vals = {}
    all_biases_flat = []

    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data["truth"]
        masks = masks_list[idx]

        truth_2d = calculate_ohc_2d(ds_truth).mean("time").compute()
        emu_2d = calculate_ohc_2d(emu_data["ds"]).mean("time").compute()
        dx = ds_truth["dx"]
        wet = (ds_truth["thetao"].isel(time=0, lev=0) * 0 + 1).compute()
        weight = (ds_truth["dy"] * ds_truth["dz"]).mean("x").compute()

        all_bias_profiles[label] = {}
        all_rmse_vals[label] = {}
        for basin in BASINS_TO_PLOT:
            t_prof = calculate_zonal_mean_2d(truth_2d, masks[basin], dx, wet).compute()
            e_prof = calculate_zonal_mean_2d(emu_2d, masks[basin], dx, wet).compute()
            bias = e_prof - t_prof
            all_bias_profiles[label][basin] = bias.drop("time")
            all_biases_flat.append(bias.values.flatten())
            all_rmse_vals[label][basin] = np.sqrt(
                ((bias**2) * weight).sum(["y", "lev"]) / weight.sum(["y", "lev"])
            ).item()

    flat = np.concatenate(all_biases_flat)
    max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
    if colorbar_fix:
        max_abs = 2.5e8

    fig, axes = plt.subplots(
        4,
        num_emulators,
        figsize=(5 * num_emulators, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if num_emulators == 1:
        axes = axes.reshape(-1, 1)

    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, (label, bp) in enumerate(all_bias_profiles.items()):
            ax = axes[i, j]
            p = bp[basin].plot.contourf(
                ax=ax,
                cmap=cmocean.cm.balance,
                norm=colors.Normalize(vmin=-max_abs, vmax=max_abs),
                add_colorbar=False,
                levels=np.linspace(-max_abs, max_abs, 25),
                yincrease=False,
                x="y",
                y="lev",
            )
            ax.set_facecolor("grey")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if j == 0:
                ax.set_ylabel(f"{basin}\nDepth (m)")
            ax.set_title(
                f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3e} {units})"
                if i == 0
                else f"RMSE: {all_rmse_vals[label][basin]:.3e}"
            )
    for j in range(num_emulators):
        axes[3, j].set_xlabel("Latitude index")
    fig.colorbar(
        p,
        ax=axes,
        orientation="horizontal",
        shrink=1,
        pad=0.02,
        aspect=40,
        label=f"Bias ({units})",
    )
    fig.suptitle(
        f"Time-Averaged OHC Zonal Mean Bias: All Emulators{_scale_suffix(emulators_dict)}",
        y=1.03,
    )
    plt.show()


def plot_zonal_bias_ocean_comparison_together(
    emulators_dict, variable_name, units, masks_list, output_dir, colorbar_fix=False
):
    """Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    num_emulators = len(emulators_dict)

    all_bias_profiles = {}
    all_rmse_vals = {}
    all_biases_flat = []

    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data["truth"]
        masks = masks_list[idx]

        truth_field = ds_truth[variable_name].mean("time")
        emu_field = emu_data["ds"][variable_name].mean("time")
        dx = ds_truth["dx"]
        wet = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
        weight = (ds_truth["dy"] * ds_truth["dz"]).mean("x").compute()

        all_bias_profiles[label] = {}
        all_rmse_vals[label] = {}
        for basin in BASINS_TO_PLOT:
            t_prof = calculate_zonal_mean(truth_field, masks[basin], dx, wet).compute()
            e_prof = calculate_zonal_mean(emu_field, masks[basin], dx, wet).compute()
            bias = e_prof - t_prof
            all_bias_profiles[label][basin] = bias
            all_biases_flat.append(bias.values.flatten())
            all_rmse_vals[label][basin] = np.sqrt(
                ((bias**2) * weight).sum(["y", "lev"]) / weight.sum(["y", "lev"])
            ).item()

    flat = np.concatenate(all_biases_flat)
    max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
    if colorbar_fix:
        max_abs = 0.514

    fig, axes = plt.subplots(
        4,
        num_emulators,
        figsize=(5 * num_emulators, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if num_emulators == 1:
        axes = axes.reshape(-1, 1)

    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, (label, bp) in enumerate(all_bias_profiles.items()):
            ax = axes[i, j]
            p = bp[basin].plot.contourf(
                ax=ax,
                cmap=cmocean.cm.balance,
                norm=colors.Normalize(vmin=-max_abs, vmax=max_abs),
                add_colorbar=False,
                levels=np.linspace(-max_abs, max_abs, 25),
                yincrease=False,
                x="y",
                y="lev",
                add_labels=False,
            )
            ax.set_facecolor("grey")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if j == 0:
                ax.set_ylabel(f"{basin}\nDepth (m)")
            ax.set_title(
                f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3f} {units})"
                if i == 0
                else f"RMSE: {all_rmse_vals[label][basin]:.3f}"
            )
    for j in range(num_emulators):
        axes[3, j].set_xlabel("Latitude index")
    fig.colorbar(
        p,
        ax=axes,
        orientation="horizontal",
        shrink=1,
        pad=0.02,
        aspect=40,
        label=f"Bias ({units})",
    )
    fig.suptitle(
        f"Time-Averaged Zonal Mean Bias: {_display_name(variable_name)}{_scale_suffix(emulators_dict)}",
        y=1.03,
    )
    plt.show()


def plot_snapshot_zonal_bias_ocean_comparison_together(
    emulators_dict,
    variable_name,
    units,
    masks_list,
    TIME_INDICES,
    output_dir=None,
    colorbar_fix=False,
):
    """Snapshot Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    num_emulators = len(emulators_dict)

    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]["truth"]

    for time_idx in TIME_INDICES:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)

        all_bias_profiles = {}
        all_rmse_vals = {}
        all_biases_flat = []

        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            masks = masks_list[idx]

            truth_snap = ds_truth[variable_name].isel(time=time_idx)
            emu_snap = emu_data["ds"][variable_name].isel(time=time_idx)
            dx = ds_truth["dx"]
            wet = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()
            weight = (ds_truth["dy"] * ds_truth["dz"]).mean("x").compute()

            all_bias_profiles[label] = {}
            all_rmse_vals[label] = {}
            for basin in BASINS_TO_PLOT:
                t_prof = calculate_zonal_mean(
                    truth_snap, masks[basin], dx, wet
                ).compute()
                e_prof = calculate_zonal_mean(emu_snap, masks[basin], dx, wet).compute()
                bias = e_prof - t_prof
                all_bias_profiles[label][basin] = bias
                all_biases_flat.append(bias.values.flatten())
                all_rmse_vals[label][basin] = np.sqrt(
                    ((bias**2) * weight).sum(["y", "lev"]) / weight.sum(["y", "lev"])
                ).item()

        flat = np.concatenate(all_biases_flat)
        max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
        if colorbar_fix:
            max_abs = 0.6

        fig, axes = plt.subplots(
            4,
            num_emulators,
            figsize=(5 * num_emulators, 14),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if num_emulators == 1:
            axes = axes.reshape(-1, 1)

        for i, basin in enumerate(BASINS_TO_PLOT):
            for j, (label, bp) in enumerate(all_bias_profiles.items()):
                ax = axes[i, j]
                p = bp[basin].plot.contourf(
                    ax=ax,
                    cmap=cmocean.cm.balance,
                    norm=colors.Normalize(vmin=-max_abs, vmax=max_abs),
                    add_colorbar=False,
                    levels=np.linspace(-max_abs, max_abs, 25),
                    yincrease=False,
                    x="y",
                    y="lev",
                    add_labels=False,
                )
                ax.set_facecolor("grey")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if j == 0:
                    ax.set_ylabel(f"{basin}\nDepth (m)")
                ax.set_title(
                    f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3f} {units})"
                    if i == 0
                    else f"RMSE: {all_rmse_vals[label][basin]:.3f}"
                )
        for j in range(num_emulators):
            axes[3, j].set_xlabel("Latitude index")
        fig.colorbar(
            p,
            ax=axes,
            orientation="horizontal",
            shrink=1,
            pad=0.02,
            aspect=40,
            label=f"Bias ({units})",
        )
        fig.suptitle(
            f"Snapshot Zonal Bias: {_display_name(variable_name)} at {time_str}{_scale_suffix(emulators_dict)}",
            y=1.03,
        )
        plt.show()


def plot_snapshot_ohc_zonal_bias_ocean_comparison_together(
    emulators_dict, masks_list, TIME_INDICES, colorbar_fix=False
):
    """Snapshot OHC Zonal Bias (Ocean layout) where each emulator uses its corresponding truth."""
    units = "J/m$^2$"
    num_emulators = len(emulators_dict)

    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]["truth"]

    for time_idx in TIME_INDICES:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)

        all_bias_profiles = {}
        all_rmse_vals = {}
        all_biases_flat = []

        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            masks = masks_list[idx]

            truth_snap = calculate_ohc_2d(ds_truth).isel(time=time_idx).compute()
            emu_snap = calculate_ohc_2d(emu_data["ds"]).isel(time=time_idx).compute()
            dx = ds_truth["dx"]
            wet = (ds_truth["thetao"].isel(time=0, lev=0) * 0 + 1).compute()
            weight = (ds_truth["dy"] * ds_truth["dz"]).mean("x").compute()

            all_bias_profiles[label] = {}
            all_rmse_vals[label] = {}
            for basin in BASINS_TO_PLOT:
                t_prof = calculate_zonal_mean_2d(
                    truth_snap, masks[basin], dx, wet
                ).compute()
                e_prof = calculate_zonal_mean_2d(
                    emu_snap, masks[basin], dx, wet
                ).compute()
                bias = e_prof - t_prof
                all_bias_profiles[label][basin] = bias
                all_biases_flat.append(bias.values.flatten())
                all_rmse_vals[label][basin] = np.sqrt(
                    ((bias**2) * weight).sum(["y", "lev"]) / weight.sum(["y", "lev"])
                ).item()

        flat = np.concatenate(all_biases_flat)
        max_abs = np.percentile(np.abs(flat[~np.isnan(flat)]), 98)
        if colorbar_fix:
            max_abs = 4.2e8

        fig, axes = plt.subplots(
            4,
            num_emulators,
            figsize=(5 * num_emulators, 14),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if num_emulators == 1:
            axes = axes.reshape(-1, 1)

        for i, basin in enumerate(BASINS_TO_PLOT):
            for j, (label, bp) in enumerate(all_bias_profiles.items()):
                ax = axes[i, j]
                p = bp[basin].plot.contourf(
                    ax=ax,
                    cmap=cmocean.cm.balance,
                    norm=colors.Normalize(vmin=-max_abs, vmax=max_abs),
                    add_colorbar=False,
                    levels=np.linspace(-max_abs, max_abs, 25),
                    yincrease=False,
                    x="y",
                    y="lev",
                )
                ax.set_facecolor("grey")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if j == 0:
                    ax.set_ylabel(f"{basin}\nDepth (m)")
                ax.set_title(
                    f"{label}\n(RMSE: {all_rmse_vals[label][basin]:.3e} {units})"
                    if i == 0
                    else f"RMSE: {all_rmse_vals[label][basin]:.3e}"
                )
        for j in range(num_emulators):
            axes[3, j].set_xlabel("Latitude index")
        fig.colorbar(
            p,
            ax=axes,
            orientation="horizontal",
            shrink=1,
            pad=0.02,
            aspect=40,
            label=f"Bias ({units})",
        )
        fig.suptitle(
            f"Snapshot OHC Zonal Bias at {time_str}{_scale_suffix(emulators_dict)}",
            y=1.03,
        )
        plt.show()


def plot_ohc_bias_comparison_together(emulators_dict, colorbar_fix=False):
    """OHC Bias comparison in one figure."""
    print(f"\nGenerating OHC Bias Comparison...")
    units = "J/m$^2$"

    datasets = list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_SLICES)

    all_biases = {}
    all_rmses = {}

    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        dz = ds_truth["dz"]
        wet = (ds_truth["thetao"].isel(time=0) * 0 + 1).compute()
        area = ds_truth["areacello"]
        ds_emu = emu_data["ds"]

        all_biases[label] = {}
        all_rmses[label] = {}

        for Slices in DEPTH_SLICES:
            key = Slices["title"]
            truth_map = calculate_ohc_map(
                ds_truth, dz, wet, Slices["min"], Slices["max"]
            ).compute()
            emu_map = calculate_ohc_map(
                ds_emu, dz, wet, Slices["min"], Slices["max"]
            ).compute()
            all_biases[label][key] = emu_map - truth_map
            all_rmses[label][key] = rmse_spatial(truth_map, emu_map, area)

    limits = {}
    for Slices in DEPTH_SLICES:
        key = Slices["title"]
        flat_list = [all_biases[label][key].values.flatten() for label in datasets]
        flat = np.concatenate(flat_list)
        flat = flat[~np.isnan(flat)]
        limit = np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0

        if colorbar_fix:
            limit = {
                "Surface (0-700m)": 3e-12,
                "Intermediate (700-2000m)": 2.5e-12,
                "Deep (2000-7000m)": 2e-12,
            }.get(key, limit)
        limits[key] = limit

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 3.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        gridspec_kw={"hspace": 0.25, "wspace": 0.05},
    )
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices["title"]
        limit = limits[key]
        norm = colors.Normalize(vmin=-limit, vmax=limit)

        for i, label in enumerate(datasets):
            ax = axes[i, j]
            p = all_biases[label][key].plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.balance,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
                add_labels=False,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)

            if i == 0:
                ax.set_title(key, fontweight="bold", y=1.12)

            rmse_str = f"RMSE: {all_rmses[label][key]:.2e} {units}"
            ax.text(
                0.5,
                1.02,
                rmse_str,
                transform=ax.transAxes,
                fontsize=13,
                ha="center",
                va="bottom",
            )

            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    label,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                )

        cb = fig.colorbar(
            p,
            ax=axes[:, j],
            orientation="horizontal",
            shrink=0.7,
            pad=0.03,
            label=f"Bias ({units})",
        )
        cb.ax.ticklabel_format(style="scientific", scilimits=(0, 0), useMathText=True)

    fig.suptitle(
        f"Time-Averaged OHC Bias Comparison{_scale_suffix(emulators_dict)}", y=0.98
    )
    plt.show()


def plot_temperature_bias_comparison_together(emulators_dict, colorbar_fix=False):
    """Temperature Bias comparison in one figure."""
    print(f"\nGenerating Temperature Bias Comparison...")
    VARIABLE_TO_PLOT = "thetao"
    units = "$^\\circ$C"

    datasets = list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_LAYERS)

    all_biases = {}
    all_rmses = {}

    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        dz = ds_truth["dz"]
        wet = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
        area = ds_truth["areacello"]
        ds_emu = emu_data["ds"]

        all_biases[label] = {}
        all_rmses[label] = {}

        for Slices in DEPTH_LAYERS:
            key = Slices["title"]
            idx_layer = Slices["index"]
            truth_map = calculate_temperature_map_layer(
                ds_truth, VARIABLE_TO_PLOT, dz, wet, idx_layer
            ).compute()
            emu_map = calculate_temperature_map_layer(
                ds_emu, VARIABLE_TO_PLOT, dz, wet, idx_layer
            ).compute()
            all_biases[label][key] = emu_map - truth_map
            all_rmses[label][key] = rmse_spatial(truth_map, emu_map, area)

    limits = {}
    for i, Slices in enumerate(DEPTH_LAYERS):
        key = Slices["title"]
        flat_list = [all_biases[label][key].values.flatten() for label in datasets]
        flat = np.concatenate(flat_list)
        flat = flat[~np.isnan(flat)]
        limit = np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0

        if colorbar_fix:
            limit = {0: 2, 1: 0.7, 2: 0.2, 3: 0.1}.get(i, limit)
        limits[key] = limit

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 3.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        gridspec_kw={"hspace": 0.25, "wspace": 0.05},
    )
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for j, Slices in enumerate(DEPTH_LAYERS):
        key = Slices["title"]
        limit = limits[key]
        norm = colors.Normalize(vmin=-limit, vmax=limit)

        for i, label in enumerate(datasets):
            ax = axes[i, j]
            p = all_biases[label][key].plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.balance,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
                add_labels=False,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)

            if i == 0:
                ax.set_title(key, fontweight="bold", y=1.12)

            rmse_str = f"RMSE: {all_rmses[label][key]:.2e} {units}"
            ax.text(
                0.5,
                1.02,
                rmse_str,
                transform=ax.transAxes,
                fontsize=13,
                ha="center",
                va="bottom",
            )

            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    label,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                )

        fig.colorbar(
            p,
            ax=axes[:, j],
            orientation="horizontal",
            shrink=0.7,
            pad=0.03,
            label=f"Bias ({units})",
        )

    fig.suptitle(
        f"Time-Averaged Temperature Bias Comparison{_scale_suffix(emulators_dict)}",
        y=0.98,
    )
    plt.show()


def plot_ohc_snapshot_bias_comparison_together(
    emulators_dict, time_indices, colorbar_fix=False, area=True
):
    """OHC Snapshot Bias where each emulator uses its corresponding truth."""
    units = "J/m$^2$"
    num_emulators = len(emulators_dict)

    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]["truth"]

    for time_idx in time_indices:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)

        all_bias_maps = {}
        all_rmse_vals = {}
        shared_limits = {}

        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            dz = ds_truth["dz"]
            wet = (ds_truth["thetao"].isel(time=0) * 0 + 1).compute()
            area_weight = ds_truth["areacello"]

            for j, Slices in enumerate(DEPTH_SLICES):
                key = Slices["title"]
                truth_map = calculate_ohc_snapshot_map(
                    ds_truth, dz, wet, Slices["min"], Slices["max"], time_idx
                ).compute()
                emu_map = calculate_ohc_snapshot_map(
                    emu_data["ds"], dz, wet, Slices["min"], Slices["max"], time_idx
                ).compute()
                bias = emu_map - truth_map
                all_bias_maps[(label, key)] = bias
                all_rmse_vals[(label, key)] = rmse_spatial(
                    truth_map, emu_map, area_weight
                )

                if idx == 0:
                    flat = bias.values.flatten()
                    flat = flat[~np.isnan(flat)]
                    shared_limits[key] = (
                        np.percentile(np.abs(flat), 99.5) if len(flat) > 0 else 1.0
                    )

        emulator_labels = list(emulators_dict.keys())
        fig, axes = plt.subplots(
            num_emulators,
            len(DEPTH_SLICES),
            figsize=(6 * len(DEPTH_SLICES), 3.5 * num_emulators),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )
        if num_emulators == 1:
            axes = axes.reshape(1, -1)
        if len(DEPTH_SLICES) == 1:
            axes = axes.reshape(-1, 1)

        for j, Slices in enumerate(DEPTH_SLICES):
            key = Slices["title"]
            limit = shared_limits[key]
            if colorbar_fix:
                limit = {0: 3e-12, 1: 2.5e-12, 2: 2e-12}.get(j, limit)
            norm = colors.Normalize(vmin=-limit, vmax=limit)

            for i, label in enumerate(emulator_labels):
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]

                p = bias.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    x="x",
                    y="y",
                    cmap=cmocean.cm.balance,
                    norm=norm,
                    add_colorbar=False,
                    rasterized=True,
                    add_labels=False,
                )

                if i == 0:
                    ax.set_title(key, fontweight="bold", y=1.12)

                rmse_str = f"RMSE: {all_rmse_vals[(label, key)]:.2e} {units}"
                ax.text(
                    0.5,
                    1.02,
                    rmse_str,
                    transform=ax.transAxes,
                    fontsize=13,
                    ha="center",
                    va="bottom",
                )

                ax.add_feature(
                    cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
                )
                ax.coastlines(zorder=11)
                if j == 0:
                    ax.text(
                        -0.07,
                        0.55,
                        label,
                        transform=ax.transAxes,
                        va="center",
                        ha="right",
                        rotation=90,
                        fontweight="bold",
                    )

            cb = fig.colorbar(
                p,
                ax=axes[:, j],
                orientation="horizontal",
                shrink=0.7,
                pad=0.03,
                label=f"Bias ({units})",
            )
            cb.ax.ticklabel_format(
                style="scientific", scilimits=(0, 0), useMathText=True
            )

        fig.suptitle(
            f"OHC Snapshot Bias at {time_str}{_scale_suffix(emulators_dict)}", y=0.98
        )
        plt.show()


def plot_temperature_snapshot_bias_comparison_together(
    emulators_dict, time_indices, colorbar_fix=False
):
    """Temperature Snapshot Bias where each emulator uses its corresponding truth."""
    VARIABLE_TO_PLOT = "thetao"
    units = "$^\\circ$C"
    num_emulators = len(emulators_dict)

    # Get first emulator's truth for time conversion
    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]["truth"]

    for time_idx in time_indices:
        # Get actual date from time coordinate
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]  # Fallback: take first 10 chars (YYYY-MM-DD)

        all_bias_maps = {}
        all_rmse_vals = {}
        shared_limits = {}

        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            dz = ds_truth["dz"]
            wet = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
            area = ds_truth["areacello"]

            for layer in DEPTH_LAYERS:
                key = layer["title"]
                truth_map = calculate_snapshot_map_SpecificLayer(
                    ds_truth, VARIABLE_TO_PLOT, dz, wet, layer["index"], time_idx
                ).compute()
                emu_map = calculate_snapshot_map_SpecificLayer(
                    emu_data["ds"], VARIABLE_TO_PLOT, dz, wet, layer["index"], time_idx
                ).compute()
                bias = emu_map - truth_map
                all_bias_maps[(label, key)] = bias
                all_rmse_vals[(label, key)] = rmse_spatial(truth_map, emu_map, area)

                if idx == 0:
                    flat = bias.values.flatten()
                    flat = flat[~np.isnan(flat)]
                    shared_limits[key] = (
                        np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0
                    )

        emulator_labels = list(emulators_dict.keys())
        fig, axes = plt.subplots(
            num_emulators,
            len(DEPTH_LAYERS),
            figsize=(6 * len(DEPTH_LAYERS), 3.5 * num_emulators),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )
        if num_emulators == 1:
            axes = axes.reshape(1, -1)
        if len(DEPTH_LAYERS) == 1:
            axes = axes.reshape(-1, 1)

        fixed_limits = {"2.5m": 2, "700m": 0.7, "2000m": 0.2, "3000m": 0.1}

        for j, layer in enumerate(DEPTH_LAYERS):
            key = layer["title"]
            limit = (
                fixed_limits.get(key, shared_limits[key])
                if colorbar_fix
                else shared_limits[key]
            )
            norm = colors.Normalize(vmin=-limit, vmax=limit)

            for i, label in enumerate(emulator_labels):
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]

                p = bias.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    x="x",
                    y="y",
                    cmap=cmocean.cm.balance,
                    norm=norm,
                    add_colorbar=False,
                    rasterized=True,
                    add_labels=False,
                )

                if i == 0:
                    ax.set_title(key, fontweight="bold", y=1.12)

                rmse_str = f"RMSE: {all_rmse_vals[(label, key)]:.2e} {units}"
                ax.text(
                    0.5,
                    1.02,
                    rmse_str,
                    transform=ax.transAxes,
                    fontsize=13,
                    ha="center",
                    va="bottom",
                )

                ax.add_feature(
                    cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
                )
                ax.coastlines(zorder=11)
                if j == 0:
                    ax.text(
                        -0.07,
                        0.55,
                        label,
                        transform=ax.transAxes,
                        va="center",
                        ha="right",
                        rotation=90,
                        fontweight="bold",
                    )

            fig.colorbar(
                p,
                ax=axes[:, j],
                orientation="horizontal",
                shrink=0.7,
                pad=0.03,
                label=f"Bias ({units})",
            )

        fig.suptitle(
            f"Temperature Snapshot Bias at {time_str}{_scale_suffix(emulators_dict)}",
            y=0.98,
        )
        plt.show()


# ============================================================================
# _TOGETHER FUNCTIONS
# ============================================================================


def plot_ohc_mean_comparison_together(
    emulators_dict, variable_name=None, units=None, colorbar_fix=False
):
    """OHC Mean Comparison: Ground truth (first emulator) + all emulators on one figure.
    Rows: Truth, Emulator1, Emulator2, ...
    Cols: Depth Slices
    """
    print("\nGenerating OHC Mean Comparison...")

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")
    dz_truth = ds_truth["dz"]
    wet_mask = (ds_truth["thetao"].isel(time=0) * 0 + 1).compute()

    # Compute display truth maps (highest-res for multiscale)
    truth_maps = {}
    for Slices in DEPTH_SLICES:
        key = Slices["title"]
        truth_maps[key] = calculate_ohc_map(
            ds_truth, dz_truth, wet_mask, Slices["min"], Slices["max"]
        ).compute()

    # Compute emulator maps and per-emulator truth maps (for RMSE)
    emu_maps = {}
    emu_truth_maps = {}
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data["ds"]
        ds_emu_truth = emu_data["truth"]
        dz_emu = ds_emu_truth["dz"]
        wet_emu = (ds_emu_truth["thetao"].isel(time=0) * 0 + 1).compute()
        emu_maps[label] = {}
        emu_truth_maps[label] = {}
        for Slices in DEPTH_SLICES:
            key = Slices["title"]
            emu_maps[label][key] = calculate_ohc_map(
                ds_emu, dz_emu, wet_emu, Slices["min"], Slices["max"]
            ).compute()
            emu_truth_maps[label][key] = calculate_ohc_map(
                ds_emu_truth, dz_emu, wet_emu, Slices["min"], Slices["max"]
            ).compute()

    # Plotting
    from matplotlib.gridspec import GridSpec

    multiscale = len(unique_truth_labels) > 1
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_SLICES)

    _hspace = 0.02
    fig = plt.figure(figsize=(4 * ncols, 2.5 * nrows + 0.8))
    gs = GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        height_ratios=[1] * nrows + [0.04],
        hspace=_hspace,
        wspace=0.05,
        top=0.93,
        bottom=0.03,
        left=0.07,
        right=0.97,
    )

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
    cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
    for cax in cbar_axes:
        pos = cax.get_position()
        new_width = pos.width * 0.7
        offset = (pos.width - new_width) / 2
        cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

    for j, Slices in enumerate(DEPTH_SLICES):
        key = Slices["title"]
        data_truth = truth_maps[key].values
        valid_data = data_truth[~np.isnan(data_truth)]
        vmin = np.percentile(valid_data, 0.5)
        vmax = np.percentile(valid_data, 99.5)

        if colorbar_fix:
            if key == "Surface (0-700m)":
                vmin, vmax = 0, 5e-11
            elif key == "Intermediate (700-2000m)":
                vmin, vmax = 0, 5e-11
            elif key == "Deep (2000-7000m)":
                vmin, vmax = 0, 4e-11

        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            if ds_name == truth_label:
                data_to_plot = truth_maps[key]
            else:
                data_to_plot = emu_maps[ds_name][key]

            p = data_to_plot.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.thermal,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)
            if i == 0:
                ax.set_title(key, fontweight="bold")
            else:
                ax.set_title("")
            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    ds_name,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                )
            # RMSE & Pattern Correlation for emulator panels (vs own truth)
            if ds_name != truth_label:
                emu_vals = data_to_plot.values.flatten()
                tru_vals = emu_truth_maps[ds_name][key].values.flatten()
                valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                ax.text(
                    0.02,
                    0.95,
                    f"RMSE: {rmse:.2e}\nCorr: {corr:.4f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        zorder=20,
                    ),
                    verticalalignment="top",
                    linespacing=1.3,
                    zorder=20,
                )

        # Colorbar per column
        cb = fig.colorbar(
            p,
            cax=cbar_axes[j],
            orientation="horizontal",
            label="OHC ($\\times 10^{-11}$ J/m$^2$)",
        )
        current_ticks = cb.ax.get_xticks()
        cb.ax.set_xticklabels([f"{t / 1e-11:.0f}" for t in current_ticks])
        cb.ax.xaxis.get_offset_text().set_visible(False)

    fig.suptitle(f"8-yr-Mean OHC Comparison{_scale_suffix(emulators_dict)}", y=0.99)
    plt.show()


def plot_temperature_mean_comparison_together(
    emulators_dict, colorbar_fix=False, n_levels=20
):
    """Temperature Mean Comparison: Ground truth (first emulator) + all emulators on one figure."""
    print(f"\nGenerating Temperature Mean Comparison (n_levels={n_levels})...")
    VARIABLE_TO_PLOT = "thetao"

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")
    dz_truth = ds_truth["dz"]
    wet_mask = (ds_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()

    # Compute display truth maps (highest-res for multiscale)
    truth_maps = {}
    for Slices in DEPTH_LAYERS:
        key = Slices["title"]
        truth_maps[key] = calculate_temperature_map_layer(
            ds_truth, VARIABLE_TO_PLOT, dz_truth, wet_mask, Slices["index"]
        ).compute()

    # Compute emulator maps and per-emulator truth maps (for RMSE)
    emu_maps = {}
    emu_truth_maps = {}
    for label, emu_data in emulators_dict.items():
        ds_emu = emu_data["ds"]
        ds_emu_truth = emu_data["truth"]
        dz_emu = ds_emu_truth["dz"]
        wet_emu = (ds_emu_truth[VARIABLE_TO_PLOT].isel(time=0) * 0 + 1).compute()
        emu_maps[label] = {}
        emu_truth_maps[label] = {}
        for Slices in DEPTH_LAYERS:
            key = Slices["title"]
            emu_maps[label][key] = calculate_temperature_map_layer(
                ds_emu, VARIABLE_TO_PLOT, dz_emu, wet_emu, Slices["index"]
            ).compute()
            emu_truth_maps[label][key] = calculate_temperature_map_layer(
                ds_emu_truth, VARIABLE_TO_PLOT, dz_emu, wet_emu, Slices["index"]
            ).compute()

    # Plotting
    from matplotlib.gridspec import GridSpec

    multiscale = len(unique_truth_labels) > 1
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(DEPTH_LAYERS)

    _hspace = 0.02
    fig = plt.figure(figsize=(4 * ncols, 2.5 * nrows + 0.8))
    gs = GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        height_ratios=[1] * nrows + [0.04],
        hspace=_hspace,
        wspace=0.05,
        top=0.93,
        bottom=0.03,
        left=0.07,
        right=0.97,
    )

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
    cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
    for cax in cbar_axes:
        pos = cax.get_position()
        new_width = pos.width * 0.7
        offset = (pos.width - new_width) / 2
        cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

    for j, Slices in enumerate(DEPTH_LAYERS):
        key = Slices["title"]
        data_truth = truth_maps[key].values
        valid_data = data_truth[~np.isnan(data_truth)]
        vmin = np.percentile(valid_data, 0.5)
        vmax = np.percentile(valid_data, 99.5)

        if colorbar_fix:
            fixed = {0: (0, 30), 1: (0, 15), 2: (0, 8), 3: (0, 4)}
            vmin, vmax = fixed.get(j, (vmin, vmax))

        levels = np.linspace(vmin, vmax, n_levels + 1)
        cmap_disc = cmocean.cm.thermal.resampled(n_levels)
        norm = colors.BoundaryNorm(levels, ncolors=cmap_disc.N, clip=False)

        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            if ds_name == truth_label:
                data_to_plot = truth_maps[key]
            else:
                data_to_plot = emu_maps[ds_name][key]

            p = data_to_plot.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmap_disc,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)
            if i == 0:
                ax.set_title(key, fontweight="bold")
            else:
                ax.set_title("")
            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    ds_name,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                )
            # RMSE & Pattern Correlation for emulator panels (vs own truth)
            if ds_name != truth_label:
                emu_vals = data_to_plot.values.flatten()
                tru_vals = emu_truth_maps[ds_name][key].values.flatten()
                valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                ax.text(
                    0.02,
                    0.95,
                    f"RMSE: {rmse:.4f}\nCorr: {corr:.4f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        zorder=20,
                    ),
                    verticalalignment="top",
                    linespacing=1.3,
                    zorder=20,
                )

        cb = fig.colorbar(
            p, cax=cbar_axes[j], orientation="horizontal", label="Temperature (°C)"
        )
        cb.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True))

    fig.suptitle(
        f"8-yr-Mean Temperature Comparison{_scale_suffix(emulators_dict)}", y=0.99
    )
    plt.show()


def plot_zonal_mean_comparison_together(
    emulators_dict,
    variable_name,
    units,
    masks_list,
    colorbar_fix=False,
    deseason_metrics=False,
):
    """Zonal Mean Comparison: Ground truth (first emulator) + all emulators on one figure.
    Layout: 4 basins x (1 truth + N emulators) columns
    When deseason_metrics=True, RMSE/Corr are computed on deseasoned per-timestep
    zonal mean profiles instead of on the time-averaged profiles.
    """
    print(f"\nGenerating Zonal Mean Comparison for {variable_name}...")

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_idx = len(emulators_dict) - 1
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_idx = 0
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")
    masks_truth = masks_list[ref_idx]
    dx_truth = ds_truth["dx"]
    wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()

    # Compute truth profiles (for plotting)
    truth_field_mean = ds_truth[variable_name].mean(dim="time")
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean(
            truth_field_mean, masks_truth[basin], dx_truth, wet_mask
        ).compute()

    # Compute emulator profiles (for plotting) and per-emulator truth profiles
    emu_profiles = {}
    emu_truth_profiles = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data["ds"]
        ds_emu_truth = emu_data["truth"]
        masks_emu = masks_list[idx]
        dx_emu = ds_emu_truth["dx"]
        wet_emu = (
            ds_emu_truth[variable_name].isel(time=0, drop=True) * 0 + 1
        ).compute()
        emu_field_mean = ds_emu[variable_name].mean("time")
        truth_field_mean_emu = ds_emu_truth[variable_name].mean(dim="time")
        emu_profiles[label] = {}
        emu_truth_profiles[label] = {}
        for basin in BASINS_TO_PLOT:
            emu_profiles[label][basin] = calculate_zonal_mean(
                emu_field_mean, masks_emu[basin], dx_emu, wet_emu
            ).compute()
            emu_truth_profiles[label][basin] = calculate_zonal_mean(
                truth_field_mean_emu, masks_emu[basin], dx_emu, wet_emu
            ).compute()

    # Compute deseasoned per-timestep zonal mean profiles for metrics
    if deseason_metrics:
        print("  Computing deseasoned zonal mean profiles for metrics...")

        def _deseason(da):
            clim = da.groupby("time.dayofyear").mean("time")
            return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear")

        emu_profiles_metrics = {}
        emu_truth_profiles_metrics = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data["ds"]
            ds_emu_truth = emu_data["truth"]
            masks_emu = masks_list[idx]
            dx_emu = ds_emu_truth["dx"]
            wet_emu = (
                ds_emu_truth[variable_name].isel(time=0, drop=True) * 0 + 1
            ).compute()
            emu_anom = _deseason(ds_emu[variable_name])
            truth_anom = _deseason(ds_emu_truth[variable_name])
            emu_profiles_metrics[label] = {}
            emu_truth_profiles_metrics[label] = {}
            for basin in BASINS_TO_PLOT:
                # Per-timestep zonal mean, then time-average the deseasoned profiles
                emu_zm = []
                tru_zm = []
                for t in range(len(ds_emu.time)):
                    emu_zm.append(
                        calculate_zonal_mean(
                            emu_anom.isel(time=t), masks_emu[basin], dx_emu, wet_emu
                        )
                    )
                    tru_zm.append(
                        calculate_zonal_mean(
                            truth_anom.isel(time=t), masks_emu[basin], dx_emu, wet_emu
                        )
                    )
                emu_profiles_metrics[label][basin] = xr.concat(
                    emu_zm, dim="time"
                ).compute()
                emu_truth_profiles_metrics[label][basin] = xr.concat(
                    tru_zm, dim="time"
                ).compute()
    else:
        emu_profiles_metrics = {
            label: {
                basin: profiles[basin].expand_dims("time") for basin in BASINS_TO_PLOT
            }
            for label, profiles in emu_profiles.items()
        }
        emu_truth_profiles_metrics = {
            label: {
                basin: emu_truth_profiles[label][basin].expand_dims("time")
                for basin in BASINS_TO_PLOT
            }
            for label in emu_profiles
        }

    # Plotting
    num_cols = 1 + len(emulators_dict)
    fig, axes = plt.subplots(
        4,
        num_cols,
        figsize=(4.5 * num_cols, 11),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Determine colorbar range
    all_data = np.concatenate(
        [truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT]
    )
    valid_data = all_data[~np.isnan(all_data)]
    vmin = np.percentile(valid_data, 0.5)
    vmax = np.percentile(valid_data, 98)
    if colorbar_fix:
        vmin, vmax = 0.54, 27.25
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmocean.cm.thermal

    for i, basin in enumerate(BASINS_TO_PLOT):
        # Column 0: Truth
        ax = axes[i, 0]
        p = truth_profiles[basin].plot.contourf(
            ax=ax,
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            levels=np.linspace(vmin, vmax, 25),
            yincrease=False,
            x="y",
            y="lev",
            add_labels=False,
        )
        ax.set_facecolor("grey")
        ax.set_ylabel(f"{basin}\nDepth (m)")
        ax.set_xlabel("")
        if i == 0:
            ax.set_title(truth_label, fontweight="bold")

        # Columns 1+: Emulators
        for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
            ax = axes[i, j]
            profiles[basin].plot.contourf(
                ax=ax,
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25),
                yincrease=False,
                x="y",
                y="lev",
                add_labels=False,
            )
            ax.set_facecolor("grey")
            ax.set_ylabel("")
            ax.set_xlabel("")
            if i == 0:
                ax.set_title(label, fontweight="bold")
            # RMSE & Pattern Correlation (on deseasoned data when deseason_metrics=True)
            emu_vals = emu_profiles_metrics[label][basin].values.flatten()
            tru_vals = emu_truth_profiles_metrics[label][basin].values.flatten()
            valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
            rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
            corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
            tx, ha = (0.98, "right") if basin == "Southern" else (0.02, "left")
            ax.text(
                tx,
                0.95,
                f"RMSE: {rmse:.4f}\nCorr: {corr:.4f}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color="black",
                ha=ha,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, zorder=20
                ),
                verticalalignment="top",
                linespacing=1.3,
                zorder=20,
            )

    for j in range(num_cols):
        axes[3, j].set_xlabel("Latitude index")
    fig.colorbar(
        p,
        ax=axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.08,
        aspect=40,
        label=f"{_display_name(variable_name)} ({units})",
    )
    fig.suptitle(
        f"Time-Averaged Zonal Mean {_display_name(variable_name)}{_scale_suffix(emulators_dict)}",
        y=0.96,
    )
    plt.show()


def plot_ohc_zonal_mean_comparison_together(
    emulators_dict, masks_list, variable_name=None, units=None, colorbar_fix=False
):
    """OHC Zonal Mean Comparison: Ground truth (first emulator) + all emulators."""
    print("\nGenerating OHC Zonal Mean Comparison...")
    units = "J/m$^2$"

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_idx = len(emulators_dict) - 1
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_idx = 0
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")
    masks_truth = masks_list[ref_idx]
    dx_truth = ds_truth["dx"]
    wet_mask = (ds_truth["thetao"].isel(time=0, drop=True) * 0 + 1).compute()

    truth_field_mean = calculate_ohc_2d(ds_truth).mean("time").compute()
    truth_profiles = {}
    for basin in BASINS_TO_PLOT:
        truth_profiles[basin] = calculate_zonal_mean_2d(
            truth_field_mean, masks_truth[basin], dx_truth, wet_mask
        ).compute()

    emu_profiles = {}
    emu_truth_profiles = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data["ds"]
        ds_emu_truth = emu_data["truth"]
        masks_emu = masks_list[idx]
        dx_emu = ds_emu_truth["dx"]
        wet_emu = (ds_emu_truth["thetao"].isel(time=0, drop=True) * 0 + 1).compute()
        emu_field_mean = calculate_ohc_2d(ds_emu).mean("time").compute()
        truth_field_mean_emu = calculate_ohc_2d(ds_emu_truth).mean("time").compute()
        emu_profiles[label] = {}
        emu_truth_profiles[label] = {}
        for basin in BASINS_TO_PLOT:
            emu_profiles[label][basin] = calculate_zonal_mean_2d(
                emu_field_mean, masks_emu[basin], dx_emu, wet_emu
            ).compute()
            emu_truth_profiles[label][basin] = calculate_zonal_mean_2d(
                truth_field_mean_emu, masks_emu[basin], dx_emu, wet_emu
            ).compute()

    num_cols = 1 + len(emulators_dict)
    fig, axes = plt.subplots(
        4,
        num_cols,
        figsize=(4.5 * num_cols, 11),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    all_data = np.concatenate(
        [truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT]
    )
    valid_data = all_data[~np.isnan(all_data)]
    vmin = np.percentile(valid_data, 0.5)
    vmax = np.percentile(valid_data, 98)
    if colorbar_fix:
        vmin, vmax = 0, 16e9
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmocean.cm.thermal

    for i, basin in enumerate(BASINS_TO_PLOT):
        ax = axes[i, 0]
        p = truth_profiles[basin].plot.contourf(
            ax=ax,
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            levels=np.linspace(vmin, vmax, 25),
            yincrease=False,
            x="y",
            y="lev",
            add_labels=False,
        )
        ax.set_facecolor("grey")
        ax.set_ylabel(f"{basin}\nDepth (m)")
        ax.set_xlabel("")
        if i == 0:
            ax.set_title(truth_label, fontweight="bold")

        for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
            ax = axes[i, j]
            profiles[basin].plot.contourf(
                ax=ax,
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25),
                yincrease=False,
                x="y",
                y="lev",
                add_labels=False,
            )
            ax.set_facecolor("grey")
            ax.set_ylabel("")
            ax.set_xlabel("")
            if i == 0:
                ax.set_title(label, fontweight="bold")
            # RMSE & Pattern Correlation vs own truth
            emu_vals = profiles[basin].values.flatten()
            tru_vals = emu_truth_profiles[label][basin].values.flatten()
            valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
            rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
            corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
            tx, ha = (0.98, "right") if basin == "Southern" else (0.02, "left")
            ax.text(
                tx,
                0.95,
                f"RMSE: {rmse:.2e}\nCorr: {corr:.4f}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color="black",
                ha=ha,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, zorder=20
                ),
                verticalalignment="top",
                linespacing=1.3,
                zorder=20,
            )

    for j in range(num_cols):
        axes[3, j].set_xlabel("Latitude index")
    cb = fig.colorbar(
        p,
        ax=axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.08,
        aspect=40,
        label=f"OHC ({units})",
    )
    cb.ax.ticklabel_format(style="scientific", scilimits=(0, 0), useMathText=True)
    fig.suptitle(
        f"Time-Averaged Zonal Mean OHC ({units}){_scale_suffix(emulators_dict)}", y=0.96
    )
    plt.show()


def plot_zonal_mean_global_comparison_together(
    emulators_dict, variable_name, units, colorbar_fix=False, deseason_metrics=False
):
    """Global Zonal Mean Comparison (all ocean, no basin split): Truth + all emulators, 8-yr average.
    When deseason_metrics=True, RMSE/Corr are computed on deseasoned per-timestep
    zonal mean profiles instead of on the time-averaged profiles.
    """
    print(f"\nGenerating Global Zonal Mean Comparison for {variable_name}...")

    first_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[first_label]["truth"]
    truth_label = emulators_dict[first_label].get("truth_label", "Truth")
    dx_truth = ds_truth["dx"]
    wet_mask = (ds_truth[variable_name].isel(time=0, drop=True) * 0 + 1).compute()
    global_mask = (wet_mask * 0 + 1).where(wet_mask.notnull())

    # Profiles for plotting (time-averaged)
    truth_field_mean = ds_truth[variable_name].mean(dim="time")
    truth_profile = calculate_zonal_mean(
        truth_field_mean, global_mask, dx_truth, wet_mask
    ).compute()

    emu_profiles = {}
    emu_truth_profiles = {}
    for idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_emu = emu_data["ds"]
        ds_emu_truth = emu_data["truth"]
        dx_emu = ds_emu_truth["dx"]
        wet_emu = (
            ds_emu_truth[variable_name].isel(time=0, drop=True) * 0 + 1
        ).compute()
        global_mask_emu = (wet_emu * 0 + 1).where(wet_emu.notnull())
        emu_field_mean = ds_emu[variable_name].mean("time")
        truth_field_mean_emu = ds_emu_truth[variable_name].mean("time")
        emu_profiles[label] = calculate_zonal_mean(
            emu_field_mean, global_mask_emu, dx_emu, wet_emu
        ).compute()
        emu_truth_profiles[label] = calculate_zonal_mean(
            truth_field_mean_emu, global_mask_emu, dx_emu, wet_emu
        ).compute()

    # Deseasoned per-timestep zonal mean profiles for metrics
    if deseason_metrics:
        print("  Computing deseasoned zonal mean profiles for metrics...")

        def _deseason(da):
            clim = da.groupby("time.dayofyear").mean("time")
            return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear")

        emu_profiles_metrics = {}
        emu_truth_profiles_metrics = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data["ds"]
            ds_emu_truth = emu_data["truth"]
            dx_emu = ds_emu_truth["dx"]
            wet_emu = (
                ds_emu_truth[variable_name].isel(time=0, drop=True) * 0 + 1
            ).compute()
            global_mask_emu = (wet_emu * 0 + 1).where(wet_emu.notnull())
            emu_anom = _deseason(ds_emu[variable_name])
            truth_anom = _deseason(ds_emu_truth[variable_name])
            emu_zm = []
            tru_zm = []
            for t in range(len(ds_emu.time)):
                emu_zm.append(
                    calculate_zonal_mean(
                        emu_anom.isel(time=t), global_mask_emu, dx_emu, wet_emu
                    )
                )
                tru_zm.append(
                    calculate_zonal_mean(
                        truth_anom.isel(time=t), global_mask_emu, dx_emu, wet_emu
                    )
                )
            emu_profiles_metrics[label] = xr.concat(emu_zm, dim="time").compute()
            emu_truth_profiles_metrics[label] = xr.concat(tru_zm, dim="time").compute()
    else:
        emu_profiles_metrics = {
            label: prof.expand_dims("time") for label, prof in emu_profiles.items()
        }
        emu_truth_profiles_metrics = {
            label: prof.expand_dims("time")
            for label, prof in emu_truth_profiles.items()
        }

    num_cols = 1 + len(emulators_dict)
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(4 * num_cols + 0.5, 3.5))
    gs = GridSpec(
        1,
        num_cols + 1,
        figure=fig,
        width_ratios=[1] * num_cols + [0.03],
        wspace=0.15,
        left=0.08,
        right=0.95,
        top=0.85,
        bottom=0.13,
    )
    axes = [fig.add_subplot(gs[0, j]) for j in range(num_cols)]
    cbar_ax = fig.add_subplot(gs[0, num_cols])
    for j in range(1, num_cols):
        axes[j].sharey(axes[0])

    all_data = truth_profile.values.flatten()
    valid_data = all_data[~np.isnan(all_data)]
    vmin = np.percentile(valid_data, 0.5)
    vmax = np.percentile(valid_data, 98)
    if colorbar_fix:
        vmin, vmax = 0, 30
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmocean.cm.thermal

    ax = axes[0]
    p = truth_profile.plot.contourf(
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        levels=np.linspace(vmin, vmax, 25),
        yincrease=False,
        x="y",
        y="lev",
        add_labels=False,
    )
    ax.set_facecolor("grey")
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Latitude index")
    ax.set_title(truth_label, fontweight="bold")

    for j, (label, profile) in enumerate(emu_profiles.items(), start=1):
        ax = axes[j]
        profile.plot.contourf(
            ax=ax,
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            levels=np.linspace(vmin, vmax, 25),
            yincrease=False,
            x="y",
            y="lev",
            add_labels=False,
        )
        ax.set_facecolor("grey")
        ax.set_ylabel("")
        ax.set_xlabel("Latitude index")
        ax.set_title(label, fontweight="bold")
        plt.setp(ax.get_yticklabels(), visible=False)
        # RMSE & Pattern Correlation (on deseasoned data when deseason_metrics=True)
        emu_vals = emu_profiles_metrics[label].values.flatten()
        tru_vals = emu_truth_profiles_metrics[label].values.flatten()
        valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
        rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
        corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
        ax.text(
            0.02,
            0.05,
            f"RMSE: {rmse:.4f}\nCorr: {corr:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, zorder=20
            ),
            verticalalignment="bottom",
            linespacing=1.3,
            zorder=20,
        )

    fig.colorbar(
        p,
        cax=cbar_ax,
        orientation="vertical",
        label=f"{_display_name(variable_name)} ({units})",
    )
    fig.suptitle(
        f"8-yr-Mean Global Zonal Mean {_display_name(variable_name)} ({units}){_scale_suffix(emulators_dict)}",
        y=1.02,
    )
    plt.show()


def plot_zonal_snapshot_comparison_together(
    emulators_dict,
    variable_name,
    units,
    masks_list,
    time_indices=None,
    colorbar_fix=False,
):
    """Zonal Snapshot Comparison: Ground truth (first emulator) + all emulators."""
    print(f"\nGenerating Zonal Snapshot Comparison for {variable_name}...")
    if time_indices is None:
        time_indices = [0]

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_idx = len(emulators_dict) - 1
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_idx = 0
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")
    masks_truth = masks_list[ref_idx]
    dx_truth = ds_truth["dx"]
    wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()

    for time_idx in time_indices:
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        truth_field = ds_truth[variable_name].isel(time=time_idx)
        truth_profiles = {}
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean(
                truth_field, masks_truth[basin], dx_truth, wet_mask
            ).compute()

        emu_profiles = {}
        emu_truth_profiles = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data["ds"]
            ds_emu_truth = emu_data["truth"]
            masks_emu = masks_list[idx]
            dx_emu = ds_emu_truth["dx"]
            wet_emu = (ds_emu_truth[variable_name].isel(time=0) * 0 + 1).compute()
            emu_field = ds_emu[variable_name].isel(time=time_idx)
            truth_field_emu = ds_emu_truth[variable_name].isel(time=time_idx)
            emu_profiles[label] = {}
            emu_truth_profiles[label] = {}
            for basin in BASINS_TO_PLOT:
                emu_profiles[label][basin] = calculate_zonal_mean(
                    emu_field, masks_emu[basin], dx_emu, wet_emu
                ).compute()
                emu_truth_profiles[label][basin] = calculate_zonal_mean(
                    truth_field_emu, masks_emu[basin], dx_emu, wet_emu
                ).compute()

        num_cols = 1 + len(emulators_dict)
        fig, axes = plt.subplots(
            4,
            num_cols,
            figsize=(4.5 * num_cols, 11),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.12, "wspace": 0.08},
        )
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        all_data = np.concatenate(
            [truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT]
        )
        valid_data = all_data[~np.isnan(all_data)]
        vmin = np.percentile(valid_data, 0.5)
        vmax = np.percentile(valid_data, 98)
        if colorbar_fix:
            vmin, vmax = 0, 25
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.thermal

        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i, 0]
            p = truth_profiles[basin].plot.contourf(
                ax=ax,
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25),
                yincrease=False,
                x="y",
                y="lev",
                add_labels=False,
            )
            ax.set_facecolor("grey")
            ax.set_ylabel(f"{basin}\nDepth (m)")
            ax.set_xlabel("")
            if i == 0:
                ax.set_title(truth_label, fontweight="bold")

            for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
                ax = axes[i, j]
                profiles[basin].plot.contourf(
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    add_colorbar=False,
                    levels=np.linspace(vmin, vmax, 25),
                    yincrease=False,
                    x="y",
                    y="lev",
                    add_labels=False,
                )
                ax.set_facecolor("grey")
                ax.set_ylabel("")
                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(label, fontweight="bold")
                # RMSE & Pattern Correlation vs own truth
                emu_vals = profiles[basin].values.flatten()
                tru_vals = emu_truth_profiles[label][basin].values.flatten()
                valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                tx, ha = (0.98, "right") if basin == "Southern" else (0.02, "left")
                ax.text(
                    tx,
                    0.95,
                    f"RMSE: {rmse:.4f}\nCorr: {corr:.4f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    ha=ha,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        zorder=20,
                    ),
                    verticalalignment="top",
                    linespacing=1.3,
                    zorder=20,
                )

        for j in range(num_cols):
            axes[3, j].set_xlabel("Latitude index")
        fig.colorbar(
            p,
            ax=axes,
            orientation="horizontal",
            shrink=0.6,
            pad=0.08,
            aspect=40,
            label=f"{_display_name(variable_name)} ({units})",
        )
        fig.suptitle(
            f"Snapshot Zonal Mean {_display_name(variable_name)} at {time_str}{_scale_suffix(emulators_dict)}",
            y=0.96,
        )
        plt.show()


def plot_ohc_zonal_snapshot_comparison_together(
    emulators_dict,
    masks_list,
    TIME_INDICES,
    variable_name=None,
    units=None,
    colorbar_fix=False,
):
    """OHC Zonal Snapshot Comparison: Ground truth (first emulator) + all emulators."""
    print("\nGenerating OHC Zonal Snapshot Comparison...")
    units = "J/m$^2$"

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_idx = len(emulators_dict) - 1
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_idx = 0
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")

    for time_idx in TIME_INDICES:
        # Get time string
        time_val = ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        masks_truth = masks_list[ref_idx]
        dx_truth = ds_truth["dx"]
        wet_mask = (ds_truth["thetao"].isel(time=0) * 0 + 1).compute()

        truth_field = calculate_ohc_2d(ds_truth).isel(time=time_idx).compute()
        truth_profiles = {}
        for basin in BASINS_TO_PLOT:
            truth_profiles[basin] = calculate_zonal_mean_2d(
                truth_field, masks_truth[basin], dx_truth, wet_mask
            ).compute()

        emu_profiles = {}
        emu_truth_profiles = {}
        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_emu = emu_data["ds"]
            ds_emu_truth = emu_data["truth"]
            masks_emu = masks_list[idx]
            dx_emu = ds_emu_truth["dx"]
            wet_emu = (ds_emu_truth["thetao"].isel(time=0) * 0 + 1).compute()
            emu_field = calculate_ohc_2d(ds_emu).isel(time=time_idx).compute()
            truth_field_emu = (
                calculate_ohc_2d(ds_emu_truth).isel(time=time_idx).compute()
            )
            emu_profiles[label] = {}
            emu_truth_profiles[label] = {}
            for basin in BASINS_TO_PLOT:
                emu_profiles[label][basin] = calculate_zonal_mean_2d(
                    emu_field, masks_emu[basin], dx_emu, wet_emu
                ).compute()
                emu_truth_profiles[label][basin] = calculate_zonal_mean_2d(
                    truth_field_emu, masks_emu[basin], dx_emu, wet_emu
                ).compute()

        num_cols = 1 + len(emulators_dict)
        fig, axes = plt.subplots(
            4,
            num_cols,
            figsize=(4.5 * num_cols, 11),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.12, "wspace": 0.08},
        )
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        all_data = np.concatenate(
            [truth_profiles[basin].values.flatten() for basin in BASINS_TO_PLOT]
        )
        valid_data = all_data[~np.isnan(all_data)]
        vmin = np.percentile(valid_data, 0.5)
        vmax = np.percentile(valid_data, 98)
        if colorbar_fix:
            vmin, vmax = 0, 16e9
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmocean.cm.thermal

        for i, basin in enumerate(BASINS_TO_PLOT):
            ax = axes[i, 0]
            p = truth_profiles[basin].plot.contourf(
                ax=ax,
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                levels=np.linspace(vmin, vmax, 25),
                yincrease=False,
                x="y",
                y="lev",
                add_labels=False,
            )
            ax.set_facecolor("grey")
            ax.set_ylabel(f"{basin}\nDepth (m)")
            ax.set_xlabel("")
            if i == 0:
                ax.set_title(truth_label, fontweight="bold")

            for j, (label, profiles) in enumerate(emu_profiles.items(), start=1):
                ax = axes[i, j]
                profiles[basin].plot.contourf(
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    add_colorbar=False,
                    levels=np.linspace(vmin, vmax, 25),
                    yincrease=False,
                    x="y",
                    y="lev",
                    add_labels=False,
                )
                ax.set_facecolor("grey")
                ax.set_ylabel("")
                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(label, fontweight="bold")
                # RMSE & Pattern Correlation vs own truth
                emu_vals = profiles[basin].values.flatten()
                tru_vals = emu_truth_profiles[label][basin].values.flatten()
                valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                tx, ha = (0.98, "right") if basin == "Southern" else (0.02, "left")
                ax.text(
                    tx,
                    0.95,
                    f"RMSE: {rmse:.2e}\nCorr: {corr:.4f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    ha=ha,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        zorder=20,
                    ),
                    verticalalignment="top",
                    linespacing=1.3,
                    zorder=20,
                )

        for j in range(num_cols):
            axes[3, j].set_xlabel("Latitude index")
        cb = fig.colorbar(
            p,
            ax=axes,
            orientation="horizontal",
            shrink=0.6,
            pad=0.08,
            aspect=40,
            label=f"OHC ({units})",
        )
        cb.ax.ticklabel_format(style="scientific", scilimits=(0, 0), useMathText=True)
        fig.suptitle(
            f"Snapshot Zonal Mean OHC at {time_str} ({units}){_scale_suffix(emulators_dict)}",
            y=0.96,
        )
        plt.show()


def plot_snapshot_comparison_multidepth_together(
    emulators_dict,
    var_name,
    time_indices,
    depth_configs,
    fix_colorbar=False,
    n_levels=20,
    remove_seasonal=False,
    deseason_metrics=False,
):
    """
    Snapshot Comparison at Multiple Depths (Columns) for specified Time Indices (Separate Figures).
    Rows are paired: each truth is followed by its corresponding emulators.
    When remove_seasonal=True, removes dayofyear climatology before extracting snapshots.
    """
    config = VAR_CONFIG.get(
        var_name, {"cmap": "viridis", "units": "", "label": var_name, "centered": False}
    )
    print(f"\nGenerating Snapshot Comparison for {config['label']}...")

    # Collect unique truths
    unique_truths = {}
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truths:
            unique_truths[tl] = emu_data["truth"]

    truth_labels = list(unique_truths.keys())
    emu_labels = list(emulators_dict.keys())
    ncols = len(depth_configs)

    # Build row order: multiscale → single highest-res truth at top + all emulators
    if len(unique_truths) > 1:
        display_truth_label = truth_labels[-1]
        row_order = [(display_truth_label, True)]
        for el in emu_labels:
            row_order.append((el, False))
    else:
        row_order = []
        for tl in truth_labels:
            row_order.append((tl, True))
            for el in emu_labels:
                if emulators_dict[el].get("truth_label", "Truth") == tl:
                    row_order.append((el, False))
    nrows = len(row_order)

    # Use first truth for time reference
    ds_truth_ref = unique_truths[truth_labels[0]]

    # If remove_seasonal, pre-compute deseasoned datasets
    if remove_seasonal:

        def _deseason(da):
            clim = da.groupby("time.dayofyear").mean("time")
            return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear")

        unique_truths_ds = {}
        for tl, ds_t in unique_truths.items():
            ds_copy = ds_t.copy()
            ds_copy[var_name] = _deseason(ds_t[var_name])
            unique_truths_ds[tl] = ds_copy

        emulators_ds = {}
        for label, emu_data in emulators_dict.items():
            ds_emu_copy = emu_data["ds"].copy()
            ds_emu_copy[var_name] = _deseason(emu_data["ds"][var_name])
            emulators_ds[label] = ds_emu_copy
    else:
        unique_truths_ds = unique_truths
        emulators_ds = {
            label: emu_data["ds"] for label, emu_data in emulators_dict.items()
        }

    # Pre-compute deseasoned datasets for metrics when deseason_metrics=True
    if deseason_metrics and not remove_seasonal:

        def _deseason_m(da):
            clim = da.groupby("time.dayofyear").mean("time")
            return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear")

        unique_truths_ds_metrics = {}
        for tl, ds_t in unique_truths.items():
            ds_copy = ds_t.copy()
            ds_copy[var_name] = _deseason_m(ds_t[var_name])
            unique_truths_ds_metrics[tl] = ds_copy

        emulators_ds_metrics = {}
        for label, emu_data in emulators_dict.items():
            ds_emu_copy = emu_data["ds"].copy()
            ds_emu_copy[var_name] = _deseason_m(emu_data["ds"][var_name])
            emulators_ds_metrics[label] = ds_emu_copy
    else:
        unique_truths_ds_metrics = unique_truths_ds
        emulators_ds_metrics = emulators_ds

    for time_idx in time_indices:
        time_val = ds_truth_ref.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        from matplotlib.gridspec import GridSpec

        _multiscale = len(unique_truths) > 1
        _hspace = 0.02
        fig = plt.figure(figsize=(4 * ncols, 2.0 * nrows + 0.8))
        gs = GridSpec(
            nrows + 1,
            ncols,
            figure=fig,
            height_ratios=[1] * nrows + [0.04],
            hspace=_hspace,
            wspace=0.05,
            top=0.93,
            bottom=0.03,
            left=0.07,
            right=0.97,
        )

        proj = ccrs.Robinson(central_longitude=210)
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
        cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
        for cax in cbar_axes:
            pos = cax.get_position()
            new_width = pos.width * 0.7
            offset = (pos.width - new_width) / 2
            cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

        for j, d_conf in enumerate(depth_configs):
            key = d_conf["title"]
            d_idx = d_conf["index"]

            # Truth Maps
            truth_maps = {}
            for tl in unique_truths.keys():
                ds_use = unique_truths_ds[tl]
                dz_truth = unique_truths[tl]["dz"]
                wet_mask = (unique_truths[tl][var_name].isel(time=0) * 0 + 1).compute()
                truth_maps[tl] = calculate_snapshot_map_SpecificLayer(
                    ds_use, var_name, dz_truth, wet_mask, d_idx, time_idx
                ).compute()

            # Emulator Maps
            emu_maps = {}
            for label, emu_data in emulators_dict.items():
                ds_emu_use = emulators_ds[label]
                ds_emu_truth = emu_data["truth"]
                dz_emu = ds_emu_truth["dz"]
                wet_emu = (ds_emu_truth[var_name].isel(time=0) * 0 + 1).compute()
                emu_maps[label] = calculate_snapshot_map_SpecificLayer(
                    ds_emu_use, var_name, dz_emu, wet_emu, d_idx, time_idx
                ).compute()

            # Deseasoned maps for metrics (only when deseason_metrics and not remove_seasonal)
            if deseason_metrics and not remove_seasonal:
                truth_maps_metrics = {}
                for tl in unique_truths.keys():
                    ds_use_m = unique_truths_ds_metrics[tl]
                    dz_truth = unique_truths[tl]["dz"]
                    wet_mask = (
                        unique_truths[tl][var_name].isel(time=0) * 0 + 1
                    ).compute()
                    truth_maps_metrics[tl] = calculate_snapshot_map_SpecificLayer(
                        ds_use_m, var_name, dz_truth, wet_mask, d_idx, time_idx
                    ).compute()

                emu_maps_metrics = {}
                for label, emu_data in emulators_dict.items():
                    ds_emu_use_m = emulators_ds_metrics[label]
                    ds_emu_truth = emu_data["truth"]
                    dz_emu = ds_emu_truth["dz"]
                    wet_emu = (ds_emu_truth[var_name].isel(time=0) * 0 + 1).compute()
                    emu_maps_metrics[label] = calculate_snapshot_map_SpecificLayer(
                        ds_emu_use_m, var_name, dz_emu, wet_emu, d_idx, time_idx
                    ).compute()
            else:
                truth_maps_metrics = truth_maps
                emu_maps_metrics = emu_maps

            # Normalization from first truth
            first_truth_map = truth_maps[truth_labels[0]]
            valid_data = first_truth_map.values[~np.isnan(first_truth_map.values)]
            if len(valid_data) > 0:
                if remove_seasonal:
                    abs_max = np.percentile(np.abs(valid_data), 95)
                    vmin, vmax = -abs_max, abs_max
                elif config["centered"]:
                    abs_max = np.percentile(np.abs(valid_data), 99.75)
                    vmin, vmax = -abs_max, abs_max
                else:
                    vmin = np.percentile(valid_data, 0.25)
                    vmax = np.percentile(valid_data, 99.75)
            else:
                vmin, vmax = 0, 1

            if fix_colorbar:
                fixed = {0: (-2, 32), 10: (-2, 15), 14: (-2, 10), 15: (-2, 10)}
                vmin, vmax = fixed.get(d_idx, (vmin, vmax))

            levels = np.linspace(vmin, vmax, n_levels + 1)
            if remove_seasonal:
                cmap_obj = plt.cm.RdBu_r
            else:
                cmap_obj = (
                    plt.get_cmap(config["cmap"])
                    if isinstance(config["cmap"], str)
                    else config["cmap"]
                )
            cmap_disc = cmap_obj.resampled(n_levels)
            norm = colors.BoundaryNorm(levels, ncolors=cmap_disc.N, clip=False)

            # Plot rows in paired order (each truth followed by its emulators)
            for row_i, (row_label, is_truth) in enumerate(row_order):
                ax = axes[row_i, j]
                data_map = truth_maps[row_label] if is_truth else emu_maps[row_label]
                p = data_map.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    x="x",
                    y="y",
                    cmap=cmap_disc,
                    norm=norm,
                    add_colorbar=False,
                    rasterized=True,
                )
                ax.add_feature(
                    cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
                )
                ax.coastlines(zorder=11)
                if row_i == 0:
                    ax.set_title(key, fontweight="bold")
                else:
                    ax.set_title("")
                if j == 0:
                    _rl = row_label.replace(" (", "\n(")
                    ax.text(
                        -0.07,
                        0.55,
                        _rl,
                        va="center",
                        ha="right",
                        rotation=90,
                        transform=ax.transAxes,
                        fontweight="bold",
                        multialignment="center",
                    )
                # RMSE & Pattern Correlation for emulator panels
                if not is_truth:
                    tl_for_emu = emulators_dict[row_label].get("truth_label", "Truth")
                    emu_vals = emu_maps_metrics[row_label].values.flatten()
                    tru_vals = truth_maps_metrics[tl_for_emu].values.flatten()
                    valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                    rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                    corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                    ax.text(
                        0.02,
                        0.95,
                        f"RMSE: {rmse:.4f}\nCorr: {corr:.4f}",
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.8,
                            zorder=20,
                        ),
                        verticalalignment="top",
                        linespacing=1.3,
                        zorder=20,
                    )

            _cb_label = (
                f"{config['label']} Anomaly ({config['units']})"
                if remove_seasonal
                else f"{config['label']} ({config['units']})"
            )
            cb = fig.colorbar(
                p,
                cax=cbar_axes[j],
                orientation="horizontal",
                label=_cb_label,
                extend="both",
            )
            if remove_seasonal:
                cb.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
            else:
                cb.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True))

        _anom = " Anomaly" if remove_seasonal else ""
        fig.suptitle(
            f"{config['label']}{_anom} Snapshot at {time_str}{_scale_suffix(emulators_dict)}",
            y=0.99,
        )
        plt.show()


def plot_ohc_snapshot_comparison_multidepth_together(
    emulators_dict,
    time_indices,
    depth_configs,
    variable_name=None,
    units=None,
    colorbar_fix=False,
    remove_seasonal=False,
):
    """
    OHC Snapshot Comparison at Multiple Depths (Columns) for specified Time Indices (Separate Figures).
    Rows are paired: each truth is followed by its corresponding emulators.
    When remove_seasonal=True, removes dayofyear climatology from thetao before OHC computation.
    """
    print(
        f"\nGenerating OHC Snapshot Comparison{'(deseasoned)' if remove_seasonal else ''}..."
    )

    # Collect unique truths
    unique_truths = {}
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truths:
            unique_truths[tl] = emu_data["truth"]

    truth_labels = list(unique_truths.keys())
    emu_labels = list(emulators_dict.keys())
    ncols = len(depth_configs)

    # Build row order: multiscale → single highest-res truth at top + all emulators
    if len(unique_truths) > 1:
        display_truth_label = truth_labels[-1]
        row_order = [(display_truth_label, True)]
        for el in emu_labels:
            row_order.append((el, False))
    else:
        row_order = []
        for tl in truth_labels:
            row_order.append((tl, True))
            for el in emu_labels:
                if emulators_dict[el].get("truth_label", "Truth") == tl:
                    row_order.append((el, False))
    nrows = len(row_order)

    # Use first truth for time reference
    ds_truth_ref = unique_truths[truth_labels[0]]

    # If remove_seasonal, pre-compute deseasoned thetao datasets
    if remove_seasonal:

        def _deseason(da):
            clim = da.groupby("time.dayofyear").mean("time")
            return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear")

        unique_truths_ds = {}
        for tl, ds_t in unique_truths.items():
            ds_copy = ds_t.copy()
            ds_copy["thetao"] = _deseason(ds_t["thetao"])
            unique_truths_ds[tl] = ds_copy

        emulators_ds = {}
        for label, emu_data in emulators_dict.items():
            ds_emu_copy = emu_data["ds"].copy()
            ds_emu_copy["thetao"] = _deseason(emu_data["ds"]["thetao"])
            emulators_ds[label] = ds_emu_copy
    else:
        unique_truths_ds = unique_truths
        emulators_ds = {
            label: emu_data["ds"] for label, emu_data in emulators_dict.items()
        }

    for time_idx in time_indices:
        time_val = ds_truth_ref.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        from matplotlib.gridspec import GridSpec

        _multiscale = len(unique_truths) > 1
        _hspace = 0.02
        fig = plt.figure(figsize=(4 * ncols, 2.5 * nrows + 0.8))
        gs = GridSpec(
            nrows + 1,
            ncols,
            figure=fig,
            height_ratios=[1] * nrows + [0.04],
            hspace=_hspace,
            wspace=0.05,
            top=0.93,
            bottom=0.03,
            left=0.07,
            right=0.97,
        )

        proj = ccrs.Robinson(central_longitude=210)
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
        cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
        for cax in cbar_axes:
            pos = cax.get_position()
            new_width = pos.width * 0.7
            offset = (pos.width - new_width) / 2
            cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

        for j, d_conf in enumerate(depth_configs):
            key = d_conf["title"]
            d_min = d_conf["min"]
            d_max = d_conf["max"]

            # Truth Maps
            truth_maps = {}
            for tl in unique_truths.keys():
                ds_use = unique_truths_ds[tl]
                dz_truth = unique_truths[tl]["dz"]
                wet_mask = (unique_truths[tl]["thetao"].isel(time=0) * 0 + 1).compute()
                truth_maps[tl] = calculate_ohc_snapshot_map(
                    ds_use, dz_truth, wet_mask, d_min, d_max, time_idx
                ).compute()

            # Emulator Maps
            emu_maps = {}
            for label, emu_data in emulators_dict.items():
                ds_emu_use = emulators_ds[label]
                ds_emu_truth = emu_data["truth"]
                dz_emu = ds_emu_truth["dz"]
                wet_emu = (ds_emu_truth["thetao"].isel(time=0) * 0 + 1).compute()
                emu_maps[label] = calculate_ohc_snapshot_map(
                    ds_emu_use, dz_emu, wet_emu, d_min, d_max, time_idx
                ).compute()

            # Normalization from first truth
            first_truth_map = truth_maps[truth_labels[0]]
            valid_data = first_truth_map.values[~np.isnan(first_truth_map.values)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 0.25)
                vmax = np.percentile(valid_data, 99.75)
            else:
                vmin, vmax = 0, 1

            if colorbar_fix:
                if key == "Surface (0-700m)":
                    vmin, vmax = 0, 5e-11
                elif key == "Intermediate (700-2000m)":
                    vmin, vmax = 0, 5e-11
                elif key == "Deep (2000-7000m)":
                    vmin, vmax = 0, 5e-11

            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            # Plot rows in paired order (each truth followed by its emulators)
            for row_i, (row_label, is_truth) in enumerate(row_order):
                ax = axes[row_i, j]
                data_map = truth_maps[row_label] if is_truth else emu_maps[row_label]
                p = data_map.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    x="x",
                    y="y",
                    cmap=cmocean.cm.thermal,
                    norm=norm,
                    add_colorbar=False,
                    rasterized=True,
                )
                ax.add_feature(
                    cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
                )
                ax.coastlines(zorder=11)
                if row_i == 0:
                    ax.set_title(key, fontweight="bold")
                else:
                    ax.set_title("")
                if j == 0:
                    _rl = row_label.replace(" (", "\n(")
                    ax.text(
                        -0.07,
                        0.55,
                        _rl,
                        va="center",
                        ha="right",
                        rotation=90,
                        transform=ax.transAxes,
                        fontweight="bold",
                        multialignment="center",
                    )
                # RMSE & Pattern Correlation for emulator panels
                if not is_truth:
                    tl_for_emu = emulators_dict[row_label].get("truth_label", "Truth")
                    emu_vals = data_map.values.flatten()
                    tru_vals = truth_maps[tl_for_emu].values.flatten()
                    valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                    rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                    corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                    ax.text(
                        0.02,
                        0.95,
                        f"RMSE: {rmse:.2e}\nCorr: {corr:.4f}",
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.8,
                            zorder=20,
                        ),
                        verticalalignment="top",
                        linespacing=1.3,
                        zorder=20,
                    )

            cb = fig.colorbar(
                p,
                cax=cbar_axes[j],
                orientation="horizontal",
                label="OHC ($\\times 10^{-11}$ J/m$^2$)",
            )
            current_ticks = cb.ax.get_xticks()
            cb.ax.set_xticklabels([f"{t / 1e-11:.0f}" for t in current_ticks])
            cb.ax.xaxis.get_offset_text().set_visible(False)

        _anom = " Anomaly" if remove_seasonal else ""
        fig.suptitle(
            f"OHC{_anom} Snapshot at {time_str}{_scale_suffix(emulators_dict)}", y=0.99
        )
        plt.show()


def plot_ke_mean_comparison_together(
    emulators_dict, depth_configs=None, var_u="uo", var_v="vo", colorbar_fix=False
):
    """
    Time-Averaged KE Comparison: Ground truth (highest-res) + all emulators on one figure.
    KE = 0.5 * (u^2 + v^2) averaged over time.
    Rows: Truth, Emulator1, Emulator2, ...
    Cols: Depth layers
    """
    print("\nGenerating Time-Averaged KE Comparison...")

    if depth_configs is None:
        depth_configs = DEPTH_LAYERS

    # Detect multiscale: use highest-res (last) truth when multiple unique truths exist
    unique_truth_labels = []
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truth_labels:
            unique_truth_labels.append(tl)

    if len(unique_truth_labels) > 1:
        ref_label = list(emulators_dict.keys())[-1]
    else:
        ref_label = list(emulators_dict.keys())[0]
    ds_truth = emulators_dict[ref_label]["truth"]
    truth_label = emulators_dict[ref_label].get("truth_label", "Truth")

    def compute_ke_mean(ds, depth_index):
        u = ds[var_u].isel(lev=depth_index)
        v = ds[var_v].isel(lev=depth_index)
        return (0.5 * (u**2 + v**2)).mean("time").compute()

    # Compute display truth maps (highest-res for multiscale)
    truth_maps = {}
    for d_conf in depth_configs:
        truth_maps[d_conf["title"]] = compute_ke_mean(ds_truth, d_conf["index"])

    # Compute emulator maps and per-emulator truth maps (for RMSE)
    emu_maps = {}
    emu_truth_maps = {}
    for label, emu_data in emulators_dict.items():
        emu_maps[label] = {}
        emu_truth_maps[label] = {}
        for d_conf in depth_configs:
            key = d_conf["title"]
            emu_maps[label][key] = compute_ke_mean(emu_data["ds"], d_conf["index"])
            emu_truth_maps[label][key] = compute_ke_mean(
                emu_data["truth"], d_conf["index"]
            )

    # Plotting
    from matplotlib.gridspec import GridSpec

    multiscale = len(unique_truth_labels) > 1
    datasets = [truth_label] + list(emulators_dict.keys())
    nrows = len(datasets)
    ncols = len(depth_configs)

    _hspace = 0.02
    fig = plt.figure(figsize=(4 * ncols, 2.5 * nrows + 0.8))
    gs = GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        height_ratios=[1] * nrows + [0.04],
        hspace=_hspace,
        wspace=0.05,
        top=0.93,
        bottom=0.03,
        left=0.07,
        right=0.97,
    )

    proj = ccrs.Robinson(central_longitude=210)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
    cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
    for cax in cbar_axes:
        pos = cax.get_position()
        new_width = pos.width * 0.7
        offset = (pos.width - new_width) / 2
        cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

    for j, d_conf in enumerate(depth_configs):
        key = d_conf["title"]

        # LogNorm for KE
        valid_data = truth_maps[key].values[~np.isnan(truth_maps[key].values)]
        flat_pos = valid_data[valid_data > 0]
        vmin = np.percentile(flat_pos, 5) if len(flat_pos) > 0 else 1e-10
        vmax = np.percentile(flat_pos, 99) if len(flat_pos) > 0 else 1.0
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        for i, ds_name in enumerate(datasets):
            ax = axes[i, j]
            if ds_name == truth_label:
                data_to_plot = truth_maps[key]
            else:
                data_to_plot = emu_maps[ds_name][key]

            p = data_to_plot.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.thermal,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax.coastlines(zorder=11)
            if i == 0:
                ax.set_title(key, fontweight="bold")
            else:
                ax.set_title("")
            if j == 0:
                ax.text(
                    -0.07,
                    0.55,
                    ds_name,
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax.transAxes,
                    fontweight="bold",
                )
            # RMSE & Pattern Correlation for emulator panels (vs own truth)
            if ds_name != truth_label:
                emu_vals = data_to_plot.values.flatten()
                tru_vals = emu_truth_maps[ds_name][key].values.flatten()
                valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                ax.text(
                    0.02,
                    0.95,
                    f"RMSE: {rmse:.2e}\nCorr: {corr:.4f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        zorder=20,
                    ),
                    verticalalignment="top",
                    linespacing=1.3,
                    zorder=20,
                )

        cb = fig.colorbar(
            p, cax=cbar_axes[j], orientation="horizontal", label="KE (m$^2$/s$^2$)"
        )
        _fix_lognorm_ticks(cb, vmin, vmax)

    fig.suptitle(f"Time-Averaged KE Comparison{_scale_suffix(emulators_dict)}", y=0.99)
    plt.show()


def plot_ke_snapshot_comparison_multidepth_together(
    emulators_dict,
    time_indices,
    depth_configs,
    var_u="uo",
    var_v="vo",
    colorbar_fix=False,
    remove_seasonal=False,
    n_levels=None,
):
    """
    KE Snapshot Comparison at Multiple Depths (Columns) for specified Time Indices.
    KE = 0.5 * (u^2 + v^2) at each time snapshot.
    When remove_seasonal=True, removes dayofyear climatology from u,v before computing KE (gives EKE snapshot).
    When n_levels is set, uses a discrete colorbar with log-spaced levels.
    Rows are paired: each truth is followed by its corresponding emulators.
    """
    print(f"\nGenerating KE Snapshot Comparison...")

    # Collect unique truths
    unique_truths = {}
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truths:
            unique_truths[tl] = emu_data["truth"]

    truth_labels = list(unique_truths.keys())
    emu_labels = list(emulators_dict.keys())
    ncols = len(depth_configs)

    # Build row order: multiscale → single highest-res truth at top + all emulators
    if len(unique_truths) > 1:
        display_truth_label = truth_labels[-1]
        row_order = [(display_truth_label, True)]
        for el in emu_labels:
            row_order.append((el, False))
    else:
        row_order = []
        for tl in truth_labels:
            row_order.append((tl, True))
            for el in emu_labels:
                if emulators_dict[el].get("truth_label", "Truth") == tl:
                    row_order.append((el, False))
    nrows = len(row_order)

    ds_truth_ref = unique_truths[truth_labels[0]]

    # If remove_seasonal, pre-compute deseasoned velocity datasets
    if remove_seasonal:

        def _deseason(da):
            clim = da.groupby("time.dayofyear").mean("time")
            return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear")

        unique_truths_ke = {}
        for tl, ds_t in unique_truths.items():
            ds_copy = ds_t.copy()
            ds_copy[var_u] = _deseason(ds_t[var_u])
            ds_copy[var_v] = _deseason(ds_t[var_v])
            unique_truths_ke[tl] = ds_copy

        emulators_ke = {}
        for label, emu_data in emulators_dict.items():
            ds_copy = emu_data["ds"].copy()
            ds_copy[var_u] = _deseason(emu_data["ds"][var_u])
            ds_copy[var_v] = _deseason(emu_data["ds"][var_v])
            emulators_ke[label] = ds_copy
    else:
        unique_truths_ke = unique_truths
        emulators_ke = {
            label: emu_data["ds"] for label, emu_data in emulators_dict.items()
        }

    def compute_ke_snapshot(ds, depth_index, time_idx):
        u = ds[var_u].isel(time=time_idx, lev=depth_index)
        v = ds[var_v].isel(time=time_idx, lev=depth_index)
        return (0.5 * (u**2 + v**2)).compute()

    for time_idx in time_indices:
        time_val = ds_truth_ref.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        from matplotlib.gridspec import GridSpec

        _multiscale = len(unique_truths) > 1
        _hspace = 0.02
        fig = plt.figure(figsize=(4 * ncols, 2.5 * nrows + 0.8))
        gs = GridSpec(
            nrows + 1,
            ncols,
            figure=fig,
            height_ratios=[1] * nrows + [0.04],
            hspace=_hspace,
            wspace=0.05,
            top=0.93,
            bottom=0.03,
            left=0.07,
            right=0.97,
        )

        proj = ccrs.Robinson(central_longitude=210)
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
        cbar_axes = [fig.add_subplot(gs[nrows, j]) for j in range(ncols)]
        for cax in cbar_axes:
            pos = cax.get_position()
            new_width = pos.width * 0.7
            offset = (pos.width - new_width) / 2
            cax.set_position([pos.x0 + offset, pos.y0, new_width, pos.height])

        for j, d_conf in enumerate(depth_configs):
            key = d_conf["title"]
            d_idx = d_conf["index"]

            # Truth Maps
            truth_maps = {}
            for tl in unique_truths.keys():
                truth_maps[tl] = compute_ke_snapshot(
                    unique_truths_ke[tl], d_idx, time_idx
                )

            # Emulator Maps
            emu_maps = {}
            for label in emulators_dict.keys():
                emu_maps[label] = compute_ke_snapshot(
                    emulators_ke[label], d_idx, time_idx
                )

            # Normalization (LogNorm for KE, optionally discrete)
            first_truth_map = truth_maps[truth_labels[0]]
            valid_data = first_truth_map.values[~np.isnan(first_truth_map.values)]
            flat_pos = valid_data[valid_data > 0]
            vmin = np.percentile(flat_pos, 5) if len(flat_pos) > 0 else 1e-10
            vmax = np.percentile(flat_pos, 99) if len(flat_pos) > 0 else 1.0

            if n_levels is not None:
                levels = np.geomspace(vmin, vmax, n_levels + 1)
                cmap_use = cmocean.cm.thermal.resampled(n_levels)
                norm = colors.BoundaryNorm(levels, ncolors=cmap_use.N, clip=False)
            else:
                cmap_use = cmocean.cm.thermal
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)

            # Plot rows in paired order
            for row_i, (row_label, is_truth) in enumerate(row_order):
                ax = axes[row_i, j]
                data_map = truth_maps[row_label] if is_truth else emu_maps[row_label]
                p = data_map.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    x="x",
                    y="y",
                    cmap=cmap_use,
                    norm=norm,
                    add_colorbar=False,
                    rasterized=True,
                )
                ax.add_feature(
                    cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
                )
                ax.coastlines(zorder=11)
                if row_i == 0:
                    ax.set_title(key, fontweight="bold")
                else:
                    ax.set_title("")
                if j == 0:
                    _rl = row_label.replace(" (", "\n(")
                    ax.text(
                        -0.07,
                        0.55,
                        _rl,
                        va="center",
                        ha="right",
                        rotation=90,
                        transform=ax.transAxes,
                        fontweight="bold",
                        multialignment="center",
                    )
                # RMSE & Pattern Correlation for emulator panels
                if not is_truth:
                    tl_for_emu = emulators_dict[row_label].get("truth_label", "Truth")
                    emu_vals = data_map.values.flatten()
                    tru_vals = truth_maps[tl_for_emu].values.flatten()
                    valid = ~(np.isnan(emu_vals) | np.isnan(tru_vals))
                    rmse = np.sqrt(np.mean((emu_vals[valid] - tru_vals[valid]) ** 2))
                    corr = np.corrcoef(emu_vals[valid], tru_vals[valid])[0, 1]
                    ax.text(
                        0.02,
                        0.95,
                        f"RMSE: {rmse:.2e}\nCorr: {corr:.4f}",
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.8,
                            zorder=20,
                        ),
                        verticalalignment="top",
                        linespacing=1.3,
                        zorder=20,
                    )

            _cb_label = "EKE (m$^2$/s$^2$)" if remove_seasonal else "KE (m$^2$/s$^2$)"
            _ext = "both" if n_levels is not None else "neither"
            cb = fig.colorbar(
                p,
                cax=cbar_axes[j],
                orientation="horizontal",
                label=_cb_label,
                extend=_ext,
            )
            if n_levels is not None:
                tick_vals = [vmin, np.sqrt(vmin * vmax), vmax]
                cb.set_ticks(tick_vals)
                cb.set_ticklabels([f"{v:.1e}" for v in tick_vals])
            else:
                _fix_lognorm_ticks(cb, vmin, vmax)

        fig.suptitle(
            f"{'EKE' if remove_seasonal else 'Kinetic Energy'} Snapshot at {time_str}{_scale_suffix(emulators_dict)}",
            y=0.99,
        )
        plt.show()


def plot_ke_snapshot_bias_comparison_together(
    emulators_dict,
    time_indices,
    depth_layers,
    var_u="uo",
    var_v="vo",
    colorbar_fix=False,
):
    """
    KE Snapshot Bias: (emulator KE - truth KE) at specific depth layers and time indices.
    KE = 0.5 * (u^2 + v^2) at each time snapshot.
    """
    units = "m$^2$/s$^2$"
    num_emulators = len(emulators_dict)

    first_label = list(emulators_dict.keys())[0]
    first_ds_truth = emulators_dict[first_label]["truth"]

    def compute_ke_snapshot(ds, depth_index, time_idx):
        u = ds[var_u].isel(time=time_idx, lev=depth_index)
        v = ds[var_v].isel(time=time_idx, lev=depth_index)
        return (0.5 * (u**2 + v**2)).compute()

    for time_idx in time_indices:
        time_val = first_ds_truth.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        all_bias_maps = {}
        all_rmse_vals = {}
        shared_limits = {}

        for idx, (label, emu_data) in enumerate(emulators_dict.items()):
            ds_truth = emu_data["truth"]
            area = ds_truth["areacello"]

            for layer in depth_layers:
                key = layer["title"]
                truth_ke = compute_ke_snapshot(ds_truth, layer["index"], time_idx)
                emu_ke = compute_ke_snapshot(emu_data["ds"], layer["index"], time_idx)
                bias = emu_ke - truth_ke
                all_bias_maps[(label, key)] = bias
                all_rmse_vals[(label, key)] = rmse_spatial(truth_ke, emu_ke, area)

                if idx == 0:
                    flat = bias.values.flatten()
                    flat = flat[~np.isnan(flat)]
                    shared_limits[key] = (
                        np.percentile(np.abs(flat), 99) if len(flat) > 0 else 1.0
                    )

        emulator_labels = list(emulators_dict.keys())
        fig, axes = plt.subplots(
            num_emulators,
            len(depth_layers),
            figsize=(6 * len(depth_layers), 3.5 * num_emulators),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )
        if num_emulators == 1:
            axes = axes.reshape(1, -1)
        if len(depth_layers) == 1:
            axes = axes.reshape(-1, 1)

        for j, layer in enumerate(depth_layers):
            key = layer["title"]
            limit = shared_limits[key]
            norm = colors.Normalize(vmin=-limit, vmax=limit)

            for i, label in enumerate(emulator_labels):
                ax = axes[i, j]
                bias = all_bias_maps[(label, key)]

                p = bias.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    x="x",
                    y="y",
                    cmap=cmocean.cm.balance,
                    norm=norm,
                    add_colorbar=False,
                    rasterized=True,
                    add_labels=False,
                )

                if i == 0:
                    ax.set_title(key, fontweight="bold", y=1.12)

                rmse_str = f"RMSE: {all_rmse_vals[(label, key)]:.2e} {units}"
                ax.text(
                    0.5,
                    1.02,
                    rmse_str,
                    transform=ax.transAxes,
                    fontsize=13,
                    ha="center",
                    va="bottom",
                )

                ax.add_feature(
                    cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
                )
                ax.coastlines(zorder=11)
                if j == 0:
                    ax.text(
                        -0.07,
                        0.55,
                        label,
                        transform=ax.transAxes,
                        va="center",
                        ha="right",
                        rotation=90,
                        fontweight="bold",
                    )

            cb = fig.colorbar(
                p,
                ax=axes[:, j],
                orientation="horizontal",
                shrink=0.7,
                pad=0.03,
                label=f"Bias ({units})",
            )
            cb.ax.ticklabel_format(
                style="scientific", scilimits=(-2, 2), useMathText=True
            )

        fig.suptitle(
            f"KE Snapshot Bias at {time_str}{_scale_suffix(emulators_dict)}", y=0.98
        )
        plt.show()


# =============================================================================
# MOVIE / ANIMATION HELPERS
# Generate animated GIFs from snapshot plotting functions.
# Each frame = one year, starting from a configurable time index.
# =============================================================================


# ---------------------------------------------------------------------------
# Parallel rendering helpers
# ---------------------------------------------------------------------------
# Shared state populated before forking workers (read-only in children via COW).
_parallel_render_ctx = {}


# ---------------------------------------------------------------------------
# Convenience movie functions (thin wrappers around make_snapshot_movie)
# ---------------------------------------------------------------------------


# =============================================================================
# SPECTRUM _TOGETHER FUNCTIONS
# Each emulator has its own corresponding ground truth.
# emulators_dict format:
#   {label: {'ds': ds_emu, 'color': color, 'truth': ds_truth, 'truth_label': str}}
# =============================================================================


def plot_isotropic_spectrum_comparison_together(
    emulators_dict,
    var_to_eval="thetao",
    lev_idx=0,
    time_window=None,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers=[0.01, 0.02],
    show_ratios=True,
    colorbar_fix=False,
    region_name=None,
    ax=None,
):
    """
    Plots isotropic power spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_to_eval : str, default='thetao'
        Variable name to evaluate
    lev_idx : int, default=0
        Level index for depth slice
    time_window : int, default=100
        Number of time steps to use
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for analysis
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for analysis
    target_wavenumbers : list, default=[0.01, 0.02]
        Wavenumbers at which to calculate and annotate ratios
    show_ratios : bool, default=True
        Whether to show ratio annotations at target wavenumbers
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    # print(f"\nGenerating Isotropic Power Spectrum Comparison for {var_to_eval}...")

    def get_spectrum(ds_target, var_name, lev_i, t_window, x_sl, y_sl):
        """Helper function to compute isotropic spectrum for a dataset."""
        dx_mean = float(ds_target.dx.sel({"x": x_sl, "y": y_sl}).mean().values)
        dy_mean = float(ds_target.dy.sel({"x": x_sl, "y": y_sl}).mean().values)

        data_slice_raw = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
        data_slice_raw = data_slice_raw.transpose("time", ...).sel(
            {"x": x_sl, "y": y_sl}
        )

        time_mean = data_slice_raw.mean(dim="time")
        data_slice_anom = data_slice_raw - time_mean
        data_prepared = data_slice_anom.fillna(0).values

        k_cent, spec = compute_isotropic_spectrum_torch(
            torch.as_tensor(data_prepared),
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        return k_cent.cpu().numpy() * 1000 * 2 * np.pi, spec.cpu().numpy()

    # Compute spectra for each truth-emulator pair
    all_spectra = {}  # {label: {'k_truth': ..., 'spec_truth': ..., 'k_emu': ..., 'spec_emu': ...}}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        truth_label = emu_data.get("truth_label", "Truth")

        # print(f"Calculating spectrum for {truth_label} (truth of {label})...")
        k_truth, spec_truth = get_spectrum(
            ds_truth, var_to_eval, lev_idx, time_window, lon_slice, lat_slice
        )

        # print(f"Calculating spectrum for {label}...")
        k_emu, spec_emu = get_spectrum(
            ds_emu, var_to_eval, lev_idx, time_window, lon_slice, lat_slice
        )

        all_spectra[label] = {
            "k_truth": k_truth,
            "spec_truth": spec_truth,
            "k_emu": k_emu,
            "spec_emu": spec_emu,
            "truth_label": truth_label,
        }

    # Plotting
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        truth_label = data["truth_label"]

        # Plot truth (dashed) — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data["k_truth"],
                data["spec_truth"].mean(0),
                "--",
                label=f"{truth_label}",
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid)
        ax.loglog(
            data["k_emu"],
            data["spec_emu"].mean(0),
            "-",
            label=f"{label}",
            linewidth=2,
            color=color,
        )

    # Add ratio annotations at target wavenumbers
    if show_ratios:
        for target_k in target_wavenumbers:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data["k_truth"] - target_k))
                idx_emu = np.argmin(np.abs(data["k_emu"] - target_k))
                actual_k = data["k_truth"][idx_truth]
                spec_truth_val = data["spec_truth"].mean(0)[idx_truth]
                spec_emu_val = data["spec_emu"].mean(0)[idx_emu]
                ratio = spec_emu_val / spec_truth_val if spec_truth_val > 0 else 0

                print(
                    f"{label} vs {data['truth_label']} at k={actual_k:.4f}: ratio={ratio:.3f}"
                )

                color = emulators_dict[label]["color"]
                x_pos = actual_k * 1.1
                y_pos = spec_emu_val * (1.5**i)

                ax.text(
                    x_pos,
                    y_pos,
                    f"{label}/{data['truth_label']}={ratio:.3f}",
                    fontsize=12,
                    color=color,
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color
                    ),
                )

            ax.axvline(x=actual_k, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    # Format axes
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=6))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=True))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=12))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
    ax.set_ylabel("Power Spectral Density")

    # Get depth value from first emulator's truth
    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"

    if region_name:
        region_str = f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
    else:
        region_str = ""
    ax.set_title(region_str)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if own_figure:
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_isotropic_spectrum_with_ratio(
    emulators_dict,
    spectrum_regions,
    var_to_eval="thetao",
    lev_idx=0,
    time_window=None,
    ratio_ylim=(0, 1.5),
):
    """
    Plot isotropic power spectrum + ratio for multiple regions.

    Layout: N_regions rows × 2 columns.
      Left column  – log-log power spectrum (truth dashed, emulator solid).
      Right column – ratio (emulator PSD / truth PSD), semilog-x with linear y.

    Parameters
    ----------
    emulators_dict : dict
        {label: {'ds', 'color', 'truth', 'truth_label'}}
    spectrum_regions : list of (name, lon_slice, lat_slice)
    var_to_eval : str
    lev_idx : int
    time_window : int or None
    ratio_ylim : tuple
        y-axis limits for the ratio panels.
    """
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    def get_spectrum(ds_target, var_name, lev_i, t_window, x_sl, y_sl):
        dx_mean = float(ds_target.dx.sel({"x": x_sl, "y": y_sl}).mean().values)
        dy_mean = float(ds_target.dy.sel({"x": x_sl, "y": y_sl}).mean().values)
        data_slice_raw = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
        data_slice_raw = data_slice_raw.transpose("time", ...).sel(
            {"x": x_sl, "y": y_sl}
        )
        time_mean = data_slice_raw.mean(dim="time")
        data_slice_anom = data_slice_raw - time_mean
        data_prepared = data_slice_anom.fillna(0).values
        k_cent, spec = compute_isotropic_spectrum_torch(
            torch.as_tensor(data_prepared),
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        return k_cent.cpu().numpy() * 1000 * 2 * np.pi, spec.cpu().numpy()

    n_regions = len(spectrum_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(12, 3.5 * n_regions))
    if n_regions == 1:
        axes = axes[np.newaxis, :]

    for row, (region_name, lon_slice, lat_slice) in enumerate(spectrum_regions):
        ax_spec = axes[row, 0]
        ax_ratio = axes[row, 1]

        # Compute spectra
        plotted_truths = set()
        for label, emu_data in emulators_dict.items():
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]

            k_truth, spec_truth = get_spectrum(
                ds_truth, var_to_eval, lev_idx, time_window, lon_slice, lat_slice
            )
            k_emu, spec_emu = get_spectrum(
                ds_emu, var_to_eval, lev_idx, time_window, lon_slice, lat_slice
            )

            spec_truth_mean = spec_truth.mean(0)
            spec_emu_mean = spec_emu.mean(0)

            # Left panel: power spectrum (truth dashed, emulator solid)
            if truth_label not in plotted_truths:
                ax_spec.loglog(
                    k_truth,
                    spec_truth_mean,
                    "--",
                    label=f"{truth_label}",
                    linewidth=2,
                    color=color,
                    alpha=0.7,
                )
                plotted_truths.add(truth_label)
            ax_spec.loglog(
                k_emu, spec_emu_mean, "-", label=f"{label}", linewidth=2, color=color
            )

            # Right panel: ratio
            # Interpolate emulator spectrum onto truth wavenumber grid
            ratio = np.interp(k_truth, k_emu, spec_emu_mean) / np.where(
                spec_truth_mean > 0, spec_truth_mean, np.nan
            )
            ax_ratio.semilogx(
                k_truth, ratio, "-", linewidth=2, color=color, label=f"{label}"
            )

        # Format left panel
        ax_spec.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=6)
        )
        ax_spec.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=True))
        ax_spec.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=12))
        ax_spec.xaxis.set_minor_formatter(plt.NullFormatter())
        ax_spec.set_ylabel("Power Spectral Density")
        ax_spec.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_spec.set_title(
            f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
        )
        ax_spec.tick_params(axis="both", labelsize=14)

        # Format right panel
        ax_ratio.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
        ax_ratio.axhspan(0.9, 1.1, color="gray", alpha=0.15)
        ax_ratio.set_ylim(ratio_ylim)
        ax_ratio.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=6)
        )
        ax_ratio.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=True))
        ax_ratio.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=12))
        ax_ratio.xaxis.set_minor_formatter(plt.NullFormatter())
        ax_ratio.set_ylabel("Emulator / OM4")
        ax_ratio.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_ratio.set_title(f"{region_name} — Ratio")
        ax_ratio.tick_params(axis="both", labelsize=14)

        # Only add x-labels on the bottom row
        if row == n_regions - 1:
            ax_spec.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
            ax_ratio.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")

    # Shared legend: row 1 = OM4 truth (from left panel), row 2 = emulators (from right panel)
    spec_handles, spec_labels = axes[0, 0].get_legend_handles_labels()
    ratio_handles, ratio_labels = axes[0, 1].get_legend_handles_labels()
    truth_h = [h for h, l in zip(spec_handles, spec_labels) if l.startswith("OM4")]
    truth_l = [l for l in spec_labels if l.startswith("OM4")]
    # Truth first, then all emulators (ratio panel has all emulator labels)
    ordered_handles = truth_h + ratio_handles
    ordered_labels = truth_l + ratio_labels
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=len(ordered_handles),
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"
    fig.suptitle(
        f"Isotropic Power Spectrum of {_display_name(var_to_eval)} ({depth_str})",
        y=1.01,
    )
    fig.tight_layout()
    plt.show()


def plot_ke_spectrum_comparison_together(
    emulators_dict,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    time_window=None,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_wavenumbers=[0.01, 0.02],
    show_ratios=True,
    colorbar_fix=False,
    region_name=None,
    ax=None,
):
    """
    Plots kinetic energy spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_u : str, default='uo'
        U-velocity variable name
    var_v : str, default='vo'
        V-velocity variable name
    lev_idx : int, default=0
        Level index for depth slice
    time_window : int, default=100
        Number of time steps to use
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for analysis
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for analysis
    target_wavenumbers : list, default=[0.01, 0.02]
        Wavenumbers at which to calculate and annotate ratios
    show_ratios : bool, default=True
        Whether to show ratio annotations at target wavenumbers
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    # print(f"\nGenerating Kinetic Energy Spectrum Comparison...")

    def get_ke_spectrum(ds_target, lev_i, t_window, x_sl, y_sl):
        """Helper function to compute KE spectrum for a dataset."""
        dx_mean = float(ds_target.dx.sel({"x": x_sl, "y": y_sl}).mean().values)
        dy_mean = float(ds_target.dy.sel({"x": x_sl, "y": y_sl}).mean().values)

        def process_var(var_name):
            data = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
            data = data.transpose("time", ...).sel({"x": x_sl, "y": y_sl})
            data_anom = data - data.mean(dim="time")
            return torch.as_tensor(data_anom.fillna(0).values)

        u_tensor = process_var(var_u)
        v_tensor = process_var(var_v)

        k_cent, spec_u = compute_isotropic_spectrum_torch(
            u_tensor,
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        _, spec_v = compute_isotropic_spectrum_torch(
            v_tensor,
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )

        spec_ke = 0.5 * (spec_u + spec_v)
        k_km = k_cent.cpu().numpy() * 1000 * 2 * np.pi
        return k_km, spec_ke.cpu().numpy()

    # Compute spectra for each truth-emulator pair
    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        truth_label = emu_data.get("truth_label", "Truth")

        # print(f"Calculating KE spectrum for {truth_label} (truth of {label})...")
        k_truth, spec_truth = get_ke_spectrum(
            ds_truth, lev_idx, time_window, lon_slice, lat_slice
        )

        # print(f"Calculating KE spectrum for {label}...")
        k_emu, spec_emu = get_ke_spectrum(
            ds_emu, lev_idx, time_window, lon_slice, lat_slice
        )

        all_spectra[label] = {
            "k_truth": k_truth,
            "spec_truth": spec_truth,
            "k_emu": k_emu,
            "spec_emu": spec_emu,
            "truth_label": truth_label,
        }

    # Plotting
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        truth_label = data["truth_label"]

        # Plot truth (dashed) — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data["k_truth"],
                data["spec_truth"].mean(0),
                "--",
                label=f"{truth_label}",
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid)
        ax.loglog(
            data["k_emu"],
            data["spec_emu"].mean(0),
            "-",
            label=f"{label}",
            linewidth=2,
            color=color,
        )

    # Add ratio annotations at target wavenumbers
    if show_ratios:
        for target_k in target_wavenumbers:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data["k_truth"] - target_k))
                idx_emu = np.argmin(np.abs(data["k_emu"] - target_k))
                actual_k = data["k_truth"][idx_truth]
                spec_truth_val = data["spec_truth"].mean(0)[idx_truth]
                spec_emu_val = data["spec_emu"].mean(0)[idx_emu]
                ratio = spec_emu_val / spec_truth_val if spec_truth_val > 0 else 0

                # print(f"{label} vs {data['truth_label']} at k={actual_k:.4f}: ratio={ratio:.3f}")

                color = emulators_dict[label]["color"]
                x_pos = actual_k * 1.1
                y_pos = spec_emu_val * (1.5**i)

                ax.text(
                    x_pos,
                    y_pos,
                    f"{label}/{data['truth_label']}={ratio:.3f}",
                    fontsize=12,
                    color=color,
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color
                    ),
                )

            ax.axvline(x=actual_k, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    # Format axes
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=6))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=True))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=12))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
    ax.set_ylabel("EKE Density")

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"

    if region_name:
        region_str = f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
    else:
        region_str = ""
    ax.set_title(region_str)

    ax.grid(True, which="both", alpha=0.3)

    if own_figure:
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_temporal_spectrum_comparison_together(
    emulators_dict,
    var_to_eval="thetao",
    lev_idx=0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies=None,
    show_ratios=True,
    colorbar_fix=False,
    region_name=None,
    ax=None,
):
    """
    Plots temporal power spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_to_eval : str, default='thetao'
        Variable name to evaluate
    lev_idx : int, default=0
        Level index for depth slice
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for spatial averaging
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for spatial averaging
    target_frequencies : list, optional
        Frequencies (in cycles/year) at which to calculate and annotate ratios.
        If None, defaults to [1.0, 4.0] (annual and seasonal cycles)
    show_ratios : bool, default=True
        Whether to show ratio annotations at target frequencies
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]

    # print(f"\nGenerating Temporal Power Spectrum Comparison for {var_to_eval}...")

    def get_temporal_spectrum(ds_target, var_name, lev_i, x_sl, y_sl):
        """Helper function to compute temporal spectrum using xrft, then average over space."""
        import xrft

        data = ds_target[var_name].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})

        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = time_vals[1] - time_vals[0]
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, "D")
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0
        dt_years = dt_days / 365.25

        # Remove seasonal cycle
        climatology = data.groupby("time.dayofyear").mean("time")
        data = data.groupby("time.dayofyear") - climatology

        # Prepare for xrft
        data = data.drop_vars("dayofyear").fillna(0.0)
        # Drop any coordinate variables whose dims include 'time' (e.g. 'lev')
        drop_coords = [
            c
            for c in data.coords
            if c not in data.dims and "time" in (data[c].dims or [])
        ]
        if drop_coords:
            data = data.drop_vars(drop_coords)
        data["time"] = np.arange(len(data.time)) * dt_years
        data = data.compute()

        power = xrft.power_spectrum(data, dim="time", detrend="linear", window="hann")
        psd = power.mean(dim=[d for d in power.dims if d != "freq_time"])

        freqs = psd.freq_time.values
        psd_vals = psd.values
        pos_mask = freqs > 0
        return freqs[pos_mask], psd_vals[pos_mask]

    # Compute spectra for each truth-emulator pair
    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        truth_label = emu_data.get("truth_label", "Truth")

        # print(f"Calculating temporal spectrum for {truth_label} (truth of {label})...")
        freq_truth, psd_truth = get_temporal_spectrum(
            ds_truth, var_to_eval, lev_idx, lon_slice, lat_slice
        )

        # print(f"Calculating temporal spectrum for {label}...")
        freq_emu, psd_emu = get_temporal_spectrum(
            ds_emu, var_to_eval, lev_idx, lon_slice, lat_slice
        )

        all_spectra[label] = {
            "freq_truth": freq_truth,
            "psd_truth": psd_truth,
            "freq_emu": freq_emu,
            "psd_emu": psd_emu,
            "truth_label": truth_label,
        }

    # Plotting
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        truth_label = data["truth_label"]

        # Plot truth (dashed), skip DC component — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data["freq_truth"][1:],
                data["psd_truth"][1:],
                "--",
                label=f"{truth_label}",
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid), skip DC component
        ax.loglog(
            data["freq_emu"][1:],
            data["psd_emu"][1:],
            "-",
            label=f"{label}",
            linewidth=2,
            color=color,
        )

    # Add ratio annotations at target frequencies
    if show_ratios and len(target_frequencies) > 0:
        for target_f in target_frequencies:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data["freq_truth"] - target_f))
                idx_emu = np.argmin(np.abs(data["freq_emu"] - target_f))
                actual_f = data["freq_truth"][idx_truth]
                psd_truth_val = data["psd_truth"][idx_truth]
                psd_emu_val = data["psd_emu"][idx_emu]
                ratio = psd_emu_val / psd_truth_val if psd_truth_val > 0 else 0

                period_years = 1.0 / actual_f if actual_f > 0 else np.inf
                print(
                    f"{label} vs {data['truth_label']} at f={actual_f:.4f} cy/yr (period {period_years:.2f}yr): ratio={ratio:.3f}"
                )

                color = emulators_dict[label]["color"]
                x_pos = actual_f * 1.2
                y_pos = psd_emu_val * (1.5**i)

                ax.text(
                    x_pos,
                    y_pos,
                    f"{label}/{data['truth_label']}={ratio:.3f}",
                    fontsize=12,
                    color=color,
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color
                    ),
                )

            ax.axvline(x=actual_f, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    # Format axes
    ax.set_xlabel("Frequency (cycles/year)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_xlim(right=10)

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"

    if region_name:
        region_str = f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
    else:
        region_str = ""
    ax.set_title(region_str)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    if own_figure:
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_ke_temporal_spectrum_comparison_together(
    emulators_dict,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    lon_slice=slice(180, 243),
    lat_slice=slice(-40, 35),
    target_frequencies=None,
    show_ratios=True,
    colorbar_fix=False,
    region_name=None,
    ax=None,
):
    """
    Plots temporal kinetic energy spectrum comparison where each emulator uses its own corresponding truth.

    Parameters:
    -----------
    emulators_dict : dict
        Dictionary of emulators with structure:
        {
            'label': {
                'ds': xr.Dataset,          # emulator dataset
                'color': str,              # matplotlib color
                'truth': xr.Dataset,       # corresponding ground truth dataset
                'truth_label': str         # label for the truth dataset
            }
        }
    var_u : str, default='uo'
        U-velocity variable name
    var_v : str, default='vo'
        V-velocity variable name
    lev_idx : int, default=0
        Level index for depth slice
    lon_slice : slice, default=slice(180, 243)
        Longitude slice for spatial averaging
    lat_slice : slice, default=slice(-40, 35)
        Latitude slice for spatial averaging
    target_frequencies : list, optional
        Frequencies (in cycles/year) at which to calculate and annotate ratios.
        If None, defaults to [1.0, 4.0] (annual and seasonal cycles)
    show_ratios : bool, default=True
        Whether to show ratio annotations at target frequencies
    colorbar_fix : bool, default=False
        Not used, kept for consistency

    Returns:
    --------
    None (displays plot)
    """

    if target_frequencies is None:
        target_frequencies = [1.0, 4.0]

    # print(f"\nGenerating Temporal EKE Power Spectrum Comparison...")

    def get_ke_temporal_spectrum(ds_target, lev_i, x_sl, y_sl):
        """Helper function to compute temporal KE spectrum using xrft, then average over space."""
        import xrft

        u_data = ds_target[var_u].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})
        v_data = ds_target[var_v].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})

        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = time_vals[1] - time_vals[0]
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, "D")
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0
        dt_years = dt_days / 365.25

        # Remove seasonal cycle
        u_data = u_data.groupby("time.dayofyear") - u_data.groupby(
            "time.dayofyear"
        ).mean("time")
        v_data = v_data.groupby("time.dayofyear") - v_data.groupby(
            "time.dayofyear"
        ).mean("time")

        # Prepare for xrft
        u_data = u_data.drop_vars("dayofyear").fillna(0.0)
        v_data = v_data.drop_vars("dayofyear").fillna(0.0)
        # Drop any coordinate variables whose dims include 'time' (e.g. 'lev')
        drop_coords = [
            c
            for c in u_data.coords
            if c not in u_data.dims and "time" in (u_data[c].dims or [])
        ]
        if drop_coords:
            u_data = u_data.drop_vars(drop_coords)
            v_data = v_data.drop_vars(drop_coords)
        u_data["time"] = np.arange(len(u_data.time)) * dt_years
        v_data["time"] = np.arange(len(v_data.time)) * dt_years
        u_data = u_data.compute()
        v_data = v_data.compute()

        power_u = xrft.power_spectrum(
            u_data, dim="time", detrend="linear", window="hann"
        )
        power_v = xrft.power_spectrum(
            v_data, dim="time", detrend="linear", window="hann"
        )
        power_ke = 0.5 * (power_u + power_v)

        psd = power_ke.mean(dim=[d for d in power_ke.dims if d != "freq_time"])
        freqs = psd.freq_time.values
        psd_vals = psd.values
        pos_mask = freqs > 0
        return freqs[pos_mask], psd_vals[pos_mask]

    # Compute spectra for each truth-emulator pair
    all_spectra = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        truth_label = emu_data.get("truth_label", "Truth")

        # print(f"Calculating temporal KE spectrum for {truth_label} (truth of {label})...")
        freq_truth, psd_truth = get_ke_temporal_spectrum(
            ds_truth, lev_idx, lon_slice, lat_slice
        )

        # print(f"Calculating temporal KE spectrum for {label}...")
        freq_emu, psd_emu = get_ke_temporal_spectrum(
            ds_emu, lev_idx, lon_slice, lat_slice
        )

        all_spectra[label] = {
            "freq_truth": freq_truth,
            "psd_truth": psd_truth,
            "freq_emu": freq_emu,
            "psd_emu": psd_emu,
            "truth_label": truth_label,
        }

    # Plotting
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4))

    plotted_truths = set()
    for label, data in all_spectra.items():
        color = emulators_dict[label]["color"]
        truth_label = data["truth_label"]

        # Plot truth (dashed), skip DC component — skip if already plotted for this truth_label
        if truth_label not in plotted_truths:
            ax.loglog(
                data["freq_truth"][1:],
                data["psd_truth"][1:],
                "--",
                label=f"{truth_label}",
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            plotted_truths.add(truth_label)
        # Plot emulator (solid), skip DC component
        ax.loglog(
            data["freq_emu"][1:],
            data["psd_emu"][1:],
            "-",
            label=f"{label}",
            linewidth=2,
            color=color,
        )

    # Add ratio annotations at target frequencies
    if show_ratios and len(target_frequencies) > 0:
        for target_f in target_frequencies:
            for i, (label, data) in enumerate(all_spectra.items()):
                idx_truth = np.argmin(np.abs(data["freq_truth"] - target_f))
                idx_emu = np.argmin(np.abs(data["freq_emu"] - target_f))
                actual_f = data["freq_truth"][idx_truth]
                psd_truth_val = data["psd_truth"][idx_truth]
                psd_emu_val = data["psd_emu"][idx_emu]
                ratio = psd_emu_val / psd_truth_val if psd_truth_val > 0 else 0

                period_years = 1.0 / actual_f if actual_f > 0 else np.inf
                print(
                    f"{label} vs {data['truth_label']} at f={actual_f:.4f} cy/yr (period {period_years:.2f}yr): ratio={ratio:.3f}"
                )

                color = emulators_dict[label]["color"]
                x_pos = actual_f * 1.2
                y_pos = psd_emu_val * (1.5**i)

                ax.text(
                    x_pos,
                    y_pos,
                    f"{label}/{data['truth_label']}={ratio:.3f}",
                    fontsize=12,
                    color=color,
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color
                    ),
                )

            ax.axvline(x=actual_f, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    # Format axes
    ax.set_xlabel("Frequency (cycles/year)")
    ax.set_ylabel("EKE Power Spectral Density")
    ax.set_xlim(right=10)

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"

    if region_name:
        region_str = f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
    else:
        region_str = ""
    ax.set_title(region_str)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    if own_figure:
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_ke_spectrum_with_ratio(
    emulators_dict,
    spectrum_regions,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    time_window=None,
    ratio_ylim=(0, 1.5),
):
    """
    KE spectrum + ratio for multiple regions (N rows × 2 cols).
    Left: OM4 truth only. Right: emulator/OM4 ratio.
    """
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    def get_ke_spectrum(ds_target, lev_i, t_window, x_sl, y_sl):
        dx_mean = float(ds_target.dx.sel({"x": x_sl, "y": y_sl}).mean().values)
        dy_mean = float(ds_target.dy.sel({"x": x_sl, "y": y_sl}).mean().values)

        def process_var(var_name):
            data = ds_target[var_name].isel(lev=lev_i, time=slice(None, t_window))
            data = data.transpose("time", ...).sel({"x": x_sl, "y": y_sl})
            data_anom = data - data.mean(dim="time")
            return torch.as_tensor(data_anom.fillna(0).values)

        u_tensor = process_var(var_u)
        v_tensor = process_var(var_v)
        k_cent, spec_u = compute_isotropic_spectrum_torch(
            u_tensor,
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        _, spec_v = compute_isotropic_spectrum_torch(
            v_tensor,
            dx=dx_mean,
            dy=dy_mean,
            n_factor=2,
            detrend="linear",
            window="hann",
            cutoff_before_bins=False,
        )
        spec_ke = 0.5 * (spec_u + spec_v)
        return k_cent.cpu().numpy() * 1000 * 2 * np.pi, spec_ke.cpu().numpy()

    n_regions = len(spectrum_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(12, 3.5 * n_regions))
    if n_regions == 1:
        axes = axes[np.newaxis, :]

    for row, (region_name, lon_slice, lat_slice) in enumerate(spectrum_regions):
        ax_spec = axes[row, 0]
        ax_ratio = axes[row, 1]

        plotted_truths = set()
        for label, emu_data in emulators_dict.items():
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]

            k_truth, spec_truth = get_ke_spectrum(
                ds_truth, lev_idx, time_window, lon_slice, lat_slice
            )
            k_emu, spec_emu = get_ke_spectrum(
                ds_emu, lev_idx, time_window, lon_slice, lat_slice
            )

            spec_truth_mean = spec_truth.mean(0)
            spec_emu_mean = spec_emu.mean(0)

            if truth_label not in plotted_truths:
                ax_spec.loglog(
                    k_truth,
                    spec_truth_mean,
                    "--",
                    label=f"{truth_label}",
                    linewidth=2,
                    color=color,
                    alpha=0.7,
                )
                plotted_truths.add(truth_label)
            ax_spec.loglog(
                k_emu, spec_emu_mean, "-", label=f"{label}", linewidth=2, color=color
            )

            ratio = np.interp(k_truth, k_emu, spec_emu_mean) / np.where(
                spec_truth_mean > 0, spec_truth_mean, np.nan
            )
            ax_ratio.semilogx(
                k_truth, ratio, "-", linewidth=2, color=color, label=f"{label}"
            )

        ax_spec.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=6)
        )
        ax_spec.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=True))
        ax_spec.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=12))
        ax_spec.xaxis.set_minor_formatter(plt.NullFormatter())
        ax_spec.set_ylabel("EKE Density")
        ax_spec.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_spec.set_title(
            f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
        )
        ax_spec.tick_params(axis="both", labelsize=14)

        ax_ratio.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
        ax_ratio.axhspan(0.9, 1.1, color="gray", alpha=0.15)
        ax_ratio.set_ylim(ratio_ylim)
        ax_ratio.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=6)
        )
        ax_ratio.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=True))
        ax_ratio.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=12))
        ax_ratio.xaxis.set_minor_formatter(plt.NullFormatter())
        ax_ratio.set_ylabel("Emulator / OM4")
        ax_ratio.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_ratio.set_title(f"{region_name} — Ratio")
        ax_ratio.tick_params(axis="both", labelsize=14)

        if row == n_regions - 1:
            ax_spec.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")
            ax_ratio.set_xlabel(r"Wavenumber $\kappa$ (km$^{-1}$)")

    all_handles, all_labels = axes[0, 0].get_legend_handles_labels()
    oh, ol, nc = reorder_legend_paired(all_handles, all_labels)
    fig.legend(
        oh, ol, loc="lower center", ncol=nc, bbox_to_anchor=(0.5, -0.03), frameon=False
    )

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"
    fig.suptitle(f"EKE Spectrum (Anomalies, {depth_str})", y=1.01)
    fig.tight_layout()
    plt.show()


def plot_temporal_spectrum_with_ratio(
    emulators_dict,
    spectrum_regions,
    var_to_eval="thetao",
    lev_idx=0,
    ratio_ylim=(0, 1.5),
):
    """
    Temporal spectrum + ratio for multiple regions (N rows × 2 cols).
    Left: OM4 truth only. Right: emulator/OM4 ratio.
    """

    def get_temporal_spectrum(ds_target, var_name, lev_i, x_sl, y_sl):
        import xrft

        data = ds_target[var_name].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})
        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = time_vals[1] - time_vals[0]
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, "D")
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0
        dt_years = dt_days / 365.25
        climatology = data.groupby("time.dayofyear").mean("time")
        data = data.groupby("time.dayofyear") - climatology
        data = data.drop_vars("dayofyear").fillna(0.0)
        drop_coords = [
            c
            for c in data.coords
            if c not in data.dims and "time" in (data[c].dims or [])
        ]
        if drop_coords:
            data = data.drop_vars(drop_coords)
        data["time"] = np.arange(len(data.time)) * dt_years
        data = data.compute()
        power = xrft.power_spectrum(data, dim="time", detrend="linear", window="hann")
        psd = power.mean(dim=[d for d in power.dims if d != "freq_time"])
        freqs = psd.freq_time.values
        psd_vals = psd.values
        pos_mask = freqs > 0
        return freqs[pos_mask], psd_vals[pos_mask]

    n_regions = len(spectrum_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(12, 3.5 * n_regions))
    if n_regions == 1:
        axes = axes[np.newaxis, :]

    for row, (region_name, lon_slice, lat_slice) in enumerate(spectrum_regions):
        ax_spec = axes[row, 0]
        ax_ratio = axes[row, 1]

        plotted_truths = set()
        for label, emu_data in emulators_dict.items():
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]

            freq_truth, psd_truth = get_temporal_spectrum(
                ds_truth, var_to_eval, lev_idx, lon_slice, lat_slice
            )
            freq_emu, psd_emu = get_temporal_spectrum(
                ds_emu, var_to_eval, lev_idx, lon_slice, lat_slice
            )

            if truth_label not in plotted_truths:
                ax_spec.loglog(
                    freq_truth[1:],
                    psd_truth[1:],
                    "--",
                    label=f"{truth_label}",
                    linewidth=2,
                    color=color,
                    alpha=0.7,
                )
                plotted_truths.add(truth_label)
            ax_spec.loglog(
                freq_emu[1:],
                psd_emu[1:],
                "-",
                label=f"{label}",
                linewidth=2,
                color=color,
            )

            ratio = np.interp(freq_truth[1:], freq_emu[1:], psd_emu[1:]) / np.where(
                psd_truth[1:] > 0, psd_truth[1:], np.nan
            )
            ax_ratio.semilogx(
                freq_truth[1:], ratio, "-", linewidth=2, color=color, label=f"{label}"
            )

        ax_spec.set_ylabel("Power Spectral Density")
        ax_spec.set_xlim(right=10)
        ax_spec.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_spec.set_title(
            f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
        )
        ax_spec.tick_params(axis="both", labelsize=14)

        ax_ratio.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
        ax_ratio.axhspan(0.9, 1.1, color="gray", alpha=0.15)
        ax_ratio.set_ylim(ratio_ylim)
        ax_ratio.set_xlim(right=10)
        ax_ratio.set_ylabel("Emulator / OM4")
        ax_ratio.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_ratio.set_title(f"{region_name} — Ratio")
        ax_ratio.tick_params(axis="both", labelsize=14)

        if row == n_regions - 1:
            ax_spec.set_xlabel("Frequency (cycles/year)")
            ax_ratio.set_xlabel("Frequency (cycles/year)")

    spec_handles, spec_labels = axes[0, 0].get_legend_handles_labels()
    ratio_handles, ratio_labels = axes[0, 1].get_legend_handles_labels()
    truth_h = [h for h, l in zip(spec_handles, spec_labels) if l.startswith("OM4")]
    truth_l = [l for l in spec_labels if l.startswith("OM4")]
    # Truth first, then all emulators (ratio panel has all emulator labels)
    ordered_handles = truth_h + ratio_handles
    ordered_labels = truth_l + ratio_labels
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=len(ordered_handles),
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"
    fig.suptitle(
        f"Temporal Power Spectrum of {_display_name(var_to_eval)} ({depth_str})", y=1.01
    )
    fig.tight_layout()
    plt.show()


def plot_ke_temporal_spectrum_with_ratio(
    emulators_dict,
    spectrum_regions,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    ratio_ylim=(0, 1.5),
):
    """
    KE temporal spectrum + ratio for multiple regions (N rows × 2 cols).
    Left: OM4 truth only. Right: emulator/OM4 ratio.
    """

    def get_ke_temporal_spectrum(ds_target, lev_i, x_sl, y_sl):
        import xrft

        u_data = ds_target[var_u].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})
        v_data = ds_target[var_v].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})
        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = time_vals[1] - time_vals[0]
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, "D")
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except:
            dt_days = 1.0
        dt_years = dt_days / 365.25
        u_data = u_data.groupby("time.dayofyear") - u_data.groupby(
            "time.dayofyear"
        ).mean("time")
        v_data = v_data.groupby("time.dayofyear") - v_data.groupby(
            "time.dayofyear"
        ).mean("time")
        u_data = u_data.drop_vars("dayofyear").fillna(0.0)
        v_data = v_data.drop_vars("dayofyear").fillna(0.0)
        drop_coords = [
            c
            for c in u_data.coords
            if c not in u_data.dims and "time" in (u_data[c].dims or [])
        ]
        if drop_coords:
            u_data = u_data.drop_vars(drop_coords)
            v_data = v_data.drop_vars(drop_coords)
        u_data["time"] = np.arange(len(u_data.time)) * dt_years
        v_data["time"] = np.arange(len(v_data.time)) * dt_years
        u_data = u_data.compute()
        v_data = v_data.compute()
        power_u = xrft.power_spectrum(
            u_data, dim="time", detrend="linear", window="hann"
        )
        power_v = xrft.power_spectrum(
            v_data, dim="time", detrend="linear", window="hann"
        )
        power_ke = 0.5 * (power_u + power_v)
        psd = power_ke.mean(dim=[d for d in power_ke.dims if d != "freq_time"])
        freqs = psd.freq_time.values
        psd_vals = psd.values
        pos_mask = freqs > 0
        return freqs[pos_mask], psd_vals[pos_mask]

    n_regions = len(spectrum_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(12, 3.5 * n_regions))
    if n_regions == 1:
        axes = axes[np.newaxis, :]

    for row, (region_name, lon_slice, lat_slice) in enumerate(spectrum_regions):
        ax_spec = axes[row, 0]
        ax_ratio = axes[row, 1]

        plotted_truths = set()
        for label, emu_data in emulators_dict.items():
            color = emu_data["color"]
            truth_label = emu_data.get("truth_label", "Truth")
            ds_truth = emu_data["truth"]
            ds_emu = emu_data["ds"]

            freq_truth, psd_truth = get_ke_temporal_spectrum(
                ds_truth, lev_idx, lon_slice, lat_slice
            )
            freq_emu, psd_emu = get_ke_temporal_spectrum(
                ds_emu, lev_idx, lon_slice, lat_slice
            )

            if truth_label not in plotted_truths:
                ax_spec.loglog(
                    freq_truth[1:],
                    psd_truth[1:],
                    "--",
                    label=f"{truth_label}",
                    linewidth=2,
                    color=color,
                    alpha=0.7,
                )
                plotted_truths.add(truth_label)
            ax_spec.loglog(
                freq_emu[1:],
                psd_emu[1:],
                "-",
                label=f"{label}",
                linewidth=2,
                color=color,
            )

            ratio = np.interp(freq_truth[1:], freq_emu[1:], psd_emu[1:]) / np.where(
                psd_truth[1:] > 0, psd_truth[1:], np.nan
            )
            ax_ratio.semilogx(
                freq_truth[1:], ratio, "-", linewidth=2, color=color, label=f"{label}"
            )

        ax_spec.set_ylabel("EKE Power Spectral Density")
        ax_spec.set_xlim(right=10)
        ax_spec.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_spec.set_title(
            f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
        )
        ax_spec.tick_params(axis="both", labelsize=14)

        ax_ratio.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5)
        ax_ratio.axhspan(0.9, 1.1, color="gray", alpha=0.15)
        ax_ratio.set_ylim(ratio_ylim)
        ax_ratio.set_xlim(right=10)
        ax_ratio.set_ylabel("Emulator / OM4")
        ax_ratio.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_ratio.set_title(f"{region_name} — Ratio")
        ax_ratio.tick_params(axis="both", labelsize=14)

        if row == n_regions - 1:
            ax_spec.set_xlabel("Frequency (cycles/year)")
            ax_ratio.set_xlabel("Frequency (cycles/year)")

    all_handles, all_labels = axes[0, 0].get_legend_handles_labels()
    oh, ol, nc = reorder_legend_paired(all_handles, all_labels)
    fig.legend(
        oh, ol, loc="lower center", ncol=nc, bbox_to_anchor=(0.5, -0.03), frameon=False
    )

    first_label = list(emulators_dict.keys())[0]
    depth_value = emulators_dict[first_label]["truth"].lev.values[lev_idx]
    depth_str = "Surface" if lev_idx == 0 else f"{depth_value}m"
    fig.suptitle(f"Temporal EKE Power Spectrum ({depth_str})", y=1.01)
    fig.tight_layout()
    plt.show()


# ============================================================
# Autocorrelation Function (ACF) Evaluation
# ============================================================


def plot_acf_comparison_together(
    emulators_dict,
    var_to_eval="zos",
    lev_idx=None,
    lon_slice=slice(300, 320),
    lat_slice=slice(25, 45),
    max_lag_years=5,
    remove_seasonal=True,
    region_name=None,
    ax=None,
):
    """
    Plots autocorrelation function comparison for a scalar variable (e.g. SSH, uo, vo).

    Parameters
    ----------
    emulators_dict : dict
        Standard emulator dictionary with 'ds', 'color', 'truth', 'truth_label'.
    var_to_eval : str
        Variable name ('zos' for SSH, or 'uo'/'vo' with lev_idx).
    lev_idx : int or None
        Depth level index.  Required for 3-D variables, ignored for 'zos'.
    lon_slice, lat_slice : slice
        Coordinate slices defining the region.
    max_lag_years : float
        Maximum lag to compute, in years.
    remove_seasonal : bool
        Whether to remove the seasonal cycle before computing ACF.
    region_name : str or None
        Optional region name for the title.
    """
    from scipy import signal

    def get_acf(ds_target, var_name, lev_i, x_sl, y_sl, max_lag):
        if var_name == "zos":
            data = ds_target[var_name].sel({"x": x_sl, "y": y_sl})
        else:
            data = ds_target[var_name].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})

        n_time = data.sizes["time"]

        # Determine time step in days
        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = time_vals[1] - time_vals[0]
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, "D")
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except Exception:
            dt_days = 1.0

        # Remove seasonal cycle using xarray groupby (handles calendar correctly)
        if remove_seasonal:
            climatology = data.groupby("time.dayofyear").mean("time")
            data = data.groupby("time.dayofyear") - climatology

        data_np = np.nan_to_num(data.values, nan=0.0)  # (time, y, x)

        # Demean each spatial point
        data_np = data_np - data_np.mean(axis=0, keepdims=True)

        # Compute unnormalized autocorrelation via FFT for all spatial points at once,
        # then sum across space (variance-weighted) and normalize by zero-lag value.
        max_lag_steps = min(max_lag, n_time - 1)
        ny, nx = data_np.shape[1], data_np.shape[2]
        acf_sum = np.zeros(max_lag_steps + 1)
        for j in range(ny):
            for i in range(nx):
                ts = data_np[:, j, i]
                if np.all(ts == 0):
                    continue
                full = signal.correlate(ts, ts, mode="full")
                mid = len(full) // 2
                acf_sum += full[mid : mid + max_lag_steps + 1]

        if acf_sum[0] > 0:
            acf_norm = acf_sum / acf_sum[0]
        else:
            acf_norm = np.full(max_lag_steps + 1, np.nan)

        lags_days = np.arange(max_lag_steps + 1) * dt_days
        return lags_days, acf_norm

    # Determine max lag in time steps
    first_label = list(emulators_dict.keys())[0]
    first_ds = emulators_dict[first_label]["truth"]
    try:
        time_vals = first_ds.time.values
        delta_t = time_vals[1] - time_vals[0]
        if isinstance(delta_t, np.timedelta64):
            dt_days = float(delta_t) / np.timedelta64(1, "D")
        else:
            dt_days = delta_t.days + delta_t.seconds / 86400.0
    except Exception:
        dt_days = 1.0
    max_lag_steps = int(max_lag_years * 365.25 / dt_days)

    all_acf = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        truth_label = emu_data.get("truth_label", "Truth")

        lags_truth, acf_truth = get_acf(
            ds_truth, var_to_eval, lev_idx, lon_slice, lat_slice, max_lag_steps
        )
        lags_emu, acf_emu = get_acf(
            ds_emu, var_to_eval, lev_idx, lon_slice, lat_slice, max_lag_steps
        )

        all_acf[label] = {
            "lags_truth": lags_truth,
            "acf_truth": acf_truth,
            "lags_emu": lags_emu,
            "acf_emu": acf_emu,
            "truth_label": truth_label,
        }

    # Plotting
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(7, 4))

    plotted_truths = set()
    for label, data in all_acf.items():
        color = emulators_dict[label]["color"]
        truth_label = data["truth_label"]
        lags_years_truth = data["lags_truth"] / 365.25
        lags_years_emu = data["lags_emu"] / 365.25

        if truth_label not in plotted_truths:
            ax.plot(
                lags_years_truth,
                data["acf_truth"],
                "--",
                label=truth_label,
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            plotted_truths.add(truth_label)

        ax.plot(
            lags_years_emu, data["acf_emu"], "-", label=label, linewidth=2, color=color
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks([0, 0.25, 0.5, 0.75])

    if region_name:
        region_str = f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
    else:
        region_str = ""
    ax.set_title(region_str)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if own_figure:
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_velocity_acf_comparison_together(
    emulators_dict,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    lon_slice=slice(300, 320),
    lat_slice=slice(25, 45),
    max_lag_years=5,
    remove_seasonal=True,
    region_name=None,
    ax=None,
):
    """
    Plots ACF of velocity anomalies: ACF_vel = 0.5*(ACF_u' + ACF_v').
    This is the Fourier pair of the temporal KE spectrum via Wiener-Khinchin theorem.

    Parameters
    ----------
    emulators_dict : dict
        Standard emulator dictionary.
    var_u, var_v : str
        Variable names for zonal and meridional velocity.
    lev_idx : int
        Depth level index.
    lon_slice, lat_slice : slice
        Coordinate slices defining the region.
    max_lag_years : float
        Maximum lag to compute, in years.
    remove_seasonal : bool
        Whether to remove the seasonal cycle before computing ACF.
    region_name : str or None
        Optional region name for the title.
    """
    from scipy import signal

    def _preprocess_component(data_np):
        """Demean a single velocity component (seasonal cycle already removed)."""
        return data_np - data_np.mean(axis=0, keepdims=True)

    def _spatial_mean_acf(data_np, max_lag_steps):
        """Compute unnormalized autocorrelation via FFT at each spatial point,
        sum across space (variance-weighted), then normalize by zero-lag."""
        # Demean each spatial point
        data_np = data_np - data_np.mean(axis=0, keepdims=True)
        ny, nx = data_np.shape[1], data_np.shape[2]
        acf_sum = np.zeros(max_lag_steps + 1)
        for j in range(ny):
            for i in range(nx):
                ts = data_np[:, j, i]
                if np.all(ts == 0):
                    continue
                full = signal.correlate(ts, ts, mode="full")
                mid = len(full) // 2
                acf_sum += full[mid : mid + max_lag_steps + 1]
        if acf_sum[0] > 0:
            return acf_sum / acf_sum[0]
        return np.full(max_lag_steps + 1, np.nan)

    def get_velocity_acf(ds_target, lev_i, x_sl, y_sl, max_lag):
        u_da = ds_target[var_u].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})
        v_da = ds_target[var_v].isel(lev=lev_i).sel({"x": x_sl, "y": y_sl})

        n_time = u_da.sizes["time"]

        try:
            time_vals = ds_target.time.values
            if len(time_vals) > 1:
                delta_t = time_vals[1] - time_vals[0]
                if isinstance(delta_t, np.timedelta64):
                    dt_days = float(delta_t) / np.timedelta64(1, "D")
                else:
                    dt_days = delta_t.days + delta_t.seconds / 86400.0
            else:
                dt_days = 1.0
        except Exception:
            dt_days = 1.0

        # Remove seasonal cycle using xarray groupby (handles calendar correctly)
        if remove_seasonal:
            u_da = u_da.groupby("time.dayofyear") - u_da.groupby("time.dayofyear").mean(
                "time"
            )
            v_da = v_da.groupby("time.dayofyear") - v_da.groupby("time.dayofyear").mean(
                "time"
            )

        u = np.nan_to_num(u_da.values, nan=0.0)
        v = np.nan_to_num(v_da.values, nan=0.0)

        u = _preprocess_component(u)
        v = _preprocess_component(v)

        max_lag_steps = min(max_lag, n_time - 1)
        acf_u = _spatial_mean_acf(u, max_lag_steps)
        acf_v = _spatial_mean_acf(v, max_lag_steps)
        acf_vel = 0.5 * (acf_u + acf_v)

        lags_days = np.arange(max_lag_steps + 1) * dt_days
        return lags_days, acf_vel

    # Determine max lag in time steps
    first_label = list(emulators_dict.keys())[0]
    first_ds = emulators_dict[first_label]["truth"]
    try:
        time_vals = first_ds.time.values
        delta_t = time_vals[1] - time_vals[0]
        if isinstance(delta_t, np.timedelta64):
            dt_days = float(delta_t) / np.timedelta64(1, "D")
        else:
            dt_days = delta_t.days + delta_t.seconds / 86400.0
    except Exception:
        dt_days = 1.0
    max_lag_steps = int(max_lag_years * 365.25 / dt_days)

    all_acf = {}
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        truth_label = emu_data.get("truth_label", "Truth")

        lags_truth, acf_truth = get_velocity_acf(
            ds_truth, lev_idx, lon_slice, lat_slice, max_lag_steps
        )
        lags_emu, acf_emu = get_velocity_acf(
            ds_emu, lev_idx, lon_slice, lat_slice, max_lag_steps
        )

        all_acf[label] = {
            "lags_truth": lags_truth,
            "acf_truth": acf_truth,
            "lags_emu": lags_emu,
            "acf_emu": acf_emu,
            "truth_label": truth_label,
        }

    # Plotting
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(7, 4))

    plotted_truths = set()
    for label, data in all_acf.items():
        color = emulators_dict[label]["color"]
        truth_label = data["truth_label"]
        lags_years_truth = data["lags_truth"] / 365.25
        lags_years_emu = data["lags_emu"] / 365.25

        if truth_label not in plotted_truths:
            ax.plot(
                lags_years_truth,
                data["acf_truth"],
                "--",
                label=truth_label,
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            plotted_truths.add(truth_label)

        ax.plot(
            lags_years_emu, data["acf_emu"], "-", label=label, linewidth=2, color=color
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks([0, 0.25, 0.5, 0.75])

    if region_name:
        region_str = f"{region_name} [{lon_slice.start}°–{lon_slice.stop}°E, {lat_slice.start}°–{lat_slice.stop}°N]"
    else:
        region_str = ""
    ax.set_title(region_str)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if own_figure:
        ax.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# Temporal Variance Numerical Evaluation
# ============================================================


def _select_region(field, area, lon_slice=None, lat_slice=None):
    """Select a lon/lat sub-region. Returns (field_sub, area_sub).
    For Global (both slices None), return as-is."""
    if lon_slice is not None and lat_slice is not None:
        f = field.sel(x=lon_slice, y=lat_slice)
        a = area.sel(x=lon_slice, y=lat_slice)
        return f, a
    return field, area


def _area_weighted_mean_2d(field, area):
    """Area-weighted mean over (y, x)."""
    w = area.where(field.notnull())
    return float((field * w).sum(["y", "x"]) / w.sum(["y", "x"]))


def _area_weighted_rmse_2d(truth_map, pred_map, area):
    """Area-weighted RMSE between two 2D maps."""
    diff2 = (truth_map - pred_map) ** 2
    w = area.where(diff2.notnull())
    return float(np.sqrt((diff2 * w).sum(["y", "x"]) / w.sum(["y", "x"])))


def _area_weighted_pearson_2d(truth_map, pred_map, area):
    """Area-weighted Pearson correlation between two 2D maps."""
    valid = truth_map.notnull() & pred_map.notnull()
    w = area.where(valid)
    w_sum = w.sum(["y", "x"])

    t_mean = (truth_map * w).sum(["y", "x"]) / w_sum
    p_mean = (pred_map * w).sum(["y", "x"]) / w_sum

    t_anom = truth_map - t_mean
    p_anom = pred_map - p_mean

    cov = (t_anom * p_anom * w).sum(["y", "x"]) / w_sum
    std_t = np.sqrt((t_anom**2 * w).sum(["y", "x"]) / w_sum)
    std_p = np.sqrt((p_anom**2 * w).sum(["y", "x"]) / w_sum)

    return float(cov / (std_t * std_p))


def _detrend_linear_xr(da, dim="time"):
    """Remove linear trend along a dimension for each grid point (xarray)."""
    t = np.arange(da.sizes[dim], dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    da_mean = da.mean(dim)
    slope = ((da - da_mean) * xr.DataArray(t - t_mean, dims=[dim])).sum(dim) / t_var
    trend = slope * xr.DataArray(t - t_mean, dims=[dim]) + da_mean
    return da - trend


def compute_temporal_variance_metrics(
    emulators_dict,
    variable_name,
    spectrum_regions,
    depth_slices=None,
):
    """
    Compute 4 temporal-variance numerical metrics for each region and depth slice.

    Metrics (per region, per depth slice):
        1. Variance RMSE:  area-weighted RMSE of (var_truth - var_emu) maps
        2. Detrended RMSE: detrend both, per-gridpoint temporal RMSE, area-weighted mean
        3. Direct RMSE:    per-gridpoint temporal RMSE, area-weighted mean
        4. Variance Correlation: area-weighted Pearson corr of variance maps

    Args:
        emulators_dict: {label: {'ds', 'truth', 'truth_label', 'color'}}
        variable_name:  e.g. 'thetao', 'so'
        spectrum_regions: list of (name, lon_slice, lat_slice) tuples
        depth_slices:   list of {'min', 'max', 'title'}, default DEPTH_SLICES

    Returns:
        dict  {emulator_label: pandas.DataFrame}
        DataFrame columns: Depth, Region, Variance RMSE, Detrended RMSE,
                           Direct RMSE, Variance Correlation
    """
    import pandas as pd

    if depth_slices is None:
        depth_slices = DEPTH_SLICES

    # Build region list: Global + the 4 named regions
    regions = [("Global", None, None)] + [
        (name, lon_sl, lat_sl) for name, lon_sl, lat_sl in spectrum_regions
    ]

    results = {}

    for emu_idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        area = ds_truth["areacello"]

        dz = ds_truth["dz"]
        wet_mask = (ds_truth[variable_name].isel(time=0) * 0 + 1).compute()

        truth_field = ds_truth[variable_name]  # (time, y, x, lev)
        emu_field = ds_emu[variable_name]

        rows = []
        for dslice in depth_slices:
            d_title = dslice["title"]
            d_min, d_max = dslice["min"], dslice["max"]

            # Depth-average to (time, y, x)
            def _depth_avg(field_3d, _dmin=d_min, _dmax=d_max):
                f = field_3d.sel(lev=slice(_dmin, _dmax))
                dz_s = dz.sel(lev=slice(_dmin, _dmax))
                w_s = wet_mask.sel(lev=slice(_dmin, _dmax))
                num = (f * dz_s * w_s).sum("lev")
                den = (dz_s * w_s).sum("lev")
                return (num / den.where(den != 0)).compute()

            truth_2d = _depth_avg(truth_field)  # (time, y, x)
            emu_2d = _depth_avg(emu_field)

            # Remove seasonal cycle (dayofyear climatology)
            truth_anom = (
                (
                    truth_2d.groupby("time.dayofyear")
                    - truth_2d.groupby("time.dayofyear").mean("time")
                )
                .drop_vars("dayofyear")
                .fillna(0.0)
            )
            emu_anom = (
                (
                    emu_2d.groupby("time.dayofyear")
                    - emu_2d.groupby("time.dayofyear").mean("time")
                )
                .drop_vars("dayofyear")
                .fillna(0.0)
            )

            # Temporal variance maps: (y, x) — on anomalies
            var_truth = truth_anom.var("time").compute()
            var_emu = emu_anom.var("time").compute()

            # Per-gridpoint temporal RMSE map: (y, x) — on anomalies
            direct_rmse_map = np.sqrt(
                ((truth_anom - emu_anom) ** 2).mean("time")
            ).compute()

            # Detrended per-gridpoint temporal RMSE map: (y, x) — detrend anomalies
            truth_dt = _detrend_linear_xr(truth_anom, dim="time")
            emu_dt = _detrend_linear_xr(emu_anom, dim="time")
            detrended_rmse_map = np.sqrt(
                ((truth_dt - emu_dt) ** 2).mean("time")
            ).compute()

            for region_name, lon_sl, lat_sl in regions:
                vt, a = _select_region(var_truth, area, lon_sl, lat_sl)
                ve, _ = _select_region(var_emu, area, lon_sl, lat_sl)
                dr, _ = _select_region(direct_rmse_map, area, lon_sl, lat_sl)
                dtr, _ = _select_region(detrended_rmse_map, area, lon_sl, lat_sl)

                variance_rmse = _area_weighted_rmse_2d(vt, ve, a)
                detrended_rmse = _area_weighted_mean_2d(dtr, a)
                direct_rmse = _area_weighted_mean_2d(dr, a)
                variance_corr = _area_weighted_pearson_2d(vt, ve, a)

                rows.append(
                    {
                        "Depth": d_title,
                        "Region": region_name,
                        "Var Corr": variance_corr,
                        "Var RMSE": variance_rmse,
                        "Detrend RMSE": detrended_rmse,
                        "Direct RMSE": direct_rmse,
                    }
                )

        results[label] = pd.DataFrame(rows)

    return results


def compute_ke_temporal_variance_metrics(
    emulators_dict,
    spectrum_regions,
    var_u="uo",
    var_v="vo",
    depth_slices=None,
):
    """
    Compute 4 temporal-variance numerical metrics for kinetic energy (KE = 0.5*(u²+v²)).

    Same 4 metrics as compute_temporal_variance_metrics but applied to KE fields.

    Args:
        emulators_dict: {label: {'ds', 'truth', 'truth_label', 'color'}}
        spectrum_regions: list of (name, lon_slice, lat_slice) tuples
        var_u, var_v: velocity variable names
        depth_slices: list of {'min', 'max', 'title'}, default DEPTH_SLICES

    Returns:
        dict  {emulator_label: pandas.DataFrame}
    """
    import pandas as pd

    if depth_slices is None:
        depth_slices = DEPTH_SLICES

    regions = [("Global", None, None)] + [
        (name, lon_sl, lat_sl) for name, lon_sl, lat_sl in spectrum_regions
    ]

    results = {}

    for emu_idx, (label, emu_data) in enumerate(emulators_dict.items()):
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        area = ds_truth["areacello"]

        dz = ds_truth["dz"]
        wet_mask = (ds_truth[var_u].isel(time=0) * 0 + 1).compute()

        def _compute_ke(ds):
            return 0.5 * (ds[var_u] ** 2 + ds[var_v] ** 2)

        truth_ke = _compute_ke(ds_truth)  # (time, y, x, lev)
        emu_ke = _compute_ke(ds_emu)

        rows = []
        for dslice in depth_slices:
            d_title = dslice["title"]
            d_min, d_max = dslice["min"], dslice["max"]

            def _depth_avg(field_3d, _dmin=d_min, _dmax=d_max):
                f = field_3d.sel(lev=slice(_dmin, _dmax))
                dz_s = dz.sel(lev=slice(_dmin, _dmax))
                w_s = wet_mask.sel(lev=slice(_dmin, _dmax))
                num = (f * dz_s * w_s).sum("lev")
                den = (dz_s * w_s).sum("lev")
                return (num / den.where(den != 0)).compute()

            truth_2d = _depth_avg(truth_ke)
            emu_2d = _depth_avg(emu_ke)

            # Remove seasonal cycle (dayofyear climatology)
            truth_anom = (
                (
                    truth_2d.groupby("time.dayofyear")
                    - truth_2d.groupby("time.dayofyear").mean("time")
                )
                .drop_vars("dayofyear")
                .fillna(0.0)
            )
            emu_anom = (
                (
                    emu_2d.groupby("time.dayofyear")
                    - emu_2d.groupby("time.dayofyear").mean("time")
                )
                .drop_vars("dayofyear")
                .fillna(0.0)
            )

            # Temporal variance maps: (y, x) — on anomalies
            var_truth = truth_anom.var("time").compute()
            var_emu = emu_anom.var("time").compute()

            # Per-gridpoint temporal RMSE map: (y, x) — on anomalies
            direct_rmse_map = np.sqrt(
                ((truth_anom - emu_anom) ** 2).mean("time")
            ).compute()

            # Detrended per-gridpoint temporal RMSE map: (y, x) — detrend anomalies
            truth_dt = _detrend_linear_xr(truth_anom, dim="time")
            emu_dt = _detrend_linear_xr(emu_anom, dim="time")
            detrended_rmse_map = np.sqrt(
                ((truth_dt - emu_dt) ** 2).mean("time")
            ).compute()

            for region_name, lon_sl, lat_sl in regions:
                vt, a = _select_region(var_truth, area, lon_sl, lat_sl)
                ve, _ = _select_region(var_emu, area, lon_sl, lat_sl)
                dr, _ = _select_region(direct_rmse_map, area, lon_sl, lat_sl)
                dtr, _ = _select_region(detrended_rmse_map, area, lon_sl, lat_sl)

                variance_rmse = _area_weighted_rmse_2d(vt, ve, a)
                detrended_rmse = _area_weighted_mean_2d(dtr, a)
                direct_rmse = _area_weighted_mean_2d(dr, a)
                variance_corr = _area_weighted_pearson_2d(vt, ve, a)

                rows.append(
                    {
                        "Depth": d_title,
                        "Region": region_name,
                        "Var Corr": variance_corr,
                        "Var RMSE": variance_rmse,
                        "Detrend RMSE": detrended_rmse,
                        "Direct RMSE": direct_rmse,
                    }
                )

        results[label] = pd.DataFrame(rows)

    return results


def format_variance_metrics_table(raw_results):
    """
    Reshape raw results into a single wide table.

    Input:  {model_label: DataFrame with cols [Depth, Region, 4 metrics]}
    Output: DataFrame with
        - rows:    MultiIndex (Region, Metric) — each Region spans 4 metric rows
        - columns: MultiIndex (Depth, Model)
    """
    import pandas as pd

    metric_cols = ["Var Corr", "Var RMSE", "Detrend RMSE", "Direct RMSE"]
    frames = []
    for model_label, df in raw_results.items():
        for _, row in df.iterrows():
            for metric in metric_cols:
                frames.append(
                    {
                        "Region": row["Region"],
                        "Depth": row["Depth"],
                        "Model": model_label,
                        "Metric": metric,
                        "Value": row[metric],
                    }
                )
    long_df = pd.DataFrame(frames)
    table = long_df.pivot_table(
        index=["Region", "Metric"],
        columns=["Depth", "Model"],
        values="Value",
        sort=False,
    )
    # Preserve region and metric ordering
    region_order = list(dict.fromkeys(long_df["Region"]))
    depth_order = list(dict.fromkeys(long_df["Depth"]))
    model_order = list(dict.fromkeys(long_df["Model"]))
    row_idx = pd.MultiIndex.from_product(
        [region_order, metric_cols], names=["Region", "Metric"]
    )
    col_idx = pd.MultiIndex.from_product(
        [depth_order, model_order], names=["Depth", "Model"]
    )
    table = table.reindex(index=row_idx, columns=col_idx)
    return table


# ============================================================
# ENSO Evaluation
# ============================================================


def compute_nino34_index(sst, area, dt=5, window=150):
    """
    Compute Niño 3.4 index from SST (surface temperature).

    Steps:
        1. Select Niño 3.4 region (190-240°E, 5°S-5°N)
        2. Remove climatology (dayofyear mean)
        3. Apply rolling mean smoothing
        4. Area-weighted spatial mean

    Args:
        sst: xr.DataArray with dims (time, y, x) — surface temperature
        area: xr.DataArray with dims (y, x) — cell area
        dt: time step in days (default 5)
        window: smoothing window in days (default 150)

    Returns:
        xr.DataArray (time,) — Niño 3.4 index
    """
    sst = sst.load()
    sst_nino = sst.sel(x=slice(190, 240), y=slice(-5, 5))
    area_nino = area.sel(x=slice(190, 240), y=slice(-5, 5)).load()

    clim = sst_nino.groupby("time.dayofyear").mean("time").compute()
    window_steps = int(window / dt)

    # Remove climatology
    anom = sst_nino.copy()
    for i, t in enumerate(sst_nino.time.values):
        day = int(t.dayofyr)
        anom[i] = (sst_nino[i] - clim.sel(dayofyear=day)).data

    # Rolling mean + area-weighted spatial mean
    anom = anom.rolling(time=window_steps).mean()
    nino34 = anom.weighted(area_nino).mean(["x", "y"])

    return nino34[window_steps:]


def plot_nino34_comparison_together(emulators_dict):
    """
    Plot Niño 3.4 index time series for all emulators vs their truths.
    Reports R² (coefficient of determination) for each emulator.
    """
    print("\nGenerating Niño 3.4 Index Comparison...")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=4)

    unique_truths = {}
    for label, emu_data in emulators_dict.items():
        tl = emu_data.get("truth_label", "Truth")
        if tl not in unique_truths:
            unique_truths[tl] = emu_data["truth"]
    single_truth = len(unique_truths) == 1

    stats_text = []
    plotted_truths = {}

    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        color = emu_data["color"]
        truth_label = emu_data.get("truth_label", "Truth")
        area = ds_truth["areacello"]

        # Surface temperature (first level)
        sst_truth = ds_truth["thetao"].isel(lev=0)
        sst_emu = ds_emu["thetao"].isel(lev=0)

        nino_truth = compute_nino34_index(sst_truth, area)
        nino_emu = compute_nino34_index(sst_emu, area)

        time_truth = np.array(
            [t.year + (t.dayofyr - 1) / 365.25 for t in nino_truth.time.values]
        )
        time_emu = np.array(
            [t.year + (t.dayofyr - 1) / 365.25 for t in nino_emu.time.values]
        )

        if truth_label not in plotted_truths:
            truth_color = "black" if single_truth else color
            ax.plot(
                time_truth,
                nino_truth.values,
                label=truth_label,
                color=truth_color,
                lw=2,
                linestyle="--",
                alpha=0.7,
            )
            plotted_truths[truth_label] = nino_truth

        # R²
        min_len = min(len(nino_truth), len(nino_emu))
        y_true = nino_truth.values[:min_len]
        y_pred = nino_emu.values[:min_len]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        # Correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        stats_text.append(
            f"{label} R²={r2:.2f}, correlation={corr:.2f}, RMSE={rmse:.3f}"
        )
        ax.plot(time_emu, nino_emu.values, label=label, color=color, lw=2)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Niño 3.4 Index [°C]")
    ax.set_xlabel("Year")
    stats_block = "\n".join(stats_text)
    ax.text(
        0.99,
        0.97,
        stats_block,
        transform=ax.transAxes,
        fontsize=14,
        va="top",
        ha="right",
    )
    ax.set_title(f"Niño 3.4 Index{_scale_suffix(emulators_dict)}", loc="center")
    oh, ol, nc = reorder_legend_paired(*ax.get_legend_handles_labels())
    fig.legend(
        oh,
        ol,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def compute_nino34_metrics(emulators_dict):
    """
    Compute numerical ENSO metrics for each emulator.

    Returns:
        pandas.DataFrame with columns: Emulator, R², Correlation, MAE, RMSE
    """
    import pandas as pd

    rows = []
    for label, emu_data in emulators_dict.items():
        ds_truth = emu_data["truth"]
        ds_emu = emu_data["ds"]
        area = ds_truth["areacello"]

        sst_truth = ds_truth["thetao"].isel(lev=0)
        sst_emu = ds_emu["thetao"].isel(lev=0)

        nino_truth = compute_nino34_index(sst_truth, area)
        nino_emu = compute_nino34_index(sst_emu, area)

        min_len = min(len(nino_truth), len(nino_emu))
        y_true = nino_truth.values[:min_len]
        y_pred = nino_emu.values[:min_len]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        rows.append(
            {
                "Emulator": label,
                "R²": r2,
                "Correlation": corr,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    return pd.DataFrame(rows)


def plot_surface_speed_gulf_stream(
    emulators_dict,
    time_indices,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    lon_range=(280, 330),
    lat_range=(25, 50),
):
    """
    Surface speed snapshot (OM4 truth + Samudra-v2) in the Gulf Stream region across resolutions.

    Two rows of N columns: row 0 = OM4 truth, row 1 = Samudra-v2 emulator.
    Shared colorbar. Surface speed = sqrt(u^2 + v^2) at the given depth level.
    """
    from matplotlib.gridspec import GridSpec

    emu_labels = list(emulators_dict.keys())
    ncols = len(emu_labels)
    nrows = 2  # row 0: truth, row 1: emulator

    def compute_speed(ds, time_idx):
        u = ds[var_u].isel(time=time_idx, lev=lev_idx)
        v = ds[var_v].isel(time=time_idx, lev=lev_idx)
        return np.sqrt(u**2 + v**2).compute()

    def subset_region(da):
        return da.sel(
            x=slice(lon_range[0], lon_range[1]), y=slice(lat_range[0], lat_range[1])
        )

    ds_ref = emulators_dict[emu_labels[0]]["truth"]

    for time_idx in time_indices:
        time_val = ds_ref.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        fig = plt.figure(figsize=(5 * ncols, 4.2 * nrows))
        gs = GridSpec(
            nrows,
            ncols,
            figure=fig,
            wspace=0.08,
            hspace=0.15,
            top=0.92,
            bottom=0.10,
            left=0.05,
            right=0.97,
        )

        proj = ccrs.PlateCarree()
        axes = [
            [fig.add_subplot(gs[r, j], projection=proj) for j in range(ncols)]
            for r in range(nrows)
        ]

        # Compute truth and emulator speed for each resolution
        truth_speeds = []
        emu_speeds = []
        for label in emu_labels:
            truth_speeds.append(
                subset_region(compute_speed(emulators_dict[label]["truth"], time_idx))
            )
            emu_speeds.append(
                subset_region(compute_speed(emulators_dict[label]["ds"], time_idx))
            )

        # Shared normalization across both truth and emulator
        all_valid = np.concatenate(
            [s.values[~np.isnan(s.values)] for s in truth_speeds + emu_speeds]
        )
        vmax = np.percentile(all_valid, 99)
        norm = colors.Normalize(vmin=0, vmax=vmax)

        for j, label in enumerate(emu_labels):
            truth_label = emulators_dict[label].get("truth_label", "Truth")
            # Row 0: Truth
            ax_t = axes[0][j]
            p = truth_speeds[j].plot.pcolormesh(
                ax=ax_t,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.speed,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax_t.set_extent(
                [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                crs=ccrs.PlateCarree(),
            )
            ax_t.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax_t.coastlines(zorder=11)
            ax_t.set_title(truth_label, fontweight="bold")

            # Row 1: Emulator
            ax_e = axes[1][j]
            emu_speeds[j].plot.pcolormesh(
                ax=ax_e,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.speed,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax_e.set_extent(
                [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                crs=ccrs.PlateCarree(),
            )
            ax_e.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax_e.coastlines(zorder=11)
            ax_e.set_title(label, fontweight="bold")

        all_axes = axes[0] + axes[1]
        fig.colorbar(
            p,
            ax=all_axes,
            orientation="horizontal",
            shrink=0.5,
            pad=0.06,
            aspect=40,
            label="Surface Speed (m/s)",
        )
        fig.suptitle(
            f"Surface Speed (Gulf Stream) — {time_str}", fontsize=_SUPTITLE_SIZE, y=0.98
        )
        plt.show()


def plot_surface_ke_gulf_stream(
    emulators_dict,
    time_indices,
    var_u="uo",
    var_v="vo",
    lev_idx=0,
    lon_range=(280, 330),
    lat_range=(25, 50),
):
    """
    Surface KE snapshot (OM4 truth + Samudra-v2) in the Gulf Stream region across resolutions.

    Two rows of N columns: row 0 = OM4 truth, row 1 = Samudra-v2 emulator.
    Shared colorbar. KE = 0.5 * (u^2 + v^2) at the given depth level.
    """
    from matplotlib.gridspec import GridSpec

    emu_labels = list(emulators_dict.keys())
    ncols = len(emu_labels)
    nrows = 2  # row 0: truth, row 1: emulator

    def compute_ke(ds, time_idx):
        u = ds[var_u].isel(time=time_idx, lev=lev_idx)
        v = ds[var_v].isel(time=time_idx, lev=lev_idx)
        return (0.5 * (u**2 + v**2)).compute()

    def subset_region(da):
        return da.sel(
            x=slice(lon_range[0], lon_range[1]), y=slice(lat_range[0], lat_range[1])
        )

    ds_ref = emulators_dict[emu_labels[0]]["truth"]

    for time_idx in time_indices:
        time_val = ds_ref.time.isel(time=time_idx).values
        if hasattr(time_val, "year"):
            time_str = f"{time_val.year:04d}-{time_val.month:02d}-{time_val.day:02d}"
        else:
            time_str = str(time_val)[:10]

        fig = plt.figure(figsize=(5 * ncols, 4.2 * nrows))
        gs = GridSpec(
            nrows,
            ncols,
            figure=fig,
            wspace=0.08,
            hspace=0.15,
            top=0.92,
            bottom=0.10,
            left=0.05,
            right=0.97,
        )

        proj = ccrs.PlateCarree()
        axes = [
            [fig.add_subplot(gs[r, j], projection=proj) for j in range(ncols)]
            for r in range(nrows)
        ]

        # Compute truth and emulator KE for each resolution
        truth_ke = []
        emu_ke = []
        for label in emu_labels:
            truth_ke.append(
                subset_region(compute_ke(emulators_dict[label]["truth"], time_idx))
            )
            emu_ke.append(
                subset_region(compute_ke(emulators_dict[label]["ds"], time_idx))
            )

        # Shared LogNorm normalization across both truth and emulator
        all_valid = np.concatenate(
            [s.values[~np.isnan(s.values)] for s in truth_ke + emu_ke]
        )
        flat_pos = all_valid[all_valid > 0]
        vmin = np.percentile(flat_pos, 5) if len(flat_pos) > 0 else 1e-10
        vmax = np.percentile(flat_pos, 99) if len(flat_pos) > 0 else 1.0
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        for j, label in enumerate(emu_labels):
            truth_label = emulators_dict[label].get("truth_label", "Truth")
            # Row 0: Truth
            ax_t = axes[0][j]
            p = truth_ke[j].plot.pcolormesh(
                ax=ax_t,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.thermal,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax_t.set_extent(
                [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                crs=ccrs.PlateCarree(),
            )
            ax_t.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax_t.coastlines(zorder=11)
            ax_t.set_title(truth_label, fontweight="bold")

            # Row 1: Emulator
            ax_e = axes[1][j]
            emu_ke[j].plot.pcolormesh(
                ax=ax_e,
                transform=ccrs.PlateCarree(),
                x="x",
                y="y",
                cmap=cmocean.cm.thermal,
                norm=norm,
                add_colorbar=False,
                rasterized=True,
            )
            ax_e.set_extent(
                [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                crs=ccrs.PlateCarree(),
            )
            ax_e.add_feature(
                cfeature.LAND, zorder=10, edgecolor="black", facecolor="lightgray"
            )
            ax_e.coastlines(zorder=11)
            ax_e.set_title(label, fontweight="bold")

        all_axes = axes[0] + axes[1]
        fig.colorbar(
            p,
            ax=all_axes,
            orientation="horizontal",
            shrink=0.5,
            pad=0.06,
            aspect=40,
            label="KE (m$^2$/s$^2$)",
        )
        fig.suptitle(
            f"Surface Kinetic Energy (Gulf Stream) — {time_str}",
            fontsize=_SUPTITLE_SIZE,
            y=0.98,
        )
        plt.show()
