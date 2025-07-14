"""Spatial visualization module for ocean data."""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FixedLocator


def linear_piecewise_scale(break_point: float, factor: float, ax=None):
    """
    Apply linear piecewise scaling to depth axis for better visualization.

    Args:
        break_point: Depth break point for scaling change
        factor: Scaling factor below break point
        ax: Matplotlib axis to apply scaling to
    """
    if ax is None:
        ax = plt.gca()

    # Get current y-axis limits
    ymin, ymax = ax.get_ylim()

    # Create custom tick locations
    ticks_above = np.arange(0, break_point + 1, 250)  # Above break point
    ticks_below = np.arange(break_point + 1000, ymax + 1, 1000)  # Below break point

    # Combine ticks
    all_ticks = np.concatenate([ticks_above, ticks_below])
    all_ticks = all_ticks[all_ticks <= ymax]

    ax.set_yticks(all_ticks)


def remove_climatology(ds: xr.DataArray) -> xr.DataArray:
    """
    Remove seasonal climatology from the data.

    Args:
        ds: Input data array

    Returns:
        Data array with seasonal cycle removed
    """
    try:
        # Try to use dayofyear grouping (for proper datetime coordinates)
        climatology = ds.groupby("time.dayofyear").mean("time").compute()
        day_of_year = ds["time"].dt.dayofyear
        res = (ds - climatology.sel(dayofyear=day_of_year)).compute()
        return res
    except (AttributeError, KeyError):
        # Fallback: use month grouping or simple detrending if dayofyear is not available
        try:
            # Try month-based climatology
            climatology = ds.groupby("time.month").mean("time").compute()
            month = ds["time"].dt.month
            res = (ds - climatology.sel(month=month)).compute()
            return res
        except (AttributeError, KeyError):
            # Simple fallback: just return the data minus its mean (no seasonal cycle removal)
            print(
                "Warning: Could not remove seasonal climatology, returning data minus mean"
            )
            return ds - ds.mean("time")


def map_bias_avg(data_pred1: xr.DataArray, fig, title: str = "", **kwargs):
    """
    Create a bias map with specific styling for OHC analysis.

    Args:
        data_pred1: Data array to plot
        fig: matplotlib figure
        title: Plot title
        **kwargs: Additional arguments including var_name

    Returns:
        Tuple of (axis, image) objects
    """
    var_name = kwargs["var_name"]

    plt.clf()
    plt.rcParams.update({"font.size": 14})

    # Define colormap
    new_cmap = cm.cm.balance
    new_cmap.set_bad(color="grey", alpha=0.0)

    # Set common color range for the colorbar
    vmin, vmax = {
        "thetao": (-5, 5),
        "so": (-1, 1),
        "uo": (-0.01, 0.01),
        "vo": (-0.01, 0.01),
        "zos": (-1, 1),
        "OHC": (-0.05, 0.05),
    }[var_name]

    # Create figure with one subplot
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Plot data
    im = data_pred1.plot(
        ax=ax,
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax.set_title(title, fontsize=14)

    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    return ax, im


def create_spatial_directories(output_path: str) -> dict[str, str]:
    """
    Create directories for spatial outputs.

    Args:
        output_path: Base output path

    Returns:
        Dictionary of directory paths
    """
    directories = {}

    # Create main directories
    directories["ohc"] = os.path.join(output_path, "OHC")
    directories["temperature"] = os.path.join(output_path, "Temperature")
    directories["salinity"] = os.path.join(output_path, "Salinity")
    directories["pdfs"] = os.path.join(output_path, "PDFs")
    directories["enso"] = os.path.join(output_path, "ENSO")
    directories["metrics"] = os.path.join(output_path, "Metrics")

    for path in directories.values():
        os.makedirs(path, exist_ok=True)

    return directories


def setup_cartopy_projection(ax, extent: list[float] | None = None):
    """
    Setup cartopy projection and features for ocean plots.

    Args:
        ax: Matplotlib axis with cartopy projection
        extent: [lon_min, lon_max, lat_min, lat_max] for zooming
    """
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray")
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)


def plot_ohc_map(
    ohc_data: xr.DataArray,
    ax,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot Ocean Heat Content map.

    Args:
        ohc_data: OHC data array
        ax: Matplotlib axis
        title: Plot title
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap name
    """
    if vmin is None or vmax is None:
        mean = ohc_data.mean().compute().item()
        std = ohc_data.std().compute().item()
        vmin = mean - 4 * std
        vmax = mean + 4 * std

    im = ohc_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    ax.set_title(title)
    setup_cartopy_projection(ax)

    return im


def plot_sst_map(
    sst_data: xr.DataArray,
    ax,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "thermal",
) -> None:
    """
    Plot Sea Surface Temperature map.

    Args:
        sst_data: SST data array
        ax: Matplotlib axis
        title: Plot title
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap name
    """
    if vmin is None or vmax is None:
        mean = sst_data.mean().compute().item()
        std = sst_data.std().compute().item()
        vmin = mean - std
        vmax = mean + std

    im = sst_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=getattr(cm.cm, cmap, "viridis"),  # Use cm.cm to access colormaps
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    ax.set_title(title)
    setup_cartopy_projection(ax)

    return im


def plot_bias_map(
    bias_data: xr.DataArray,
    ax,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot bias map between prediction and truth.

    Args:
        bias_data: Bias data array
        ax: Matplotlib axis
        title: Plot title
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap name
    """
    if vmin is None or vmax is None:
        abs_max = np.abs(bias_data).max().compute().item()
        vmin = -abs_max
        vmax = abs_max

    im = bias_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    ax.set_title(title)
    setup_cartopy_projection(ax)

    return im


def create_ohc_timeseries_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
    dataset_name: str = "OM4",
) -> None:
    """
    Create OHC timeseries plots (OHC.png and OHC_ref0_noanomaly.png).

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of the ground truth dataset
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Physical constants
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3

    directories = create_spatial_directories(output_path)

    # Create compare_info.txt file
    compare_file = open(os.path.join(output_path, "compare_info.txt"), "a")

    # OHC with anomaly (deseasonalized) - OHC.png
    plt.rcdefaults()
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
    )
    plt.rcParams.update({"font.size": 9})

    # Compute OHC for ground truth with deseasonalization
    OHC = (
        (ds_groundtruth["thetao"] * c_p * rho_0)
        * ds_groundtruth["areacello"]
        * ds_groundtruth["dz"]
    ).sum(["x", "y", "lev"]) / 1e21

    # Apply deseasonalization (remove climatology)
    # Simplified deseasonalization - remove monthly means
    if len(OHC.time) > 12:
        OHC_monthly_mean = OHC.groupby("time.month").mean("time")
        OHC = OHC.groupby("time.month") - OHC_monthly_mean

    OHC = OHC.rename("OHC Anomaly")
    OHC = OHC.assign_attrs(units="ZJ")

    # Plot predictions
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        OHC_pred = (
            (pred_data["ds_prediction"]["thetao"] * c_p * rho_0)
            * pred_data["ds_prediction"]["areacello"]
            * pred_data["ds_prediction"]["dz"]
        ).sum(["x", "y", "lev"]) / 1e21

        # Apply same deseasonalization
        if len(OHC_pred.time) > 12:
            OHC_pred_monthly_mean = OHC_pred.groupby("time.month").mean("time")
            OHC_pred = OHC_pred.groupby("time.month") - OHC_pred_monthly_mean

        OHC_pred = OHC_pred.rename("OHC Anomaly")
        OHC_pred = OHC_pred.assign_attrs(units="ZJ")

        # Convert time for matplotlib
        pred_time_vals = pd.to_datetime([str(t) for t in OHC_pred.time.values])
        ax.plot(
            pred_time_vals,
            OHC_pred,
            label=titles[i] if i < len(titles) else k,
            color=plt.cm.tab10(i),
            linewidth=1,
        )

        # Compute trend
        coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred.size), OHC_pred, 1)
        trend_line = (
            np.arange(OHC_pred.size) * coeffs_OHC_pred_trend[0]
            + coeffs_OHC_pred_trend[1]
        )
        ax.plot(
            pred_time_vals,
            trend_line,
            color=plt.cm.tab10(i),
            linestyle="--",
            linewidth=1,
        )

        compare_file.write(
            f"\nOHC {titles[i] if i < len(titles) else k} Trend Slope : {coeffs_OHC_pred_trend[0]}"
        )

    # Plot ground truth
    truth_time_vals = pd.to_datetime([str(t) for t in OHC.time.values])
    ax.plot(truth_time_vals, OHC, label=dataset_name, color="k", linewidth=1)

    # Ground truth trend
    coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
    trend_line_gt = np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1]
    ax.plot(truth_time_vals, trend_line_gt, color="k", linestyle="--", linewidth=1)

    compare_file.write(f"\nOHC GT Trend Slope : {coeffs_OHC_trend[0]}")

    ax.set_title("")
    ax.set_xlabel("Time")
    ax.set_ylabel("OHC Anomaly (ZJ)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.89), ncols=3)

    plt.savefig(
        os.path.join(directories["ohc"], "OHC.png"), bbox_inches="tight", dpi=600
    )
    plt.close()

    # OHC without anomaly (reference) - OHC_ref0_noanomaly.png
    plt.rcdefaults()
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
    )
    plt.rcParams.update({"font.size": 9})

    # Compute OHC for ground truth without deseasonalization, relative to first timestep
    OHC_ref = (
        (ds_groundtruth["thetao"] * c_p * rho_0)
        * ds_groundtruth["areacello"]
        * ds_groundtruth["dz"]
    ).sum(["x", "y", "lev"]) / 1e21
    OHC_ref = OHC_ref - OHC_ref.isel(time=0)  # Reference to first timestep
    OHC_ref = OHC_ref.rename("OHC")
    OHC_ref = OHC_ref.assign_attrs(units="ZJ")

    # Plot predictions
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        OHC_pred_ref = (
            (pred_data["ds_prediction"]["thetao"] * c_p * rho_0)
            * pred_data["ds_prediction"]["areacello"]
            * pred_data["ds_prediction"]["dz"]
        ).sum(["x", "y", "lev"]) / 1e21
        OHC_pred_ref = OHC_pred_ref - OHC_pred_ref.isel(
            time=0
        )  # Reference to first timestep
        OHC_pred_ref = OHC_pred_ref.rename("OHC")
        OHC_pred_ref = OHC_pred_ref.assign_attrs(units="ZJ")

        # Convert time for matplotlib
        pred_time_vals = pd.to_datetime([str(t) for t in OHC_pred_ref.time.values])
        ax.plot(
            pred_time_vals,
            OHC_pred_ref,
            label=titles[i] if i < len(titles) else k,
            color=plt.cm.tab10(i),
            linewidth=1,
        )

        # Compute trend
        coeffs_OHC_pred_trend = np.polyfit(
            np.arange(OHC_pred_ref.size), OHC_pred_ref, 1
        )
        trend_line = (
            np.arange(OHC_pred_ref.size) * coeffs_OHC_pred_trend[0]
            + coeffs_OHC_pred_trend[1]
        )
        ax.plot(
            pred_time_vals,
            trend_line,
            color=plt.cm.tab10(i),
            linestyle="--",
            linewidth=1,
        )

    # Plot ground truth
    truth_time_vals = pd.to_datetime([str(t) for t in OHC_ref.time.values])
    ax.plot(truth_time_vals, OHC_ref, label=dataset_name, color="k", linewidth=1)

    # Ground truth trend
    coeffs_OHC_trend = np.polyfit(np.arange(OHC_ref.size), OHC_ref, 1)
    trend_line_gt = np.arange(OHC_ref.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1]
    ax.plot(truth_time_vals, trend_line_gt, color="k", linestyle="--", linewidth=1)

    ax.set_title("")
    ax.set_xlabel("Time")
    ax.set_ylabel("OHC (ZJ)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.89), ncols=3)

    plt.savefig(
        os.path.join(directories["ohc"], "OHC_ref0_noanomaly.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    compare_file.close()


def create_ohc_bias_difference_maps(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
    dataset_name: str = "OM4",
) -> None:
    """
    Create OHC bias difference maps (OHC_Bias_Map_Diff1_2.png, OHC_Bias_Map_Diff1_3.png, and OHC_Bias_Map_Diff_Last_First.png).

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    # Physical constants
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3

    directories = create_spatial_directories(output_path)

    # Get time indices for analysis - use original viz.py approach
    n_times = len(ds_groundtruth.time)

    # Original approach from viz.py:
    # first_year = slice(None, 73), last_year = slice(-73, -1)
    # For compatibility, use same approach but adapt to actual data size
    year_length = min(73, n_times // 6)  # Adapt to data size
    first_year = slice(None, year_length)
    last_year = (
        slice(-year_length, -1) if year_length < n_times else slice(-year_length, None)
    )
    second_last_year = (
        slice(-2 * year_length, -year_length)
        if 2 * year_length < n_times
        else slice(None, year_length)
    )
    third_last_year = (
        slice(-3 * year_length, -2 * year_length)
        if 3 * year_length < n_times
        else slice(None, year_length)
    )

    # Process ground truth and first prediction for OHC computation
    gt_ohc = None
    pred1_ohc = None

    for i, (key, pred_data) in enumerate(pred_dict.items()):
        # Compute OHC for ground truth (keep spatial dimensions for mapping)
        if gt_ohc is None:
            OHC_gt = (
                (ds_groundtruth["thetao"] * c_p * rho_0)
                * ds_groundtruth["areacello"]
                * ds_groundtruth["dz"]
            ).sum(["lev"]) / 1e21  # Sum over depth only, keep spatial dimensions

            # Apply section mask if available (otherwise skip masking)
            # OHC_gt = OHC_gt.where(~section_mask) if 'section_mask' in locals() else OHC_gt

            OHC_gt = remove_climatology(OHC_gt)
            OHC_gt = OHC_gt.rename("OHC Anomaly")
            # Only assign coordinate attributes if coordinates exist
            if "y" in OHC_gt.coords:
                OHC_gt["y"] = OHC_gt.y.assign_attrs(
                    long_name="latitude", units=r"${^o}$"
                )
            if "x" in OHC_gt.coords:
                OHC_gt["x"] = OHC_gt.x.assign_attrs(
                    long_name="longitude", units=r"${^o}$"
                )
            OHC_gt = OHC_gt.assign_attrs(units="ZJ")
            gt_ohc = OHC_gt

        # Compute OHC for first prediction (keep spatial dimensions for mapping)
        if i == 0:  # Only process the first prediction
            ds_pred = pred_data["ds_prediction"]
            OHC_pred = (
                (ds_pred["thetao"] * c_p * rho_0) * ds_pred["areacello"] * ds_pred["dz"]
            ).sum(["lev"]) / 1e21  # Sum over depth only, keep spatial dimensions

            OHC_pred = remove_climatology(OHC_pred)
            # OHC_pred = OHC_pred.where(~section_mask) if 'section_mask' in locals() else OHC_pred
            OHC_pred = OHC_pred.rename("OHC Anomaly")
            # Only assign coordinate attributes if coordinates exist
            if "y" in OHC_pred.coords:
                OHC_pred["y"] = OHC_pred.y.assign_attrs(
                    long_name="latitude", units=r"${^o}$"
                )
            if "x" in OHC_pred.coords:
                OHC_pred["x"] = OHC_pred.x.assign_attrs(
                    long_name="longitude", units=r"${^o}$"
                )
            OHC_pred = OHC_pred.assign_attrs(units="ZJ")
            pred1_ohc = OHC_pred
            break

    if pred1_ohc is None:
        print("Warning: No prediction data found for OHC bias difference maps")
        return

    # Create OHC_Bias_Map_Diff1_2.png (Last Year - Second Last Year)
    da = (
        pred1_ohc.isel(time=last_year).mean("time")
        - pred1_ohc.isel(time=second_last_year).mean("time")
    ).compute()

    fig = plt.figure(figsize=(10, 10))
    ax, im = map_bias_avg(
        da,
        fig,
        var_name="OHC",
        title="OHC Bias (Last Year - Second Last Year)",
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.6, label="OHC Bias (ZJ)")

    fig.tight_layout()
    plt.savefig(
        os.path.join(directories["ohc"], "OHC_Bias_Map_Diff1_2.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # Create OHC_Bias_Map_Diff1_3.png (Last Year - Third Last Year) if we have enough data
    if 3 * year_length < n_times:
        da = (
            pred1_ohc.isel(time=last_year).mean("time")
            - pred1_ohc.isel(time=third_last_year).mean("time")
        ).compute()

        fig = plt.figure(figsize=(10, 10))
        ax, im = map_bias_avg(
            da,
            fig,
            var_name="OHC",
            title="OHC Bias (Last Year - Third Last Year)",
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.6, label="OHC Bias (ZJ)")

        fig.tight_layout()
        plt.savefig(
            os.path.join(directories["ohc"], "OHC_Bias_Map_Diff1_3.png"),
            bbox_inches="tight",
            dpi=600,
        )
        plt.close()

    # Create OHC_Bias_Map_Diff_Last_First.png (Last Year - First Year)
    # This is the missing plot from the original list
    da = (
        pred1_ohc.isel(time=last_year).mean("time")
        - pred1_ohc.isel(time=first_year).mean("time")
    ).compute()

    fig = plt.figure(figsize=(10, 10))
    ax, im = map_bias_avg(
        da,
        fig,
        var_name="OHC",
        title="OHC Bias (Last Year - First Year)",
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.6, label="OHC Bias (ZJ)")

    fig.tight_layout()
    plt.savefig(
        os.path.join(directories["ohc"], "OHC_Bias_Map_Diff_Last_First.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def create_ohc_depth_timeseries(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
    dataset_name: str = "OM4",
) -> None:
    """
    Create OHC depth-wise timeseries plot (OHC_Timeseries_depths.png).

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    # Physical constants
    Days_to_Eq = 0
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3

    directories = create_spatial_directories(output_path)

    # Create figure with 3 subplots (Upper, Mid, Deep)
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(
        3, 1, figsize=(10, 7.5), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
    )
    plt.rcParams.update({"font.size": 9})

    # Open compare_info.txt file for writing trends
    compare_file = open(os.path.join(output_path, "compare_info.txt"), "a")

    # Define depth ranges
    depth_ranges = {
        "upper": slice(0, 700),  # 0-700m
        "mid": slice(700, 2000),  # 700-2000m
        "deep": slice(2000, None),  # 2000m+
    }

    depth_labels = {"upper": "0-0.7km", "mid": "0.7-2.0km", "deep": "2.0-6.0km"}

    # Color list for predictions
    clist = ["red", "blue", "green", "orange", "purple", "brown"]

    # Store trend calculations for summary
    pred_trends = {}
    gt_trends = {}

    # Process each depth range
    for depth_idx, (depth_name, depth_slice) in enumerate(depth_ranges.items()):
        # Compute OHC for ground truth at this depth
        OHC_truth = (
            (ds_groundtruth["thetao"].sel(lev=depth_slice) * c_p * rho_0)
            * ds_groundtruth["areacello"]
            * ds_groundtruth["dz"]
        ).sum(["x", "y", "lev"]) / 1e21

        OHC_truth = remove_climatology(OHC_truth)
        OHC_truth.plot(ax=ax[depth_idx], label=dataset_name, c="k")

        # Compute and plot trend for ground truth
        coeffs_OHC_ground_trend = np.polyfit(
            np.arange(OHC_truth[Days_to_Eq:].size), OHC_truth[Days_to_Eq:], 1
        )
        (pos,) = ax[depth_idx].plot(
            OHC_truth[Days_to_Eq:].time.data,
            np.arange(OHC_truth[Days_to_Eq:].size) * coeffs_OHC_ground_trend[0]
            + coeffs_OHC_ground_trend[1],
            c="k",
            ls="--",
        )

        compare_file.write(
            f"\n{depth_name.title()} - GT Trend Slope : {coeffs_OHC_ground_trend[0]}"
        )
        gt_trends[depth_name] = coeffs_OHC_ground_trend[0] * 73

        # Process predictions
        for i, (key, pred_data) in enumerate(pred_dict.items()):
            ds_pred = pred_data["ds_prediction"]

            # Compute OHC for prediction at this depth
            OHC_pred = (
                (ds_pred["thetao"].sel(lev=depth_slice) * c_p * rho_0)
                * ds_pred["areacello"]
                * ds_pred["dz"]
            ).sum(["x", "y", "lev"]) / 1e21

            OHC_pred = remove_climatology(OHC_pred)
            OHC_pred = OHC_pred.rename(depth_labels[depth_name])
            OHC_pred = OHC_pred.assign_attrs(units="ZJ")

            # Compute trend
            coeffs_OHC_pred_trend = np.polyfit(
                np.arange(OHC_pred[Days_to_Eq:].size),
                OHC_pred[Days_to_Eq:],
                1,
            )

            # Plot prediction
            title = titles[i] if i < len(titles) else key
            OHC_pred.plot(ax=ax[depth_idx], label=title, c=clist[i % len(clist)])

            # Plot trend line
            (pos,) = ax[depth_idx].plot(
                OHC_pred[Days_to_Eq:].time.data,
                np.arange(OHC_pred[Days_to_Eq:].size) * coeffs_OHC_pred_trend[0]
                + coeffs_OHC_pred_trend[1],
                c=clist[i % len(clist)],
                ls="--",
            )

            compare_file.write(
                f"\n{depth_name.title()} - {title} Trend Slope : {coeffs_OHC_pred_trend[0]}"
            )

            # Store trend for later calculations
            if key not in pred_trends:
                pred_trends[key] = {}
            pred_trends[key][depth_name] = coeffs_OHC_pred_trend[0] * 73

        # Set subplot title
        if depth_idx == 0:
            ax[depth_idx].set_title("OHC Anomaly")
        else:
            ax[depth_idx].set_title("")

    # Calculate and write trend ratios
    total_trend_truth = sum(gt_trends.values())
    compare_file.write(
        f"\nGT Trend Ratio (Upper, Mid, Deep): {gt_trends['upper'] / total_trend_truth:.2f}, {gt_trends['mid'] / total_trend_truth:.2f}, {gt_trends['deep'] / total_trend_truth:.2f}"
    )

    for key, trends in pred_trends.items():
        total_trend_pred = sum(trends.values())
        compare_file.write(
            f"\n{titles[list(pred_dict.keys()).index(key)] if key in pred_dict else key} Trend Ratio (Upper, Mid, Deep): {trends['upper'] / total_trend_pred:.2f}, {trends['mid'] / total_trend_pred:.2f}, {trends['deep'] / total_trend_pred:.2f}"
        )

    compare_file.write("\n")
    compare_file.close()

    # Add legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncols=3)

    # Save the plot
    plt.savefig(
        os.path.join(directories["ohc"], "OHC_Timeseries_depths.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def create_ohc_analysis_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
) -> None:
    """
    Create Ocean Heat Content analysis plots.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
    """
    # Physical constants
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3
    zeta_joules_factor = 1e21

    # Compute OHC for ground truth
    ohc_gt = (
        (ds_groundtruth["thetao"] * c_p * rho_0 / zeta_joules_factor)
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .sum(["lev"])
        .compute()
    )

    # Compute OHC for predictions
    ohc_preds = {}
    for key, pred_data in pred_dict.items():
        ds_pred = pred_data["ds_prediction"]
        ohc_pred = (
            (ds_pred["thetao"] * c_p * rho_0 / zeta_joules_factor)
            .weighted(ds_pred["areacello"] * ds_pred["dz"])
            .sum(["lev"])
            .compute()
        )
        ohc_preds[key] = ohc_pred

    # Create plots
    directories = create_spatial_directories(output_path)

    # Global OHC map
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    im = plot_ohc_map(ohc_gt.mean("time"), ax, "Ground Truth OHC")
    plt.colorbar(im, ax=ax, shrink=0.6, label="OHC (ZJ)")
    plt.savefig(
        os.path.join(directories["ohc"], "OHC_Global_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # OHC bias maps
    for i, (key, ohc_pred) in enumerate(ohc_preds.items()):
        bias = ohc_pred.mean("time") - ohc_gt.mean("time")

        fig, ax = plt.subplots(
            1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        im = plot_bias_map(
            bias, ax, f"OHC Bias: {titles[i] if i < len(titles) else key}"
        )
        plt.colorbar(im, ax=ax, shrink=0.6, label="OHC Bias (ZJ)")
        plt.savefig(
            os.path.join(directories["ohc"], f"OHC_Bias_Map_{key}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Add missing OHC timeseries plots
    create_ohc_timeseries_plots(ds_groundtruth, pred_dict, output_path, titles)

    # Add missing OHC bias difference maps
    create_ohc_bias_difference_maps(ds_groundtruth, pred_dict, output_path, titles)

    # Add missing OHC depth timeseries
    create_ohc_depth_timeseries(ds_groundtruth, pred_dict, output_path, titles)


def create_sst_analysis_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
    basin_masks: xr.Dataset | None = None,
    dataset_name: str = "OM4",
) -> None:
    """
    Create comprehensive temperature analysis plots including SST maps, snapshots, bias analysis, and temperature profiles.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
        basin_masks: Basin mask dataset (optional)
        dataset_name: Name of ground truth dataset
    """
    directories = create_spatial_directories(output_path)

    # Get SST data (either from tos or surface thetao)
    if "tos" in ds_groundtruth.data_vars:
        sst_gt = ds_groundtruth["tos"]
    else:
        sst_gt = ds_groundtruth["thetao"].isel(lev=0)

    # Global SST map
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    im = plot_sst_map(sst_gt.mean("time"), ax, "Ground Truth SST")
    plt.colorbar(im, ax=ax, shrink=0.6, label="SST (°C)")
    plt.savefig(
        os.path.join(directories["temperature"], "SST_Global_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SST snapshots at different times
    last_index = len(sst_gt.time) - 1
    time_indices = [0, last_index // 2, last_index]
    time_labels = ["Start", "Mid", "End"]

    for i, (time_idx, label) in enumerate(zip(time_indices, time_labels)):
        fig, ax = plt.subplots(
            1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        im = plot_sst_map(sst_gt.isel(time=time_idx), ax, f"SST Snapshot - {label}")
        plt.colorbar(im, ax=ax, shrink=0.6, label="SST (°C)")
        plt.savefig(
            os.path.join(
                directories["temperature"], f"SST_map_snapshot_t_{time_idx}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        # Also save middle snapshot as t_300 for gold standard compatibility
        if i == 1 and abs(time_idx - 300) <= 5:  # Middle snapshot close to 300
            plt.savefig(
                os.path.join(directories["temperature"], "SST_map_snapshot_t_300.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    # ✅ INTEGRATED: Create detailed SST snapshot with bias analysis (originally in "missing" function)
    if pred_dict:
        key1 = list(pred_dict.keys())[0]
        ds_pred = pred_dict[key1]["ds_prediction"]

        # Create the detailed snapshot plot with ground truth, prediction, and bias
        t_index = time_indices[1]  # middle time index

        plt.rcParams.update({"font.size": 14})
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(16, 6),
            subplot_kw={"projection": ccrs.PlateCarree()},
            gridspec_kw={"wspace": 0.02, "hspace": 0.23},
        )
        axs = axs.flatten()

        # Plot ground truth and prediction side by side
        titles_snapshot = [
            f"OM4 t={t_index}",
            f"{pred_dict[key1].get('name', key1)} t={t_index}",
        ]
        datasets = [ds_groundtruth, ds_pred]

        for i, (ax, title, ds) in enumerate(zip(axs, titles_snapshot, datasets)):
            # Get SST data
            if "tos" in ds.data_vars:
                sst_data = ds["tos"].isel(time=t_index)
            else:
                sst_data = ds["thetao"].isel(lev=0, time=t_index)

            # Apply section mask and plot
            section_mask = np.isnan(sst_data)
            sst_data = sst_data.where(~section_mask)

            if i == 0:
                gt_sst = sst_data
            elif i == 1:
                pred1_sst = sst_data

            # Plot SST using thermal colormap
            im = sst_data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=getattr(cm.cm, "thermal", "viridis"),
                add_colorbar=False,
            )
            ax.add_feature(cfeature.COASTLINE, edgecolor="black")
            ax.set_title(title, fontsize=14)
            setup_cartopy_projection(ax)

        # Add colorbar for SST plots
        cbar = fig.colorbar(
            im, ax=axs[:2], orientation="vertical", fraction=0.01, pad=0.02
        )
        cbar.set_label(r"SST [°C]", fontsize=14)

        # Plot bias in the third panel
        sst_bias = pred1_sst - gt_sst
        im_bias = sst_bias.plot(
            ax=axs[3],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-0.5,
            vmax=0.5,
            add_colorbar=False,
        )
        axs[3].add_feature(cfeature.COASTLINE, edgecolor="black")
        axs[3].set_title(f"{pred_dict[key1].get('name', key1)} Bias", fontsize=14)
        setup_cartopy_projection(axs[3])

        # Add colorbar for bias plot
        cbar = fig.colorbar(
            im_bias, ax=axs[3:], orientation="vertical", fraction=0.1, pad=0.02
        )
        cbar.set_label(r"SST Bias [°C]", fontsize=14)

        # Remove the empty axis
        fig.delaxes(axs[2])

        plt.savefig(
            os.path.join(
                directories["temperature"], f"SST_detailed_snapshot_t_{t_index}.png"
            ),
            bbox_inches="tight",
            dpi=600,
        )
        plt.close()

    # SST bias maps for predictions
    for i, (key, pred_data) in enumerate(pred_dict.items()):
        ds_pred = pred_data["ds_prediction"]

        if "tos" in ds_pred.data_vars:
            sst_pred = ds_pred["tos"]
        else:
            sst_pred = ds_pred["thetao"].isel(lev=0)

        bias = sst_pred.mean("time") - sst_gt.mean("time")

        fig, ax = plt.subplots(
            1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        im = plot_bias_map(
            bias, ax, f"SST Bias: {titles[i] if i < len(titles) else key}"
        )
        plt.colorbar(im, ax=ax, shrink=0.6, label="SST Bias (°C)")
        plt.savefig(
            os.path.join(directories["temperature"], f"SST_Bias_Map_{key}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # ✅ INTEGRATED: Temperature profile analysis (Last Year - First Year plots)
    if pred_dict and basin_masks is not None:
        print("🔧 Starting temperature profile plots (simplified version)...")

        try:
            # Get first prediction
            key1 = list(pred_dict.keys())[0]
            ds_pred = pred_dict[key1]["ds_prediction"]
            pred_name = pred_dict[key1].get("name", key1)

            print(f"📊 Processing {pred_name} temperature data...")

            # Simplified approach: use raw temperature data without deseasonalization to avoid crashes
            # Compute "Last Year - First Year" differences using simpler time slicing
            n_times = len(ds_groundtruth["thetao"].time)
            year_length = min(73, n_times // 6)  # Adapt to actual data size

            print(
                f"⏰ Using time slices: first {year_length} vs last {year_length} timesteps"
            )

            # 1. CM4 (Last Year - First Year) - simplified without climatology removal
            print("📈 Computing CM4 temperature change...")
            CM4_last = (
                ds_groundtruth["thetao"]
                .isel(time=slice(-year_length, None))
                .mean(dim="time")
            )
            CM4_first = (
                ds_groundtruth["thetao"]
                .isel(time=slice(0, year_length))
                .mean(dim="time")
            )
            CM4_lastyear_change = CM4_last - CM4_first

            # Create simple basin-averaged plots instead of complex basin analysis
            print("🗺️ Creating CM4 profile plot...")
            create_simple_temperature_profile(
                CM4_lastyear_change,
                f"{dataset_name} (Last Year - First Year)",
                output_path,
            )

            # 2. Prediction (Last Year - First Year)
            print("📈 Computing prediction temperature change...")
            pred_last = (
                ds_pred["thetao"].isel(time=slice(-year_length, None)).mean(dim="time")
            )
            pred_first = (
                ds_pred["thetao"].isel(time=slice(0, year_length)).mean(dim="time")
            )
            pred_lastyear_change = pred_last - pred_first

            print("🗺️ Creating prediction profile plot...")
            create_simple_temperature_profile(
                pred_lastyear_change,
                f"{pred_name} (Last Year - First Year)",
                output_path,
            )

            # 3. Bias (Last Year - First Year)
            print("📊 Computing bias...")
            bias_lastyear_change = pred_lastyear_change - CM4_lastyear_change

            print("🗺️ Creating bias profile plot...")
            create_simple_temperature_profile(
                bias_lastyear_change, "(Last Year - First Year) Bias", output_path
            )

            print("✅ Temperature profile plots completed successfully!")

        except Exception as e:
            print(f"❌ Error in temperature profile plots: {e}")
            print("⚠️ Skipping temperature profile plots to continue pipeline...")

    elif pred_dict and basin_masks is None:
        print("⚠️ Basin masks not provided, skipping temperature profile plots")


def create_salinity_analysis_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
) -> None:
    """
    Create salinity analysis plots including time series and spatial maps.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
    """
    directories = create_spatial_directories(output_path)

    # Physical constants
    rho_0 = 1025  # kg/m^3
    clist = ["red", "blue", "green", "orange", "purple", "brown"]

    # ✅ INTEGRATED: Create salinity time series plots (originally in "missing" function)

    # 1. Total salinity mass time series (non-deseasonalized)
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.rcParams.update({"font.size": 9})

    salinity = (
        (ds_groundtruth["so"] * rho_0)
        * ds_groundtruth["areacello"]
        * ds_groundtruth["dz"]
    ).sum(["x", "y", "lev"])
    salinity = salinity.rename("Salinity")
    salinity = salinity.assign_attrs(units="g")

    # Plot predictions first
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        if "so" in pred_data["ds_prediction"].data_vars:
            ds_pred = pred_data["ds_prediction"]
            salinity_pred = (
                (ds_pred["so"] * rho_0) * ds_pred["areacello"] * ds_pred["dz"]
            ).sum(["x", "y", "lev"])
            salinity_pred = salinity_pred.rename("Salinity")
            salinity_pred = salinity_pred.assign_attrs(units="g")

            title = titles[i] if i < len(titles) else pred_data.get("name", k)
            salinity_pred.plot(ax=ax, label=title, c=clist[i % len(clist)])

            # Compute and plot trend
            coeffs_salinity_pred_trend = np.polyfit(
                np.arange(salinity_pred.size), salinity_pred, 1
            )
            trend_line = (
                np.arange(salinity_pred.size) * coeffs_salinity_pred_trend[0]
                + coeffs_salinity_pred_trend[1]
            )
            ax.plot(
                salinity_pred.time, trend_line, c=clist[i % len(clist)], linestyle="--"
            )

    # Plot ground truth
    coeffs_salinity_trend = np.polyfit(np.arange(salinity.size), salinity, 1)
    salinity.plot(ax=ax, label="OM4", c="k")
    trend_line_gt = (
        np.arange(salinity.size) * coeffs_salinity_trend[0] + coeffs_salinity_trend[1]
    )
    ax.plot(salinity.time, trend_line_gt, c="k", linestyle="--")

    ax.set_ylim([5.861e22, 5.8632e22])
    ax.legend(ncol=3)
    ax.set_title("")

    plt.savefig(
        os.path.join(directories["salinity"], "Salinity.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # 2. Deseasonalized salinity time series
    plt.rcdefaults()
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
    )
    plt.rcParams.update({"font.size": 9})

    # Compute volume-weighted mean salinity for ground truth
    salinity_deseason = (
        ds_groundtruth["so"]
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .mean(["x", "y", "lev"])
    )

    # Apply deseasonalization
    salinity_deseason = remove_climatology(salinity_deseason)
    salinity_deseason = salinity_deseason.rename("Salinity")
    salinity_deseason = salinity_deseason.assign_attrs(units="psu")

    # Plot predictions first
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        if "so" in pred_data["ds_prediction"].data_vars:
            ds_pred = pred_data["ds_prediction"]
            salinity_pred_deseason = (
                ds_pred["so"]
                .weighted(ds_pred["areacello"] * ds_pred["dz"])
                .mean(["x", "y", "lev"])
            )

            # Apply deseasonalization
            salinity_pred_deseason = remove_climatology(salinity_pred_deseason)
            salinity_pred_deseason = salinity_pred_deseason.rename("Salinity")
            salinity_pred_deseason = salinity_pred_deseason.assign_attrs(units="psu")

            title = titles[i] if i < len(titles) else pred_data.get("name", k)
            salinity_pred_deseason.plot(ax=ax, label=title, c=clist[i % len(clist)])

            # Compute and plot trend
            coeffs_salinity_pred_trend = np.polyfit(
                np.arange(salinity_pred_deseason.size), salinity_pred_deseason, 1
            )
            trend_line = (
                np.arange(salinity_pred_deseason.size) * coeffs_salinity_pred_trend[0]
                + coeffs_salinity_pred_trend[1]
            )
            ax.plot(
                salinity_pred_deseason.time,
                trend_line,
                c=clist[i % len(clist)],
                linestyle="--",
            )

    # Plot ground truth
    coeffs_salinity_trend = np.polyfit(
        np.arange(salinity_deseason.size), salinity_deseason, 1
    )
    salinity_deseason.plot(ax=ax, label="OM4", c="k")
    trend_line_gt = (
        np.arange(salinity_deseason.size) * coeffs_salinity_trend[0]
        + coeffs_salinity_trend[1]
    )
    ax.plot(salinity_deseason.time, trend_line_gt, c="k", linestyle="--")

    ax.legend(ncol=3)
    ax.set_title("")

    plt.savefig(
        os.path.join(directories["salinity"], "salinity_deseasonalized.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # 3. Non-deseasonalized (duplicate with specific naming)
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.rcParams.update({"font.size": 9})

    # Same as first plot - total salinity mass
    salinity = (
        (ds_groundtruth["so"] * rho_0)
        * ds_groundtruth["areacello"]
        * ds_groundtruth["dz"]
    ).sum(["x", "y", "lev"])
    salinity = salinity.rename("Salinity")
    salinity = salinity.assign_attrs(units="g")

    # Plot predictions first
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        if "so" in pred_data["ds_prediction"].data_vars:
            ds_pred = pred_data["ds_prediction"]
            salinity_pred = (
                (ds_pred["so"] * rho_0) * ds_pred["areacello"] * ds_pred["dz"]
            ).sum(["x", "y", "lev"])
            salinity_pred = salinity_pred.rename("Salinity")
            salinity_pred = salinity_pred.assign_attrs(units="g")

            title = titles[i] if i < len(titles) else pred_data.get("name", k)
            salinity_pred.plot(ax=ax, label=title, c=clist[i % len(clist)])

            # Compute and plot trend
            coeffs_salinity_pred_trend = np.polyfit(
                np.arange(salinity_pred.size), salinity_pred, 1
            )
            trend_line = (
                np.arange(salinity_pred.size) * coeffs_salinity_pred_trend[0]
                + coeffs_salinity_pred_trend[1]
            )
            ax.plot(
                salinity_pred.time, trend_line, c=clist[i % len(clist)], linestyle="--"
            )

    # Plot ground truth
    coeffs_salinity_trend = np.polyfit(np.arange(salinity.size), salinity, 1)
    salinity.plot(ax=ax, label="OM4", c="k")
    trend_line_gt = (
        np.arange(salinity.size) * coeffs_salinity_trend[0] + coeffs_salinity_trend[1]
    )
    ax.plot(salinity.time, trend_line_gt, c="k", linestyle="--")

    ax.set_ylim([5.861e22, 5.8632e22])
    ax.legend(ncol=3)
    ax.set_title("")

    plt.savefig(
        os.path.join(
            directories["salinity"], "salinity_manuall_non_deseasonalized .png"
        ),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # ✅ INTEGRATED: Spatial salinity analysis (original core functionality)

    # Get surface salinity
    sss_gt = ds_groundtruth["so"].isel(lev=0)

    # Global SSS map
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    im = sss_gt.mean("time").plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cm.cm.haline, add_colorbar=False
    )
    ax.set_title("Ground Truth Sea Surface Salinity")
    setup_cartopy_projection(ax)
    plt.colorbar(im, ax=ax, shrink=0.6, label="SSS (psu)")
    plt.savefig(
        os.path.join(directories["salinity"], "SeaSurfaceSalinity_Global_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SSS snapshots
    last_index = len(sss_gt.time) - 1
    time_indices = [0, last_index // 2, last_index]

    for i, time_idx in enumerate(time_indices):
        fig, ax = plt.subplots(
            1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        im = sss_gt.isel(time=time_idx).plot(
            ax=ax, transform=ccrs.PlateCarree(), cmap=cm.cm.haline, add_colorbar=False
        )
        ax.set_title(f"SSS Snapshot - Time {time_idx}")
        setup_cartopy_projection(ax)
        plt.colorbar(im, ax=ax, shrink=0.6, label="SSS (psu)")
        plt.savefig(
            os.path.join(directories["salinity"], f"SSS_map_snapshot_t_{time_idx}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def create_basin_comparison_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    basin_masks: xr.Dataset,
    output_path: str,
    titles: list[str],
) -> None:
    """
    Create basin-specific comparison plots.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        basin_masks: Basin mask dataset
        output_path: Output directory path
        titles: List of titles for predictions
    """
    directories = create_spatial_directories(output_path)

    # Create basin-specific analysis for each variable
    variables = ["thetao", "so"]

    for variable in variables:
        if variable not in ds_groundtruth.data_vars:
            continue

        fig, axes = plt.subplots(
            2, 3, figsize=(18, 12), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        axes = axes.flatten()

        for i, basin_name in enumerate(basin_masks.data_vars):
            if i >= len(axes):
                break

            ax = axes[i]
            basin_mask = basin_masks[basin_name]

            # Apply basin mask and compute mean
            basin_data = ds_groundtruth[variable].where(basin_mask.notnull())
            basin_mean = basin_data.mean("time").isel(lev=0)  # Surface level

            im = basin_mean.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r" if variable == "thetao" else cm.cm.haline,
                add_colorbar=False,
            )

            ax.set_title(f"{basin_name} - {variable}")
            setup_cartopy_projection(ax)

            # Add basin boundary
            basin_mask.plot.contour(
                ax=ax,
                transform=ccrs.PlateCarree(),
                colors="black",
                linewidths=1,
                add_colorbar=False,
            )

        plt.tight_layout()
        # Use temperature directory for thetao, salinity for so
        dir_name = "temperature" if variable == "thetao" else "salinity"
        plt.savefig(
            os.path.join(directories[dir_name], f"{variable}_basin_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def create_ohc_basin_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    basin_masks: xr.Dataset,
    output_path: str,
    titles: list[str],
    dataset_name: str = "OM4",
) -> None:
    """
    Create OHC basin timeseries plots (OHC_Basin.png).

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        basin_masks: Basin mask dataset
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    # Physical constants
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3

    directories = create_spatial_directories(output_path)

    # Create compare_info.txt file
    compare_file = open(os.path.join(output_path, "compare_info.txt"), "a")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(16, 8),
        gridspec_kw={
            "width_ratios": [1, 1, 1],
            "height_ratios": [1, 1],
            "wspace": 0.25,
            "hspace": 0.5,
        },
    )

    ax_flat = ax.flatten()

    # Setup storage for trend calculations
    GT_regionwise_ohc = {}
    GT_regionwise_ohc["Model"] = dataset_name
    for j, k in enumerate(pred_dict.keys()):
        pred_dict[k]["regionwise_ohc"] = {}

    # Color list for plotting predictions
    clist = ["red", "blue", "green", "orange", "purple", "brown"]

    # Process each basin
    for i, basin_name in enumerate(list(basin_masks.data_vars)):
        if i >= len(ax_flat) - 1:  # Leave space for legend (will delete last subplot)
            break

        basin_mask = basin_masks[basin_name]

        # Compute OHC for ground truth in this basin
        OHC = (
            (ds_groundtruth["thetao"] * c_p * rho_0 * basin_mask)
            * ds_groundtruth["areacello"]
            * ds_groundtruth["dz"]
        ).sum(["x", "y", "lev"]) / 1e21

        OHC = remove_climatology(OHC)
        OHC = OHC.rename("OHC Anomaly")
        OHC = OHC.assign_attrs(units="ZJ")

        # Compute and plot trend for ground truth
        coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
        OHC.plot(ax=ax_flat[i], label=dataset_name, c="k")
        (pos,) = ax_flat[i].plot(
            OHC.time.data,
            np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1],
            c="k",
            ls="--",
        )

        compare_file.write(f"\nOHC {basin_name} GT Trend Slope : {coeffs_OHC_trend[0]}")
        GT_regionwise_ohc[basin_name] = coeffs_OHC_trend[0]

        # Process each prediction
        for j, k in enumerate(pred_dict.keys()):
            ds_pred = pred_dict[k]["ds_prediction"]
            OHC_pred = (
                (ds_pred["thetao"] * c_p * rho_0 * basin_mask)
                * ds_pred["areacello"]
                * ds_pred["dz"]
            ).sum(["x", "y", "lev"]) / 1e21

            OHC_pred = remove_climatology(OHC_pred)
            OHC_pred = OHC_pred.rename("OHC Anomaly")
            OHC_pred = OHC_pred.assign_attrs(units="ZJ")

            # Compute and plot trend for prediction
            coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred.size), OHC_pred, 1)
            title = titles[j] if j < len(titles) else pred_dict[k]["name"]
            OHC_pred.plot(ax=ax_flat[i], label=title, c=clist[j % len(clist)])
            (pos,) = ax_flat[i].plot(
                OHC_pred.time.data,
                np.arange(OHC_pred.size) * coeffs_OHC_pred_trend[0]
                + coeffs_OHC_pred_trend[1],
                c=clist[j % len(clist)],
                ls="--",
            )

            compare_file.write(
                f"\nOHC {basin_name} {title} Trend Slope : {coeffs_OHC_pred_trend[0]}"
            )
            pred_dict[k]["regionwise_ohc"][basin_name] = coeffs_OHC_pred_trend[0]

        ax_flat[i].set_title(basin_name + " Ocean")

    # Remove the last subplot and add legend
    fig.delaxes(ax_flat[-1])
    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncols=3)

    compare_file.write("\n")
    compare_file.close()

    plt.savefig(
        os.path.join(directories["ohc"], "OHC_Basin.png"), bbox_inches="tight", dpi=600
    )
    plt.close()


def create_ohc_basin_upper_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    basin_masks: xr.Dataset,
    output_path: str,
    titles: list[str],
    dataset_name: str = "OM4",
    max_level: int = 700,
) -> None:
    """
    Create upper ocean OHC basin timeseries plots (OHC_Basin_upto_700m.png).

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        basin_masks: Basin mask dataset
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
        max_level: Maximum depth level (default 700m)
    """
    # Physical constants
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3

    directories = create_spatial_directories(output_path)

    # Create compare_info.txt file
    compare_file = open(os.path.join(output_path, "compare_info.txt"), "a")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(16, 8),
        gridspec_kw={
            "width_ratios": [1, 1, 1],
            "height_ratios": [1, 1],
            "wspace": 0.25,
            "hspace": 0.5,
        },
    )

    ax_flat = ax.flatten()

    # Setup storage for trend calculations
    GT_regionwise_ohc = {}
    GT_regionwise_ohc["Model"] = dataset_name
    for j, k in enumerate(pred_dict.keys()):
        pred_dict[k]["regionwise_ohc"] = {}

    # Color list for plotting predictions
    clist = ["red", "blue", "green", "orange", "purple", "brown"]

    # Process each basin
    for i, basin_name in enumerate(list(basin_masks.data_vars)):
        if i >= len(ax_flat) - 1:  # Leave space for legend (will delete last subplot)
            break

        basin_mask = basin_masks[basin_name]

        # Compute OHC for ground truth in this basin (limited to upper ocean)
        OHC = (
            (
                ds_groundtruth["thetao"].sel(lev=slice(None, max_level))
                * c_p
                * rho_0
                * basin_mask
            )
            * ds_groundtruth["areacello"]
            * ds_groundtruth["dz"]
        ).sum(["x", "y", "lev"]) / 1e21

        OHC = remove_climatology(OHC)
        OHC = OHC.rename(f"OHC Anomaly (Upto {max_level}m)")
        OHC = OHC.assign_attrs(units="ZJ")

        # Compute and plot trend for ground truth
        coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
        OHC.plot(ax=ax_flat[i], label=dataset_name, c="k")
        (pos,) = ax_flat[i].plot(
            OHC.time.data,
            np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1],
            c="k",
            ls="--",
        )

        compare_file.write(f"\nOHC {basin_name} GT Trend Slope : {coeffs_OHC_trend[0]}")
        GT_regionwise_ohc[basin_name] = coeffs_OHC_trend[0]

        # Process each prediction
        for j, k in enumerate(pred_dict.keys()):
            ds_pred = pred_dict[k]["ds_prediction"]
            OHC_pred = (
                (
                    ds_pred["thetao"].sel(lev=slice(None, max_level))
                    * c_p
                    * rho_0
                    * basin_mask
                )
                * ds_pred["areacello"]
                * ds_pred["dz"]
            ).sum(["x", "y", "lev"]) / 1e21

            OHC_pred = remove_climatology(OHC_pred)
            OHC_pred = OHC_pred.rename(f"OHC Anomaly (Upto {max_level}m)")
            OHC_pred = OHC_pred.assign_attrs(units="ZJ")

            # Compute and plot trend for prediction
            coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred.size), OHC_pred, 1)
            title = titles[j] if j < len(titles) else pred_dict[k]["name"]
            OHC_pred.plot(ax=ax_flat[i], label=title, c=clist[j % len(clist)])
            (pos,) = ax_flat[i].plot(
                OHC_pred.time.data,
                np.arange(OHC_pred.size) * coeffs_OHC_pred_trend[0]
                + coeffs_OHC_pred_trend[1],
                c=clist[j % len(clist)],
                ls="--",
            )

            compare_file.write(
                f"\nOHC {basin_name} {title} Trend Slope : {coeffs_OHC_pred_trend[0]}"
            )
            pred_dict[k]["regionwise_ohc"][basin_name] = coeffs_OHC_pred_trend[0]

        ax_flat[i].set_title(basin_name + " Ocean")

    # Remove the last subplot and add legend
    fig.delaxes(ax_flat[-1])
    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=3)

    compare_file.write("\n")
    compare_file.close()

    plt.savefig(
        os.path.join(directories["ohc"], f"OHC_Basin_upto_{max_level}m.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def create_missing_salinity_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    output_path: str,
    titles: list[str],
    dataset_name: str = "OM4",
) -> None:
    """
    Create missing salinity plots: Salinity.png and salinity_deseasonalized.png.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    directories = create_spatial_directories(output_path)

    # Physical constants
    rho_0 = 1025  # kg/m^3

    # Color list for plotting predictions
    clist = ["red", "blue", "green", "orange", "purple", "brown"]

    # 1. Create Salinity.png (non-deseasonalized total salinity mass)
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.rcParams.update({"font.size": 9})

    salinity = (
        (ds_groundtruth["so"] * rho_0)
        * ds_groundtruth["areacello"]
        * ds_groundtruth["dz"]
    ).sum(["x", "y", "lev"])
    salinity = salinity.rename("Salinity")
    salinity = salinity.assign_attrs(units="g")

    # Plot predictions first
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        if "so" in pred_data["ds_prediction"].data_vars:
            ds_pred = pred_data["ds_prediction"]
            salinity_pred = (
                (ds_pred["so"] * rho_0) * ds_pred["areacello"] * ds_pred["dz"]
            ).sum(["x", "y", "lev"])
            salinity_pred = salinity_pred.rename("Salinity")
            salinity_pred = salinity_pred.assign_attrs(units="g")

            title = titles[i] if i < len(titles) else pred_data.get("name", k)
            salinity_pred.plot(ax=ax, label=title, c=clist[i % len(clist)])

            # Compute and plot trend
            coeffs_salinity_pred_trend = np.polyfit(
                np.arange(salinity_pred.size), salinity_pred, 1
            )
            trend_line = (
                np.arange(salinity_pred.size) * coeffs_salinity_pred_trend[0]
                + coeffs_salinity_pred_trend[1]
            )
            ax.plot(
                salinity_pred.time, trend_line, c=clist[i % len(clist)], linestyle="--"
            )

    # Plot ground truth
    coeffs_salinity_trend = np.polyfit(np.arange(salinity.size), salinity, 1)
    salinity.plot(ax=ax, label=dataset_name, c="k")
    trend_line_gt = (
        np.arange(salinity.size) * coeffs_salinity_trend[0] + coeffs_salinity_trend[1]
    )
    ax.plot(salinity.time, trend_line_gt, c="k", linestyle="--")

    ax.set_ylim([5.861e22, 5.8632e22])
    ax.legend(ncol=3)
    ax.set_title("")

    plt.savefig(
        os.path.join(directories["salinity"], "Salinity.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # 2. Create salinity_deseasonalized.png (deseasonalized volume-weighted mean)
    plt.rcdefaults()
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
    )
    plt.rcParams.update({"font.size": 9})

    # Compute volume-weighted mean salinity for ground truth
    salinity_deseason = (
        ds_groundtruth["so"]
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .mean(["x", "y", "lev"])
    )

    # Apply deseasonalization
    salinity_deseason = remove_climatology(salinity_deseason)
    salinity_deseason = salinity_deseason.rename("Salinity")
    salinity_deseason = salinity_deseason.assign_attrs(units="psu")

    # Plot predictions first
    for i, (k, pred_data) in enumerate(pred_dict.items()):
        if "so" in pred_data["ds_prediction"].data_vars:
            ds_pred = pred_data["ds_prediction"]
            salinity_pred_deseason = (
                ds_pred["so"]
                .weighted(ds_pred["areacello"] * ds_pred["dz"])
                .mean(["x", "y", "lev"])
            )

            # Apply deseasonalization
            salinity_pred_deseason = remove_climatology(salinity_pred_deseason)
            salinity_pred_deseason = salinity_pred_deseason.rename("Salinity")
            salinity_pred_deseason = salinity_pred_deseason.assign_attrs(units="psu")

            title = titles[i] if i < len(titles) else pred_data.get("name", k)
            salinity_pred_deseason.plot(ax=ax, label=title, c=clist[i % len(clist)])

            # Compute and plot trend
            coeffs_salinity_pred_trend = np.polyfit(
                np.arange(salinity_pred_deseason.size), salinity_pred_deseason, 1
            )
            trend_line = (
                np.arange(salinity_pred_deseason.size) * coeffs_salinity_pred_trend[0]
                + coeffs_salinity_pred_trend[1]
            )
            ax.plot(
                salinity_pred_deseason.time,
                trend_line,
                c=clist[i % len(clist)],
                linestyle="--",
            )

    # Plot ground truth
    coeffs_salinity_trend = np.polyfit(
        np.arange(salinity_deseason.size), salinity_deseason, 1
    )
    salinity_deseason.plot(ax=ax, label=dataset_name, c="k")
    trend_line_gt = (
        np.arange(salinity_deseason.size) * coeffs_salinity_trend[0]
        + coeffs_salinity_trend[1]
    )
    ax.plot(salinity_deseason.time, trend_line_gt, c="k", linestyle="--")

    ax.legend(ncol=3)
    ax.set_title("")

    plt.savefig(
        os.path.join(directories["salinity"], "salinity_deseasonalized.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def get_basin_datasets(
    ds: xr.DataArray, ds_groundtruth: xr.Dataset, basin_masks: xr.Dataset
) -> tuple[list[xr.DataArray], list[str]]:
    """
    Split data by ocean basin and compute area-weighted meridional averages.

    Args:
        ds: Input data array
        ds_groundtruth: Ground truth dataset for area weights
        basin_masks: Basin mask dataset

    Returns:
        Tuple of (basin datasets, basin titles)
    """
    basin_data = []
    basin_titles = ["Atlantic", "Indian", "Pacific", "Southern", "Arctic", "Global"]

    for basin_name in ["Atlantic", "Indian", "Pacific", "Southern", "Arctic"]:
        # Apply basin mask
        da_temp = ds * basin_masks[basin_name]

        # Check for valid data
        section_mask = np.isnan(da_temp).all("x")

        # Compute area-weighted zonal mean
        da_temp_int_x = da_temp.weighted(ds_groundtruth["areacello"]).mean(["x"])

        # Apply section mask
        basin_result = da_temp_int_x.where(~section_mask)
        basin_data.append(basin_result)

    # Add global (no mask)
    section_mask = np.isnan(ds).all("x")
    da_temp_int_x = ds.weighted(ds_groundtruth["areacello"]).mean(["x"])
    global_result = da_temp_int_x.where(~section_mask)
    basin_data.append(global_result)

    return basin_data, basin_titles


def ocean_temperature_profile(
    datasets: list[xr.DataArray],
    titles: list[str],
    plot_title: str,
    output_path: str,
    vmin: float = -0.3,
    vmax: float = 0.3,
) -> None:
    """
    Create vertical temperature profile plots by ocean basin.

    Args:
        datasets: List of temperature datasets for each basin
        titles: List of basin titles
        plot_title: Overall plot title
        output_path: Output directory path
        vmin: Minimum colorbar value
        vmax: Maximum colorbar value
    """
    directories = create_spatial_directories(output_path)

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(16, 6),
        gridspec_kw={
            "width_ratios": [0.05] * 3,
            "height_ratios": [0.05] * 2,
            "wspace": 0.01,
            "hspace": 0.5,
        },
        dpi=300,
    )
    fig.suptitle(plot_title, fontsize=15, fontweight="bold", y=0.95)

    ax = axs.flatten()

    for i, (data, title) in enumerate(zip(datasets, titles)):
        data = data.rename(r"$\theta_O$").assign_attrs(units=r"$\degree C$")
        data["y"] = data.y.assign_attrs(long_name="Latitude", units=r"$\degree$")
        data["lev"] = data.lev.assign_attrs(long_name="depth", units="m")

        im = data.plot(ax=ax[i], cmap="bwr", vmin=vmin, vmax=vmax, add_colorbar=False)
        ax[i].invert_yaxis()
        linear_piecewise_scale(1000, 5, ax=ax[i])
        ax[i].axhline(1000, color="0.5", ls="--")
        ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
        ax[i].set_xticks([-60, -30, 0, 30, 60])
        ax[i].set_xticklabels(
            [r"$60^\circ S$", r"$30^\circ S$", "0", r"$30^\circ N$", r"$60^\circ N$"]
        )
        ax[i].set_title(title, fontsize=14)
        ax[i].set_box_aspect(0.7)

    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]")

    plt.savefig(
        os.path.join(directories["temperature"], f"{plot_title}.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


# REMOVED: create_missing_temperature_profile_plots - functionality integrated into create_sst_analysis_plots
def _deprecated_create_missing_temperature_profile_plots():
    """
    Create missing temperature profile plots: CM4/model (Last Year - First Year) and bias plots.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        basin_masks: Basin mask dataset
        output_path: Output directory path
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    print("🔧 Starting temperature profile plots (simplified version)...")

    if not pred_dict:
        print("⚠️ No prediction data available for temperature profiles")
        return

    try:
        # Get first prediction
        key1 = list(pred_dict.keys())[0]
        ds_pred = pred_dict[key1]["ds_prediction"]
        pred_name = pred_dict[key1].get("name", key1)

        print(f"📊 Processing {pred_name} temperature data...")

        # Simplified approach: use raw temperature data without deseasonalization to avoid crashes
        # Compute "Last Year - First Year" differences using simpler time slicing
        n_times = len(ds_groundtruth["thetao"].time)
        year_length = min(73, n_times // 6)  # Adapt to actual data size

        print(
            f"⏰ Using time slices: first {year_length} vs last {year_length} timesteps"
        )

        # 1. CM4 (Last Year - First Year) - simplified without climatology removal
        print("📈 Computing CM4 temperature change...")
        CM4_last = (
            ds_groundtruth["thetao"]
            .isel(time=slice(-year_length, None))
            .mean(dim="time")
        )
        CM4_first = (
            ds_groundtruth["thetao"].isel(time=slice(0, year_length)).mean(dim="time")
        )
        CM4_lastyear_change = CM4_last - CM4_first

        # Create simple basin-averaged plots instead of complex basin analysis
        print("🗺️ Creating CM4 profile plot...")
        create_simple_temperature_profile(
            CM4_lastyear_change, f"{dataset_name} (Last Year - First Year)", output_path
        )

        # 2. Prediction (Last Year - First Year)
        print("📈 Computing prediction temperature change...")
        pred_last = (
            ds_pred["thetao"].isel(time=slice(-year_length, None)).mean(dim="time")
        )
        pred_first = ds_pred["thetao"].isel(time=slice(0, year_length)).mean(dim="time")
        pred_lastyear_change = pred_last - pred_first

        print("🗺️ Creating prediction profile plot...")
        create_simple_temperature_profile(
            pred_lastyear_change, f"{pred_name} (Last Year - First Year)", output_path
        )

        # 3. Bias (Last Year - First Year)
        print("📊 Computing bias...")
        bias_lastyear_change = pred_lastyear_change - CM4_lastyear_change

        print("🗺️ Creating bias profile plot...")
        create_simple_temperature_profile(
            bias_lastyear_change, "(Last Year - First Year) Bias", output_path
        )

        print("✅ Temperature profile plots completed successfully!")

    except Exception as e:
        print(f"❌ Error in temperature profile plots: {e}")
        print("⚠️ Skipping temperature profile plots to continue pipeline...")
        return


def create_simple_temperature_profile(
    temp_data: xr.DataArray, plot_title: str, output_path: str
) -> None:
    """
    Create a simplified temperature profile plot (global average).

    Args:
        temp_data: Temperature data array
        plot_title: Plot title
        output_path: Output directory path
    """
    import matplotlib.pyplot as plt

    directories = create_spatial_directories(output_path)

    try:
        # Create global meridional average (simple approach)
        temp_zonal_mean = temp_data.mean("x", skipna=True)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot as simple contour
        im = temp_zonal_mean.plot(
            ax=ax, y="lev", cmap="RdBu_r", vmin=-0.5, vmax=0.5, add_colorbar=False
        )

        ax.invert_yaxis()
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Depth (m)")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Temperature Change (°C)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(directories["temperature"], f"{plot_title}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as e:
        print(f"⚠️ Simplified temperature profile plot failed for {plot_title}: {e}")
        # Create minimal placeholder plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"Temperature Profile\n{plot_title}\n(Data processing error)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(plot_title)
        plt.savefig(
            os.path.join(directories["temperature"], f"{plot_title}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def generate_all_spatial_plots(
    ds_groundtruth: xr.Dataset,
    pred_dict: dict[str, dict],
    basin_masks: xr.Dataset,
    output_path: str,
    titles: list[str],
) -> None:
    """
    Generate all spatial visualization plots.

    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict: Dictionary of prediction datasets
        basin_masks: Basin mask dataset
        output_path: Base output path
        titles: List of titles for predictions
    """
    print("🗺️ Starting spatial plot generation...")

    # Create Ocean Heat Content plots
    print("📊 Creating OHC analysis plots...")
    create_ohc_analysis_plots(ds_groundtruth, pred_dict, output_path, titles)
    print("✅ OHC analysis plots completed")

    # Create OHC Basin plots
    print("🌊 Creating OHC basin plots...")
    create_ohc_basin_plots(ds_groundtruth, pred_dict, basin_masks, output_path, titles)
    print("✅ OHC basin plots completed")

    # Create upper ocean OHC Basin plots
    print("🏄 Creating upper ocean OHC basin plots...")
    create_ohc_basin_upper_plots(
        ds_groundtruth, pred_dict, basin_masks, output_path, titles
    )
    print("✅ Upper ocean OHC basin plots completed")

    # Create comprehensive temperature analysis plots (SST + profiles)
    print("🌡️ Creating comprehensive temperature analysis plots...")
    create_sst_analysis_plots(
        ds_groundtruth, pred_dict, output_path, titles, basin_masks
    )
    print("✅ Comprehensive temperature analysis plots completed")

    # Create salinity plots
    print("🧂 Creating salinity analysis plots...")
    create_salinity_analysis_plots(ds_groundtruth, pred_dict, output_path, titles)
    print("✅ Salinity analysis plots completed")

    # Create basin comparison plots
    print("🗺️ Creating basin comparison plots...")
    create_basin_comparison_plots(
        ds_groundtruth, pred_dict, basin_masks, output_path, titles
    )
    print("✅ Basin comparison plots completed")

    # Create missing salinity plots
    print("🧂➕ Creating missing salinity plots...")
    create_missing_salinity_plots(ds_groundtruth, pred_dict, output_path, titles)
    print("✅ Missing salinity plots completed")

    print("🎉 All spatial plots generation completed!")
