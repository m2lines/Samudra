"""ENSO visualization module for ocean data."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from typing import Dict, List, Tuple
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm


def nino34_index_xr(sst_data: xr.DataArray, area_weights: xr.DataArray, 
                    dt: int = 5, window: int = 150) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute Niño 3.4 index using xarray.
    
    Args:
        sst_data: Sea surface temperature data
        area_weights: Area weights for spatial averaging
        dt: Time step in days
        window: Window size for rolling mean
        
    Returns:
        Tuple of (Niño 3.4 index, climatology)
    """
    # Niño 3.4 region: 5°N-5°S, 170°W-120°W (190-240 in 0-360 convention)
    nino34_region = sst_data.sel(y=slice(-5, 5), x=slice(190, 240))
    nino34_area = area_weights.sel(y=slice(-5, 5), x=slice(190, 240))
    
    # Calculate area-weighted mean SST in Niño 3.4 region
    nino34_sst = nino34_region.weighted(nino34_area).mean(["y", "x"])
    
    # Calculate climatology
    climatology = nino34_sst.groupby("time.dayofyear").mean("time")
    
    # Calculate anomaly
    nino34_anomaly = nino34_sst.groupby("time.dayofyear") - climatology
    
    # Apply rolling mean
    nino34_smoothed = nino34_anomaly.rolling(time=window, center=True).mean()
    
    return nino34_smoothed, climatology


def plot_map_v2(data: xr.DataArray, ax, title: str, cmap: str = "RdBu_r",
                vmin: float = None, vmax: float = None, 
                add_colorbar: bool = True) -> None:
    """
    Plot a global map with cartopy projection.
    
    Args:
        data: 2D data array to plot
        ax: Matplotlib axis with cartopy projection
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        add_colorbar: Whether to add colorbar
    """
    # Plot the data
    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar
    )
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.set_global()
    ax.set_title(title)
    
    return im


def plot_nino34_index_over_time_v2(nino_gt: xr.DataArray, nino_preds: Dict[str, xr.DataArray],
                                  titles: List[str], colors: List[str],
                                  output_path: str, dataset_name: str = "OM4") -> None:
    """
    Create combined Niño 3.4 index timeseries and map plot.
    
    Args:
        nino_gt: Ground truth Niño 3.4 index
        nino_preds: Dictionary of prediction Niño 3.4 indices
        titles: List of prediction titles
        colors: List of colors for predictions
        output_path: Output file path
        dataset_name: Name of ground truth dataset
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout: top for timeseries, bottom for map
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])
    
    # Top subplot: Niño 3.4 timeseries
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot ground truth
    truth_time_vals = pd.to_datetime([str(t) for t in nino_gt.time.values])
    ax1.plot(truth_time_vals, nino_gt.values, 'k-', linewidth=2, label=dataset_name)
    
    # Plot predictions
    for i, (key, nino_pred) in enumerate(nino_preds.items()):
        color = colors[i % len(colors)]
        title = titles[i] if i < len(titles) else key
        pred_time_vals = pd.to_datetime([str(t) for t in nino_pred.time.values])
        ax1.plot(pred_time_vals, nino_pred.values, color=color, linewidth=2, label=title)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='El Niño threshold')
    ax1.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='La Niña threshold')
    
    ax1.set_ylabel('Niño 3.4 Index (°C)')
    ax1.set_title('Niño 3.4 Index Time Series')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Bottom subplot: Niño 3.4 region map
    ax2 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())
    
    # Create a simple SST field to show the Niño 3.4 region
    # Use the first available data to show the region
    if len(nino_preds) > 0:
        # Get some reference SST data to show the region
        # This is a simplified version - in practice you'd use actual SST data
        lon = np.linspace(0, 360, 360)
        lat = np.linspace(-90, 90, 180)
        dummy_sst = xr.DataArray(
            np.random.normal(20, 5, (180, 360)),
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon']
        )
        
        # Plot base map
        im = dummy_sst.plot(
            ax=ax2, transform=ccrs.PlateCarree(),
            cmap='RdYlBu_r', alpha=0.7, add_colorbar=False
        )
    
    # Highlight Niño 3.4 region
    nino34_lon = [190, 240, 240, 190, 190]
    nino34_lat = [-5, -5, 5, 5, -5]
    ax2.plot(nino34_lon, nino34_lat, 'r-', linewidth=3, transform=ccrs.PlateCarree())
    ax2.fill(nino34_lon, nino34_lat, color='red', alpha=0.2, transform=ccrs.PlateCarree())
    
    # Add map features
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_global()
    ax2.set_title('Niño 3.4 Region (5°N-5°S, 170°W-120°W)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_climatology_plot(ds_groundtruth: xr.Dataset, output_path: str) -> None:
    """
    Create DJF climatology plot for the first 10 years.
    
    Args:
        ds_groundtruth: Ground truth dataset
        output_path: Output file path
    """
    # Check if we have enough data (>120 time steps for 10+ years)
    if len(ds_groundtruth.time) <= 120:
        print("Not enough data for climatology plot (need >120 time steps)")
        return
    
    # Get SST data (use tos if available, otherwise surface thetao)
    if "tos" in ds_groundtruth.data_vars:
        sst_data = ds_groundtruth["tos"]
    else:
        sst_data = ds_groundtruth["thetao"].isel(lev=0)
    
    # Select first 10 years of data
    first_10_years = sst_data.isel(time=slice(0, 120))  # ~10 years at 5-day intervals
    
    # Compute DJF (December-January-February) seasonal mean
    djf_months = first_10_years.where(
        first_10_years.time.dt.month.isin([12, 1, 2]), drop=True
    )
    djf_climatology = djf_months.mean("time")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot climatology
    im = plot_map_v2(
        djf_climatology, ax, 
        "DJF SST Climatology (First 10 Years)",
        cmap="RdYlBu_r", 
        add_colorbar=True
    )
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.6, label="SST (°C)")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_enso_visualizations(ds_groundtruth: xr.Dataset,
                                pred_dict_processed: Dict[str, Dict],
                                output_path: str, colors: List[str], 
                                titles: List[str], dataset_name: str = "OM4") -> None:
    """
    Generate all ENSO visualization plots.
    
    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict_processed: Dictionary of prediction datasets
        output_path: Output directory path
        colors: List of colors for plotting
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    enso_path = os.path.join(output_path, "ENSO")
    os.makedirs(enso_path, exist_ok=True)
    
    # Get SST data
    if "tos" in ds_groundtruth.data_vars:
        sst_data = ds_groundtruth["tos"]
    else:
        sst_data = ds_groundtruth["thetao"].isel(lev=0)
    
    area_weights = ds_groundtruth["areacello"]
    
    # Compute Niño 3.4 indices
    nino_gt, _ = nino34_index_xr(sst_data, area_weights)
    
    nino_preds = {}
    for key, pred_data in pred_dict_processed.items():
        ds_pred = pred_data["ds_prediction"]
        if "tos" in ds_pred.data_vars:
            sst_pred = ds_pred["tos"]
        else:
            sst_pred = ds_pred["thetao"].isel(lev=0)
        
        nino_pred, _ = nino34_index_xr(sst_pred, area_weights)
        nino_preds[key] = nino_pred
    
    # Create Niño 3.4 timeseries + map plot
    nino_output = os.path.join(enso_path, "Nino_Figure_Short_with_map_single.png")
    plot_nino34_index_over_time_v2(
        nino_gt, nino_preds, titles, colors, nino_output, dataset_name
    )
    
    # Create climatology plot
    climatology_output = os.path.join(enso_path, "Climatology.png")
    create_climatology_plot(ds_groundtruth, climatology_output)
    
    print(f"Generated ENSO visualization plots in {enso_path}")