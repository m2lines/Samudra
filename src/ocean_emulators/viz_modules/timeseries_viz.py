"""Time series visualization module for ocean data."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from typing import Dict, List, Tuple, Optional
import pandas as pd


def create_timeseries_directories(output_path: str, variables: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Create directories for time series outputs.
    
    Args:
        output_path: Base output path
        variables: List of variables to create directories for
        
    Returns:
        Dictionary of directory paths
    """
    if variables is None:
        variables = ["thetao", "so", "uo", "vo", "zos"]
    
    directories = {}
    
    # Create main timeseries directory
    timeseries_path = os.path.join(output_path, "Timeseries")
    os.makedirs(timeseries_path, exist_ok=True)
    directories["timeseries"] = timeseries_path
    
    # Create subdirectories for each variable
    for var in variables:
        var_path = os.path.join(timeseries_path, f"{var}_timeseries")
        os.makedirs(var_path, exist_ok=True)
        directories[f"{var}_timeseries"] = var_path
    
    return directories


def plot_single_variable_timeseries(profile_groundtruth: xr.Dataset, 
                                   pred_profiles: Dict[str, xr.Dataset],
                                   variable: str, level: int, 
                                   output_path: str, 
                                   var_labels: Dict[str, str],
                                   colors: List[str],
                                   titles: List[str]) -> None:
    """
    Plot time series for a single variable at a specific level.
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles
        variable: Variable name to plot
        level: Depth level index
        output_path: Output directory path
        var_labels: Dictionary of variable labels
        colors: List of colors for plotting
        titles: List of titles for each prediction
    """
    if variable not in profile_groundtruth.data_vars:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot ground truth
    if "lev" in profile_groundtruth[variable].dims:
        truth_data = profile_groundtruth[variable].isel(lev=level)
    else:
        truth_data = profile_groundtruth[variable]
    # Convert cftime to pandas datetime for matplotlib compatibility
    time_vals = pd.to_datetime([str(t) for t in truth_data.time.values])
    ax.plot(time_vals, truth_data, 'k-', linewidth=2, label='Ground Truth')
    
    # Plot predictions
    for i, (key, pred_profile) in enumerate(pred_profiles.items()):
        if variable in pred_profile.data_vars:
            if "lev" in pred_profile[variable].dims:
                pred_data = pred_profile[variable].isel(lev=level)
            else:
                pred_data = pred_profile[variable]
            color = colors[i % len(colors)]
            title = titles[i] if i < len(titles) else key
            # Convert cftime to pandas datetime for matplotlib compatibility
            pred_time_vals = pd.to_datetime([str(t) for t in pred_data.time.values])
            ax.plot(pred_time_vals, pred_data, color=color, linewidth=2, label=title)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(var_labels.get(variable, variable))
    ax.set_title(f'{variable} at depth level {level}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{level}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_variable_timeseries(profile_groundtruth: xr.Dataset, 
                                pred_profiles: Dict[str, xr.Dataset],
                                directories: Dict[str, str],
                                var_labels: Dict[str, str],
                                colors: List[str],
                                titles: List[str],
                                variables: Optional[List[str]] = None) -> None:
    """
    Plot time series for all variables at all levels.
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles
        directories: Dictionary of output directory paths
        var_labels: Dictionary of variable labels
        colors: List of colors for plotting
        titles: List of titles for each prediction
        variables: List of variables to plot (defaults to all available)
    """
    if variables is None:
        variables = ["thetao", "so", "uo", "vo", "zos"]
    
    for variable in variables:
        if variable not in profile_groundtruth.data_vars:
            continue
            
        var_output_path = directories.get(f"{variable}_timeseries")
        if not var_output_path:
            continue
        
        # Get number of levels for this variable
        if "lev" in profile_groundtruth[variable].dims:
            n_levels = profile_groundtruth[variable].sizes["lev"]
            for level in range(n_levels):
                plot_single_variable_timeseries(
                    profile_groundtruth, pred_profiles, variable, level,
                    var_output_path, var_labels, colors, titles
                )
        else:
            # For 2D variables (no depth dimension)
            plot_single_variable_timeseries(
                profile_groundtruth, pred_profiles, variable, 0,
                var_output_path, var_labels, colors, titles
            )


def plot_multi_variable_grid(profile_groundtruth: xr.Dataset, 
                            pred_profiles: Dict[str, xr.Dataset],
                            variables: List[str], 
                            depth_levels: List[int],
                            output_path: str,
                            var_labels: Dict[str, str],
                            colors: List[str],
                            titles: List[str],
                            plot_title: str = "Multi-variable Time Series") -> None:
    """
    Create a grid plot showing multiple variables at different depths.
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles
        variables: List of variables to plot
        depth_levels: List of depth levels to plot
        output_path: Output file path
        var_labels: Dictionary of variable labels
        colors: List of colors for plotting
        titles: List of titles for each prediction
        plot_title: Title for the entire plot
    """
    n_vars = len(variables)
    n_levels = len(depth_levels)
    
    fig, axes = plt.subplots(n_vars, n_levels, figsize=(4*n_levels, 3*n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    if n_levels == 1:
        axes = axes.reshape(-1, 1)
    
    for i, variable in enumerate(variables):
        for j, level in enumerate(depth_levels):
            ax = axes[i, j]
            
            if variable in profile_groundtruth.data_vars:
                # Plot ground truth
                if "lev" in profile_groundtruth[variable].dims:
                    truth_data = profile_groundtruth[variable].isel(lev=level)
                else:
                    truth_data = profile_groundtruth[variable]
                # Convert cftime to pandas datetime for matplotlib compatibility
                truth_time_vals = pd.to_datetime([str(t) for t in truth_data.time.values])
                ax.plot(truth_time_vals, truth_data, 'k-', linewidth=2, label='Ground Truth')
                
                # Plot predictions
                for k, (key, pred_profile) in enumerate(pred_profiles.items()):
                    if variable in pred_profile.data_vars:
                        if "lev" in pred_profile[variable].dims:
                            pred_data = pred_profile[variable].isel(lev=level)
                        else:
                            pred_data = pred_profile[variable]
                        color = colors[k % len(colors)]
                        title = titles[k] if k < len(titles) else key
                        # Convert cftime to pandas datetime for matplotlib compatibility
                        pred_time_vals = pd.to_datetime([str(t) for t in pred_data.time.values])
                        ax.plot(pred_time_vals, pred_data, color=color, linewidth=2, label=title)
            
            ax.set_title(f'{variable} (level {level})')
            ax.set_ylabel(var_labels.get(variable, variable))
            ax.grid(True, alpha=0.3)
            
            if i == n_vars - 1:  # Only add x-label to bottom row
                ax.set_xlabel('Time')
            
            if i == 0 and j == 0:  # Only add legend to first subplot
                ax.legend()
    
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temperature_profile_comparison(datasets: List[xr.Dataset], 
                                      titles: List[str], 
                                      output_path: str,
                                      plot_title: str = "Temperature Profiles",
                                      vmin: float = -0.3, 
                                      vmax: float = 0.3) -> None:
    """
    Plot temperature profiles for comparison.
    
    Args:
        datasets: List of datasets to compare
        titles: List of titles for each dataset
        output_path: Output file path
        plot_title: Title for the plot
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4*n_datasets, 6))
    
    if n_datasets == 1:
        axes = [axes]
    
    for i, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[i]
        
        if "thetao" in ds.data_vars:
            # Take time mean for profile plot
            temp_profile = ds["thetao"].mean("time")
            
            # Check available dimensions and create appropriate plot
            if "lev" in temp_profile.dims:
                if len(temp_profile.dims) == 1:
                    # 1D profile (depth only)
                    ax.plot(temp_profile.values, temp_profile.lev.values)
                    ax.set_xlabel("Temperature (°C)")
                    ax.set_ylabel("Depth (m)")
                    ax.invert_yaxis()
                else:
                    # 2D plot (depth vs spatial)
                    im = temp_profile.plot(
                        ax=ax, 
                        y="lev", 
                        vmin=vmin, 
                        vmax=vmax, 
                        cmap="RdBu_r",
                        add_colorbar=i == n_datasets-1
                    )
                    ax.invert_yaxis()
            else:
                # No depth dimension, create simple plot
                if len(temp_profile.dims) == 0:
                    # Scalar value
                    ax.text(0.5, 0.5, f"{temp_profile.values:.2f}°C", 
                           ha='center', va='center', transform=ax.transAxes)
                else:
                    # Plot whatever dimensions are available
                    temp_profile.plot(ax=ax)
            
            ax.set_title(title)
    
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_timeseries_summary_plot(profile_groundtruth: xr.Dataset, 
                                  pred_profiles: Dict[str, xr.Dataset],
                                  output_path: str,
                                  var_labels: Dict[str, str],
                                  colors: List[str],
                                  titles: List[str],
                                  variables: Optional[List[str]] = None) -> None:
    """
    Create a summary plot showing key time series.
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles
        output_path: Output file path
        var_labels: Dictionary of variable labels
        colors: List of colors for plotting
        titles: List of titles for each prediction
        variables: List of variables to include in summary
    """
    # Select key variables and levels to show
    if variables is None:
        key_variables = ["thetao", "so", "zos"]
    else:
        # Use the provided variables, but limit to reasonable number for summary
        key_variables = variables[:3]  # Max 3 for readable summary
    
    key_levels = [0, 5, 10]  # Surface, mid-depth, deep
    
    plot_multi_variable_grid(
        profile_groundtruth, pred_profiles, key_variables, key_levels,
        output_path, var_labels, colors, titles,
        "Ocean Variables Time Series Summary"
    )


def create_global_timeseries_plots(ds_groundtruth: xr.Dataset,
                                   pred_dict_processed: Dict[str, Dict],
                                   output_path: str,
                                   colors: List[str],
                                   titles: List[str],
                                   dataset_name: str = "OM4") -> None:
    """
    Create global volume-weighted timeseries plots for temperature and salinity.
    
    Args:
        ds_groundtruth: Ground truth dataset
        pred_dict_processed: Dictionary of prediction datasets
        output_path: Output directory path
        colors: List of colors for plotting
        titles: List of titles for predictions
        dataset_name: Name of the ground truth dataset
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    
    timeseries_path = os.path.join(output_path, "Timeseries")
    Days_to_Eq = 0  # Days to equilibration
    
    # Global Temperature Timeseries
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 1, figsize=(16, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5})
    
    # Compute global volume-weighted mean temperature
    thetao = (
        ds_groundtruth["thetao"]
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .mean(["x", "y", "lev"])
    )
    thetao = thetao.rename(r"$\theta_O$")
    thetao = thetao.assign_attrs(units=r"$\degree C$")
    
    # Plot predictions
    for i, (k, pred_data) in enumerate(pred_dict_processed.items()):
        thetao_pred = (
            pred_data["ds_prediction"]["thetao"]
            .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
            .mean(["x", "y", "lev"])
        )
        color = colors[i % len(colors)]
        title = titles[i] if i < len(titles) else k
        # Convert cftime to pandas datetime for matplotlib compatibility
        pred_time_vals = pd.to_datetime([str(t) for t in thetao_pred.time.values])
        ax.plot(pred_time_vals, thetao_pred, color=color, linewidth=2, label=title)
        
        # Compute trend
        coeffs_ = np.polyfit(
            np.arange(thetao_pred[Days_to_Eq:].size), thetao_pred[Days_to_Eq:], 1
        )
        print(f"{title}: {coeffs_[0] * 73}")
    
    # Plot ground truth
    truth_time_vals = pd.to_datetime([str(t) for t in thetao.time.values])
    ax.plot(truth_time_vals, thetao, 'k-', linewidth=2, label=dataset_name)
    
    # Compute ground truth trend
    coeffs_OHC_ground_trend = np.polyfit(
        np.arange(thetao[Days_to_Eq:].size), thetao[Days_to_Eq:], 1
    )
    print("Thetao GT: ", coeffs_OHC_ground_trend[0] * 73)
    
    ax.legend(ncol=3)
    ax.set_title("")
    ax.set_xlabel('Time')
    ax.set_ylabel(r"$\theta_O$ ($\degree C$)")
    plt.savefig(
        os.path.join(timeseries_path, "Global_Thetao_Timeseries.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()
    
    # Global Salinity Timeseries
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 1, figsize=(16, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5})
    
    # Compute global volume-weighted mean salinity
    salinity = (
        ds_groundtruth["so"]
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .mean(["x", "y", "lev"])
    )
    salinity = salinity.rename("S")
    salinity = salinity.assign_attrs(units="psu")
    
    # Plot predictions
    for i, (k, pred_data) in enumerate(pred_dict_processed.items()):
        salinity_pred = (
            pred_data["ds_prediction"]["so"]
            .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
            .mean(["x", "y", "lev"])
        )
        color = colors[i % len(colors)]
        title = titles[i] if i < len(titles) else k
        # Convert cftime to pandas datetime for matplotlib compatibility  
        pred_time_vals = pd.to_datetime([str(t) for t in salinity_pred.time.values])
        ax.plot(pred_time_vals, salinity_pred, color=color, linewidth=2, label=title)
        
        # Compute trend
        coeffs_ = np.polyfit(
            np.arange(salinity_pred[Days_to_Eq:].size), salinity_pred[Days_to_Eq:], 1
        )
        print(f"{title}: {coeffs_[0] * 73}")
    
    # Plot ground truth
    truth_time_vals = pd.to_datetime([str(t) for t in salinity.time.values])
    ax.plot(truth_time_vals, salinity, 'k-', linewidth=2, label=dataset_name)
    
    # Compute ground truth trend
    coeffs_salinity_ground_trend = np.polyfit(
        np.arange(salinity[Days_to_Eq:].size), salinity[Days_to_Eq:], 1
    )
    print("Salinity GT: ", coeffs_salinity_ground_trend[0] * 73)
    
    ax.legend(ncol=3, loc="upper left")
    ax.set_ylim([34.72, 34.73])
    ax.set_title("")
    ax.set_xlabel('Time')
    ax.set_ylabel("S (psu)")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.savefig(
        os.path.join(timeseries_path, "Global_Salinity_Timeseries.png"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def plot_temperature_timeseries_grid_shallow_both(profile_groundtruth: xr.Dataset,
                                                pred_profiles: Dict[str, xr.Dataset],
                                                output_path: str,
                                                colors: List[str],
                                                titles: List[str],
                                                dataset_name: str = "OM4") -> None:
    """
    Create 1x2 grid showing temperature timeseries at shallow depths (2.5m, 775m).
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles  
        output_path: Output directory path
        colors: List of colors for plotting
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    shallow_levels = [2.5, 775]
    plt.rcParams.update({"font.size": 14})
    
    cols = 2
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3))
    axes = axes.flatten()
    
    plot_idx = 0
    variable = "thetao"
    
    if variable in profile_groundtruth.data_vars:
        for lev in shallow_levels:
            ax = axes[plot_idx]
            
            # Find closest depth level
            if "lev" in profile_groundtruth[variable].dims:
                closest_lev_idx = np.argmin(np.abs(profile_groundtruth.lev.values - lev))
                
                # Plot ground truth
                truth_data = profile_groundtruth[variable].isel(lev=closest_lev_idx)
                truth_time_vals = pd.to_datetime([str(t) for t in truth_data.time.values])
                ax.plot(truth_time_vals, truth_data, 'k-', linewidth=2, label=dataset_name)
                mins, maxs = ax.get_ylim()
                
                # Plot predictions
                for i, (key, pred_profile) in enumerate(pred_profiles.items()):
                    if variable in pred_profile.data_vars:
                        pred_closest_lev_idx = np.argmin(np.abs(pred_profile.lev.values - lev))
                        pred_data = pred_profile[variable].isel(lev=pred_closest_lev_idx)
                        color = colors[i % len(colors)]
                        title = titles[i] if i < len(titles) else key
                        pred_time_vals = pd.to_datetime([str(t) for t in pred_data.time.values])
                        ax.plot(pred_time_vals, pred_data, color=color, linewidth=2, label=title)
                
                # Styling
                ax.set_title(f"{lev}m" + r" $\theta_O$", fontsize=14)
                ax.set_xlabel("Time")
                ax.set_ylabel(r"$\theta_O$ [$\degree C$]")
                
                plot_idx += 1
    
    # Layout and legend
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.455, 0.9), ncols=3)
    
    # Save
    timeseries_path = os.path.join(output_path, "Timeseries")
    os.makedirs(timeseries_path, exist_ok=True)
    output_file = os.path.join(timeseries_path, "temperature_timeseries_grid_shallow_both.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    plt.close()


def plot_temp_timeseries_grid_shallow_skipped(profile_groundtruth: xr.Dataset,
                                             pred_profiles: Dict[str, xr.Dataset],
                                             output_path: str,
                                             colors: List[str],
                                             titles: List[str],
                                             dataset_name: str = "OM4") -> None:
    """
    Create 1x3 grid showing temperature timeseries at skipped shallow depths (2.5m, 105m, 550m).
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles
        output_path: Output directory path
        colors: List of colors for plotting
        titles: List of titles for predictions
        dataset_name: Name of ground truth dataset
    """
    shallow_levels = [2.5, 105, 550]
    num_shallow_levels = len(shallow_levels)
    plt.rcParams.update({"font.size": 12})
    
    cols = 3
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    axes = axes.flatten()
    
    plot_idx = 0
    variable = "thetao"
    var_labels = {"thetao": r"$T$ $( ^\circ C )$"}
    
    if variable in profile_groundtruth.data_vars:
        for lev in shallow_levels:
            ax = axes[plot_idx]
            
            # Find closest depth level
            if "lev" in profile_groundtruth[variable].dims:
                closest_lev_idx = np.argmin(np.abs(profile_groundtruth.lev.values - lev))
                
                # Plot ground truth
                truth_data = profile_groundtruth[variable].isel(lev=closest_lev_idx)
                truth_time_vals = pd.to_datetime([str(t) for t in truth_data.time.values])
                ax.plot(truth_time_vals, truth_data, 'k-', linewidth=2, label=dataset_name)
                min_val, max_val = ax.get_ylim()
                
                # Plot predictions
                for i, (key, pred_profile) in enumerate(pred_profiles.items()):
                    if variable in pred_profile.data_vars:
                        pred_closest_lev_idx = np.argmin(np.abs(pred_profile.lev.values - lev))
                        pred_data = pred_profile[variable].isel(lev=pred_closest_lev_idx)
                        color = colors[i % len(colors)]
                        title = titles[i] if i < len(titles) else key
                        pred_time_vals = pd.to_datetime([str(t) for t in pred_data.time.values])
                        ax.plot(pred_time_vals, pred_data, color=color, linewidth=2, label=title)
                
                # Y-axis adjustment for temperature
                if variable == "thetao":
                    ax.set_ylim(min_val - 0.25, max_val + 0.25)
                
                # Title with actual depth value
                actual_depth = profile_groundtruth.lev.values[closest_lev_idx]
                ax.set_title(
                    r"$\theta_O$" + f" at {actual_depth:.1f}m" + r" ($\degree C$)",
                    fontsize=14,
                )
                ax.set_xlabel("Time")
                ax.set_ylabel(var_labels[variable])
                
                plot_idx += 1
    
    # Layout and legend
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncols=3, 
               bbox_to_anchor=(0.34, -0.03), fontsize=12)
    
    # Save
    timeseries_path = os.path.join(output_path, "Timeseries")
    os.makedirs(timeseries_path, exist_ok=True)
    output_file = os.path.join(timeseries_path, "temp_timeseries_grid_shallow_skipped.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    plt.close()


def generate_all_timeseries_plots(profile_groundtruth: xr.Dataset, 
                                 pred_profiles: Dict[str, xr.Dataset],
                                 output_path: str,
                                 var_labels: Dict[str, str],
                                 colors: List[str],
                                 titles: List[str],
                                 variables: Optional[List[str]] = None,
                                 ds_groundtruth: Optional[xr.Dataset] = None,
                                 pred_dict_processed: Optional[Dict[str, Dict]] = None) -> None:
    """
    Generate all time series plots.
    
    Args:
        profile_groundtruth: Ground truth profile data
        pred_profiles: Dictionary of prediction profiles
        output_path: Base output path
        var_labels: Dictionary of variable labels
        colors: List of colors for plotting
        titles: List of titles for each prediction
        variables: List of variables to plot (defaults to all available)
    """
    # Use provided variables or default
    if variables is None:
        variables = ["thetao", "so", "uo", "vo", "zos"]
    
    # Create directories only for the variables we're plotting
    directories = create_timeseries_directories(output_path, variables)
    
    # Plot individual variable time series
    plot_all_variable_timeseries(
        profile_groundtruth, pred_profiles, directories,
        var_labels, colors, titles, variables
    )
    
    # Create summary plots
    summary_path = os.path.join(directories["timeseries"], "timeseries_grid_shallow_all_vars.png")
    create_timeseries_summary_plot(
        profile_groundtruth, pred_profiles, summary_path,
        var_labels, colors, titles, variables
    )
    
    # Create temperature profiles comparison
    datasets = [profile_groundtruth] + list(pred_profiles.values())
    dataset_titles = ["Ground Truth"] + titles
    profile_path = os.path.join(directories["timeseries"], "temperature_profiles_comparison.png")
    plot_temperature_profile_comparison(
        datasets, dataset_titles, profile_path
    )
    
    # Create global timeseries plots if raw datasets are provided
    if ds_groundtruth is not None and pred_dict_processed is not None:
        create_global_timeseries_plots(
            ds_groundtruth, pred_dict_processed, output_path, colors, titles
        )
    
    # Create the missing timeseries grid plots
    plot_temperature_timeseries_grid_shallow_both(
        profile_groundtruth, pred_profiles, output_path, colors, titles
    )
    
    plot_temp_timeseries_grid_shallow_skipped(
        profile_groundtruth, pred_profiles, output_path, colors, titles
    )