"""Analysis module for ocean visualization."""

import numpy as np
import xarray as xr


def profile_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate area-weighted spatial mean profiles.

    Args:
        ds: Input dataset with area weights

    Returns:
        Dataset with spatial means computed
    """
    return ds.weighted(ds.areacello).mean(["y", "x"])


def process_mask(mask: xr.DataArray, data: xr.Dataset) -> xr.DataArray:
    """
    Process basin mask for plotting.

    Args:
        mask: Basin mask data
        data: Reference dataset for coordinates

    Returns:
        Processed mask with proper coordinates
    """
    mask = mask.where(mask != 0, np.nan)
    mask = mask.transpose("lat", "lon")
    mask = mask.assign_coords(lat=data.y.values, lon=data.x.values)
    mask = mask.rename({"lat": "y", "lon": "x"})
    return mask


def create_basin_masks(basins: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
    """
    Create processed basin masks for analysis.

    Args:
        basins: Raw basin data
        data: Reference dataset for coordinates

    Returns:
        Dataset with processed basin masks
    """
    # Atlantic mask (excluding southern ocean)
    atlantic_mask0 = basins["basin_atlantic"]
    atlantic_mask = atlantic_mask0.where(atlantic_mask0["lat"] >= -32)
    atlantic_mask = process_mask(atlantic_mask, data)

    # Pacific mask (excluding southern ocean)
    pacific_mask0 = basins["basin_pacific"]
    pacific_mask = pacific_mask0.where(pacific_mask0["lat"] >= -32)
    pacific_mask = process_mask(pacific_mask, data)

    # Indian Ocean mask (excluding southern ocean)
    indian_ocean_mask0 = basins["basin_indian"]
    indian_ocean_mask = indian_ocean_mask0.where(indian_ocean_mask0["lat"] >= -32)
    indian_ocean_mask = process_mask(indian_ocean_mask, data)

    # Southern Ocean mask
    southern_ocean_mask0 = basins["basin_southern"]
    southern_ocean_mask = process_mask(southern_ocean_mask0, data)

    # Arctic Ocean mask
    arctic_mask0 = basins["basin_arctic"]
    arctic_ocean_mask = process_mask(arctic_mask0, data)

    return xr.Dataset(
        {
            "Atlantic": atlantic_mask,
            "Pacific": pacific_mask,
            "Southern": southern_ocean_mask,
            "Indian": indian_ocean_mask,
            "Arctic": arctic_ocean_mask,
        }
    )


def get_basin_datasets(ds: xr.Dataset) -> dict[str, xr.Dataset]:
    """
    Extract basin-specific datasets.

    Args:
        ds: Input dataset

    Returns:
        Dictionary of basin-specific datasets
    """
    basin_datasets = {}

    # Assuming basin_masks is available globally (would need to be passed in)
    for basin in ["Atlantic", "Pacific", "Indian", "Southern", "Arctic"]:
        # This would need access to basin_masks - should be passed as parameter
        basin_datasets[basin] = ds  # Placeholder - would apply basin mask

    return basin_datasets


def compute_trends(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    """
    Compute linear trends along a dimension.

    Args:
        ds: Input dataset
        dim: Dimension to compute trends along

    Returns:
        Dataset with trend coefficients
    """

    def linear_trend(y):
        """Compute linear trend coefficient."""
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    return ds.apply(linear_trend, dim=dim)


def compute_mae(pred: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """
    Compute Mean Absolute Error between prediction and truth.

    Args:
        pred: Prediction dataset
        truth: Ground truth dataset

    Returns:
        Dataset with MAE values
    """
    return np.abs(pred - truth).mean()


def compute_rmse(pred: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """
    Compute Root Mean Square Error between prediction and truth.

    Args:
        pred: Prediction dataset
        truth: Ground truth dataset

    Returns:
        Dataset with RMSE values
    """
    return np.sqrt(((pred - truth) ** 2).mean())


def compute_ohc(
    temperature: xr.DataArray,
    areacello: xr.DataArray,
    dz: xr.DataArray,
    c_p: float = 3850,
    rho_0: float = 1025,
    zeta_joules_factor: float = 1e21,
) -> xr.DataArray:
    """
    Compute Ocean Heat Content.

    Args:
        temperature: Temperature data
        areacello: Grid cell areas
        dz: Depth thickness
        c_p: Specific heat capacity of seawater
        rho_0: Reference density
        zeta_joules_factor: Scaling factor for units

    Returns:
        Ocean Heat Content data
    """
    ohc = (
        (temperature * c_p * rho_0 / zeta_joules_factor)
        .weighted(areacello * dz)
        .sum(["lev"])
    )

    return ohc


def nino_index_compute_clim(
    T: xr.DataArray, area: xr.DataArray, dt: int = 5, window: int = 150
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute Niño 3.4 index climatology.

    Args:
        T: Temperature data
        area: Area weights
        dt: Time step in days
        window: Window size for rolling mean

    Returns:
        Tuple of (Niño index, climatology)
    """
    # Niño 3.4 region: 5°N-5°S, 170°W-120°W
    nino34_region = T.sel(
        y=slice(-5, 5),
        x=slice(190, 240),  # Converting from 170°W-120°W to 0-360 convention
    )

    # Calculate area-weighted mean
    nino34_sst = nino34_region.weighted(area).mean(["y", "x"])

    # Calculate climatology
    climatology = nino34_sst.groupby("time.dayofyear").mean("time")

    # Calculate anomaly
    nino34_anomaly = nino34_sst.groupby("time.dayofyear") - climatology

    # Apply rolling mean
    nino34_smoothed = nino34_anomaly.rolling(time=window, center=True).mean()

    return nino34_smoothed, climatology


def detrend_data(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    """
    Remove linear trends from data.

    Args:
        ds: Input dataset
        dim: Dimension to detrend along

    Returns:
        Detrended dataset
    """

    def detrend(y):
        """Remove linear trend from time series."""
        x = np.arange(len(y))
        p = np.polyfit(x, y, 1)
        return y - (p[0] * x + p[1])

    return ds.apply(detrend, dim=dim)


def compute_seasonal_cycle(
    ds: xr.Dataset, groupby_coord: str = "time.month"
) -> xr.Dataset:
    """
    Compute seasonal cycle.

    Args:
        ds: Input dataset
        groupby_coord: Coordinate to group by for seasonal cycle

    Returns:
        Seasonal cycle dataset
    """
    return ds.groupby(groupby_coord).mean("time")


def compute_profile_metrics(
    pred_profiles: dict[str, xr.Dataset], truth_profile: xr.Dataset
) -> dict[str, dict[str, xr.Dataset]]:
    """
    Compute profile-based metrics for all predictions.

    Args:
        pred_profiles: Dictionary of prediction profiles
        truth_profile: Ground truth profile

    Returns:
        Dictionary of metrics for each prediction
    """
    metrics = {}

    for key, pred_profile in pred_profiles.items():
        metrics[key] = {
            "mae": compute_mae(pred_profile, truth_profile),
            "rmse": compute_rmse(pred_profile, truth_profile),
            "bias": (pred_profile - truth_profile).mean(),
        }

    return metrics


def compute_basin_statistics(
    ds: xr.Dataset, basin_masks: xr.Dataset
) -> dict[str, xr.Dataset]:
    """
    Compute statistics for each ocean basin.

    Args:
        ds: Input dataset
        basin_masks: Basin mask dataset

    Returns:
        Dictionary of basin statistics
    """
    basin_stats = {}

    for basin_name in basin_masks.data_vars:
        basin_mask = basin_masks[basin_name]
        basin_data = ds.where(basin_mask.notnull())

        basin_stats[basin_name] = {
            "mean": basin_data.weighted(ds.areacello).mean(["y", "x"]),
            "std": basin_data.weighted(ds.areacello).std(["y", "x"]),
            "trend": compute_trends(basin_data.weighted(ds.areacello).mean(["y", "x"])),
        }

    return basin_stats


def compute_depth_integrated_quantities(
    ds: xr.Dataset, variable: str, max_depth: float = None
) -> xr.DataArray:
    """
    Compute depth-integrated quantities.

    Args:
        ds: Input dataset
        variable: Variable to integrate
        max_depth: Maximum depth for integration

    Returns:
        Depth-integrated quantity
    """
    var_data = ds[variable]

    if max_depth is not None:
        var_data = var_data.sel(lev=slice(0, max_depth))

    # Integrate over depth using depth thickness weights
    integrated = (var_data * ds.dz).sum("lev")

    return integrated
