# %%
"""
Regrid basin_arctic.nc data to match the grid from om4.zarr using xESMF.

This script follows the approach from the MOM6 Analysis Cookbook:
https://mom6-analysiscookbook.readthedocs.io/en/latest/notebooks/Horizontal_Remapping.html
"""

import xarray as xr
import xesmf as xe
import numpy as np
from scipy.spatial import KDTree


# %%
def load_source_data():
    """Load the source basin data."""
    print("Loading source data from ../temp/basin_Arctic.nc...")
    ds = xr.open_dataset("../temp/basin_Arctic.nc")
    print(f"Source data shape: {ds.basin.shape}")
    print(f"Source lon range: {ds.lon.min().values:.2f} to {ds.lon.max().values:.2f}")
    print(f"Source lat range: {ds.lat.min().values:.2f} to {ds.lat.max().values:.2f}")
    return ds


# %%
def load_target_grid():
    """Load the target grid from om4.zarr."""
    print("\nLoading target grid from ../temp/om4.zarr...")
    ds = xr.open_zarr("../temp/om4.zarr")

    # Create a minimal grid dataset with just the coordinates we need
    target_grid = xr.Dataset(
        {
            "lon": ds.x,  # x coordinate represents longitude
            "lat": ds.y,  # y coordinate represents latitude
        }
    )

    print(f"Target grid shape: lat={len(target_grid.lat)}, lon={len(target_grid.lon)}")
    print(
        f"Target lon range: {target_grid.lon.min().values:.2f} to {target_grid.lon.max().values:.2f}"
    )
    print(
        f"Target lat range: {target_grid.lat.min().values:.2f} to {target_grid.lat.max().values:.2f}"
    )

    return target_grid


# %%
def regrid_with_kdtree(source_ds, target_grid):
    """
    Regrid source data to target grid using KDTree for nearest neighbor lookup.

    This approach directly maps each target grid point to its nearest source point,
    handling NaNs by finding the closest non-NaN value.
    """
    print("\nRegridding using KDTree nearest neighbor lookup...")

    # Get source data
    source_data = source_ds.basin.values
    source_lons, source_lats = np.meshgrid(source_ds.lon.values, source_ds.lat.values)

    # Find valid (non-NaN) source points
    valid_mask = ~np.isnan(source_data)

    if not np.any(valid_mask):
        raise ValueError("All source data points are NaN - cannot regrid")

    # Create arrays of valid source points and their values
    valid_points = np.column_stack((source_lons[valid_mask], source_lats[valid_mask]))
    valid_values = source_data[valid_mask]

    print(
        f"Found {len(valid_values)} valid source points out of {source_data.size} total"
    )

    # Build KDTree for efficient nearest neighbor search
    print("Building KDTree...")
    tree = KDTree(valid_points)

    # Create target grid points
    target_lons, target_lats = np.meshgrid(
        target_grid.lon.values, target_grid.lat.values
    )
    target_points = np.column_stack((target_lons.flatten(), target_lats.flatten()))

    print(f"Querying {len(target_points)} target points...")
    # Find nearest neighbors for all target points
    distances, indices = tree.query(target_points, k=1)

    # Get values for target points
    target_values = valid_values[indices]

    # Reshape to target grid shape
    regridded_data = target_values.reshape(target_lons.shape)

    # Create output DataArray
    basin_regridded = xr.DataArray(
        regridded_data,
        coords={
            "lat": target_grid.lat,
            "lon": target_grid.lon,
        },
        dims=["lat", "lon"],
        name="basin",
    )

    # Add attributes
    basin_regridded.attrs.update(source_ds.basin.attrs)
    basin_regridded.attrs["regrid_method"] = "nearest_neighbor_kdtree"
    basin_regridded.attrs["regridded_from"] = "basin_Arctic.nc"
    basin_regridded.attrs["regridded_to"] = "om4.zarr grid"
    basin_regridded.attrs["nan_handling"] = (
        "NaNs excluded from source, nearest valid value used"
    )

    print(f"Regridded data shape: {basin_regridded.shape}")
    print(
        f"Data range: {basin_regridded.min().values:.2f} to {basin_regridded.max().values:.2f}"
    )
    print(f"Mean distance to nearest source point: {distances.mean():.4f} degrees")

    return basin_regridded


# %%
def save_regridded_data(
    basin_regridded, output_path="../temp/basin_arctic_regridded.nc"
):
    """Save the regridded data to a new NetCDF file."""
    print(f"\nSaving regridded data to {output_path}...")

    # Create a new dataset with the regridded data
    ds_out = xr.Dataset({"basin": basin_regridded})

    # Add global attributes
    ds_out.attrs["title"] = "Arctic basin mask regridded to OM4 half-degree grid"
    ds_out.attrs["source"] = "basin_Arctic.nc"
    ds_out.attrs["regrid_method"] = "nearest_neighbor_kdtree"
    ds_out.attrs["created_by"] = "regrid_basins.py"

    # Save to NetCDF
    ds_out.to_netcdf(output_path)
    print(f"Data saved successfully to {output_path}")

    return ds_out


# %%
# Load the data
print("=== Step 1: Loading Data ===")
source_ds = load_source_data()
target_grid = load_target_grid()

# %%
# Regrid the basin data
print("\n=== Step 2: Regridding Data ===")
basin_regridded = regrid_with_kdtree(source_ds, target_grid)

# %%
# Save the regridded data
print("\n=== Step 3: Saving Results ===")
output_ds = save_regridded_data(basin_regridded)

print("\n=== Regridding completed successfully! ===")
print("\nRegridded dataset summary:")
print(output_ds)

# %%
# Create comparison plot
print("\n=== Step 4: Creating Comparison Plot ===")
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original data
    source_ds.basin.plot(ax=ax1, x="lon", y="lat", cmap="tab10")
    ax1.set_title("Original Basin Data (1° grid)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Regridded data
    basin_regridded.plot(ax=ax2, x="lon", y="lat", cmap="tab10")
    ax2.set_title("Regridded Basin Data (0.5° grid)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig("./temp/basin_regrid_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to ./temp/basin_regrid_comparison.png")
    plt.show()

except ImportError:
    print("\nMatplotlib not available - skipping comparison plot")

# %%
# Summary information
print("\n=== Summary ===")
print(f"Original grid: {source_ds.basin.shape} (lat x lon)")
print(f"Target grid: {basin_regridded.shape} (lat x lon)")
print(f"Original resolution: ~{abs(source_ds.lat[1] - source_ds.lat[0]).values:.1f}°")
print(f"Target resolution: ~{abs(target_grid.lat[1] - target_grid.lat[0]).values:.1f}°")
print(f"Regridding method: nearest_neighbor_kdtree")
print(f"Output file: ./temp/basin_arctic_regridded.nc")
