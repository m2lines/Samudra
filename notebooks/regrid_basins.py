# %%
"""
Regrid all basin data to match the grid from om4.zarr using xESMF.

This script:
1. Loads all basin files from BASINS_PATH
2. Combines them into a single grid with unique integer IDs per basin
3. Regrids the combined data using KDTree nearest neighbor
4. Splits the regridded data back into per-basin boolean grids
5. Saves all basin masks as variables in a zarr file

This approach is more efficient than regridding each basin separately.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

BASINS_PATH = Path("/Users/jder/oa/data/basins/")
HIGH_RES_DATA_PATH = Path("/Users/jder/oa/data/half_deg_10y/")

# Define basin names and their IDs
BASIN_INFO = {
    "Arctic": {"file": "basin_Arctic.nc", "id": 1},
    "Atlantic": {"file": "basin_At_noArctic.nc", "id": 2},
    "Indian": {"file": "basin_In.nc", "id": 3},
    "Pacific": {"file": "basin_Pa.nc", "id": 4},
    "Southern": {"file": "basin_SO_32S.nc", "id": 5},
}


# %%
def load_all_basin_data():
    """Load all basin data and combine into a single grid with unique IDs."""
    print("Loading all basin data...")

    combined_data = None
    reference_coords = None

    for basin_name, info in BASIN_INFO.items():
        file_path = BASINS_PATH / info["file"]
        print(f"  Loading {basin_name} from {info['file']}...")

        ds = xr.open_dataset(file_path)

        # Store reference coordinates from first file
        if reference_coords is None:
            reference_coords = {"lon": ds.lon, "lat": ds.lat}
            combined_data = np.zeros_like(ds.basin.values, dtype=np.int32)

        # Verify coordinates match across files
        if not (
            np.allclose(ds.lon.values, reference_coords["lon"].values)
            and np.allclose(ds.lat.values, reference_coords["lat"].values)
        ):
            raise ValueError(
                f"Coordinates in {info['file']} don't match reference grid"
            )

        # Add basin data with unique ID where basin exists (non-NaN and non-zero)
        basin_mask = ~np.isnan(ds.basin.values) & (ds.basin.values != 0)
        combined_data[basin_mask] = info["id"]

        print(f"    Found {np.sum(basin_mask)} points for {basin_name}")

    # Ensure we have data
    if combined_data is None or reference_coords is None:
        raise ValueError("No basin data was loaded")

    # Create combined dataset
    combined_ds = xr.Dataset(
        {
            "basin_id": (
                ["lat", "lon"],
                combined_data,
                {
                    "long_name": "Combined basin IDs",
                    "description": "Integer ID for each basin: "
                    + ", ".join(
                        [f"{info['id']}={name}" for name, info in BASIN_INFO.items()]
                    ),
                    "valid_range": [0, len(BASIN_INFO)],
                },
            )
        },
        coords=reference_coords,
    )

    print(f"\nCombined basin data:")
    print(f"  Grid shape: {combined_data.shape}")
    print(f"  Total basin points: {np.sum(combined_data > 0)}")
    basin_points = combined_data[combined_data > 0]
    if len(basin_points) > 0:
        print(f"  Basin ID range: {np.min(basin_points)} to {np.max(basin_points)}")

    return combined_ds


# %%
def load_target_grid():
    """Load the target grid from om4.zarr."""
    print("\nLoading target grid from OM4.zarr...")
    ds = xr.open_zarr(HIGH_RES_DATA_PATH / "OM4.zarr")

    # Create a minimal grid dataset with just the coordinates we need
    # Rename dimensions to match our expected naming convention
    target_grid = xr.Dataset(
        {
            "lon": ds.x.rename({"x": "lon"}),  # x coordinate represents longitude
            "lat": ds.y.rename({"y": "lat"}),  # y coordinate represents latitude
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
def regrid_combined_basins(source_ds, target_grid):
    """
    Regrid combined basin data to target grid using KDTree for nearest neighbor lookup.
    """
    print("\nRegridding combined basin data using KDTree...")

    # Get source data
    source_data = source_ds.basin_id.values
    source_lons, source_lats = np.meshgrid(source_ds.lon.values, source_ds.lat.values)

    # Include ALL source points (both basin and non-basin/land)
    # This ensures that points closest to land/non-basin areas remain unassigned
    valid_mask = ~np.isnan(source_data)  # Only exclude NaN values, include zeros

    if not np.any(valid_mask):
        raise ValueError("All source data points are NaN - cannot regrid")

    # Create arrays of all valid source points and their values (including zeros)
    valid_points = np.column_stack((source_lons[valid_mask], source_lats[valid_mask]))
    valid_values = source_data[valid_mask]

    basin_points = np.sum(valid_values > 0)
    zero_points = np.sum(valid_values == 0)
    print(
        f"Found {len(valid_values)} total source points: {basin_points} basin points + {zero_points} non-basin points"
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
    basin_id_regridded = xr.DataArray(
        regridded_data,
        coords={
            "lat": target_grid.lat,
            "lon": target_grid.lon,
        },
        dims=["lat", "lon"],
        name="basin_id",
        attrs={
            "long_name": "Regridded basin IDs",
            "description": "Integer ID for each basin: "
            + ", ".join([f"{info['id']}={name}" for name, info in BASIN_INFO.items()]),
            "regrid_method": "nearest_neighbor_kdtree",
            "regridded_from": "combined basin files",
            "regridded_to": "om4.zarr grid",
            "valid_range": [1, len(BASIN_INFO)],
        },
    )

    print(f"Regridded data shape: {basin_id_regridded.shape}")
    print(
        f"Basin ID range: {basin_id_regridded.min().values} to {basin_id_regridded.max().values}"
    )
    print(f"Mean distance to nearest source point: {distances.mean():.4f} degrees")

    return basin_id_regridded


# %%
def create_boolean_basin_masks(basin_id_regridded):
    """
    Split the regridded basin ID data into separate boolean masks for each basin.
    Only points with positive basin IDs are included; zero values represent land/non-basin areas.
    """
    print("\nCreating boolean masks for each basin...")

    basin_masks = {}

    for basin_name, info in BASIN_INFO.items():
        basin_id = info["id"]

        # Create boolean mask for this basin (only positive IDs, zeros remain as False)
        mask = basin_id_regridded == basin_id

        # Add attributes
        mask.attrs = {
            "long_name": f"{basin_name} basin mask",
            "description": f"Boolean mask for {basin_name} basin (1=in basin, 0=not in basin or land)",
            "basin_id": basin_id,
            "regrid_method": "nearest_neighbor_kdtree",
            "source": f"basin_{basin_name}.nc via combined regridding",
            "note": "Zero values in source represent land/non-basin areas and remain unassigned",
        }

        basin_masks[f"basin_{basin_name.lower()}"] = mask

        # Print statistics
        n_points = int(mask.sum().values)
        total_points = mask.size
        zero_points = int((basin_id_regridded == 0).sum().values)
        percentage = (n_points / total_points) * 100

        print(f"  {basin_name:>10}: {n_points:>7} points ({percentage:>5.2f}% of grid)")

    # Print statistics about non-basin (land/zero) points
    zero_points = int((basin_id_regridded == 0).sum().values)
    total_points = basin_id_regridded.size
    zero_percentage = (zero_points / total_points) * 100
    print(
        f"  {'Land/Non-basin':>10}: {zero_points:>7} points ({zero_percentage:>5.2f}% of grid)"
    )

    return basin_masks


# %%
def save_basin_masks_zarr(
    basin_masks, target_grid, output_path=BASINS_PATH / "basin_masks_regridded.zarr"
):
    """Save all basin masks to a single zarr file."""
    print(f"\nSaving basin masks to {output_path}...")

    # Create dataset with all basin masks
    ds_out = xr.Dataset(basin_masks)

    # Add coordinate information
    ds_out = ds_out.assign_coords({"lon": target_grid.lon, "lat": target_grid.lat})

    # Add global attributes
    ds_out.attrs.update(
        {
            "title": "Basin masks regridded to OM4 half-degree grid",
            "description": "Boolean masks for ocean basins regridded from 1° to 0.5° resolution",
            "source_files": ", ".join([info["file"] for info in BASIN_INFO.values()]),
            "regrid_method": "nearest_neighbor_kdtree",
            "created_by": "regrid_basins.py",
            "basins_included": ", ".join(BASIN_INFO.keys()),
            "grid_resolution": "0.5 degree",
            "coordinate_system": "WGS84",
        }
    )

    # Save to zarr (remove existing if it exists)
    import shutil

    if Path(output_path).exists():
        shutil.rmtree(output_path)

    ds_out.to_zarr(output_path, mode="w")
    print(f"Basin masks saved successfully to {output_path}")

    return ds_out


# %%
def create_comparison_plot(basin_masks, output_path="../temp/basin_comparison.png"):
    """Create a comparison plot showing original vs regridded basin data."""
    print(f"\nCreating comparison plot...")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Load original Arctic basin for comparison
        original_ds = xr.open_dataset(BASINS_PATH / "basin_Arctic.nc")
        original_mask = ~np.isnan(original_ds.basin.values) & (
            original_ds.basin.values != 0
        )

        # Get regridded Arctic basin
        regridded_mask = basin_masks["basin_arctic"]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot original data
        im1 = ax1.imshow(
            original_mask.astype(float),
            extent=[
                original_ds.lon.min(),
                original_ds.lon.max(),
                original_ds.lat.min(),
                original_ds.lat.max(),
            ],
            cmap="Blues",
            origin="lower",
            aspect="auto",
        )
        ax1.set_title(
            f"Original Arctic Basin\n({original_ds.basin.shape[0]}×{original_ds.basin.shape[1]} grid, ~1° resolution)"
        )
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.grid(True, alpha=0.3)

        # Plot regridded data
        im2 = ax2.imshow(
            regridded_mask.values.astype(float),
            extent=[
                regridded_mask.lon.min(),
                regridded_mask.lon.max(),
                regridded_mask.lat.min(),
                regridded_mask.lat.max(),
            ],
            cmap="Blues",
            origin="lower",
            aspect="auto",
        )
        ax2.set_title(
            f"Regridded Arctic Basin\n({regridded_mask.shape[0]}×{regridded_mask.shape[1]} grid, ~0.5° resolution)"
        )
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.grid(True, alpha=0.3)

        # Add colorbars
        plt.colorbar(im1, ax=ax1, label="Basin mask", shrink=0.8)
        plt.colorbar(im2, ax=ax2, label="Basin mask", shrink=0.8)

        # Add statistics text
        orig_points = int(np.sum(original_mask))
        regrid_points = int(regridded_mask.sum().values)

        fig.suptitle(
            f"Basin Regridding Comparison\n"
            f"Original: {orig_points:,} points | "
            f"Regridded: {regrid_points:,} points | "
            f"Ratio: {regrid_points / orig_points:.2f}×",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {output_path}")
        plt.show()

        return True

    except ImportError:
        print("Matplotlib not available - skipping comparison plot")
        return False
    except Exception as e:
        print(f"Error creating plot: {e}")
        return False


# %%
# Main execution
if __name__ == "__main__":
    # Load and combine all basin data
    print("=== Step 1: Loading and Combining Basin Data ===")
    combined_source = load_all_basin_data()

    # Load target grid
    print("\n=== Step 2: Loading Target Grid ===")
    target_grid = load_target_grid()

    # Regrid the combined basin data
    print("\n=== Step 3: Regridding Combined Data ===")
    basin_id_regridded = regrid_combined_basins(combined_source, target_grid)

    # Create boolean masks for each basin
    print("\n=== Step 4: Creating Boolean Basin Masks ===")
    basin_masks = create_boolean_basin_masks(basin_id_regridded)

    # Save all masks to zarr
    print("\n=== Step 5: Saving Results ===")
    output_ds = save_basin_masks_zarr(basin_masks, target_grid)

    # Create comparison plot
    print("\n=== Step 6: Creating Comparison Plot ===")
    create_comparison_plot(basin_masks)

    print("\n=== Basin Regridding Completed Successfully! ===")
    print(f"\nFinal dataset summary:")
    print(output_ds)

    print(f"\nBasin coverage summary:")
    for var_name in output_ds.data_vars:
        if str(var_name).startswith("basin_"):
            mask = output_ds[var_name]
            n_points = int(mask.sum().values)
            total_points = mask.size
            percentage = (n_points / total_points) * 100
            basin_name = str(var_name).replace("basin_", "").title()
            print(
                f"  {basin_name:>10}: {n_points:>7} points ({percentage:>5.2f}% of grid)"
            )
