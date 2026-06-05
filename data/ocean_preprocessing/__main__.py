# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The module CLI entrypoint; routines for processing OM4 and CM4 datasets."""

import os

# Set blosc to single-threaded mode globally to avoid thread safety issues
# This prevents blosc decompression errors when multiple workers read from blosc-compressed Zarr stores
# Must be set before any blosc operations occur
os.environ["BLOSC_NTHREADS"] = "1"
os.environ["BLOSC_NOLOCK"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLOSC_TRACE"] = "1"

import datetime
import logging
import sys
import warnings
from typing import Literal

import fire
import fsspec
import xarray as xr

from ocean_preprocessing.dataset_validation import ds_processed_validate
from ocean_preprocessing.plotting import rotated_vectors_qc_plots
from ocean_preprocessing.preprocessing import (
    account_for_partial_depths,
    flatten_by_depth_level,
    horizontal_regrid,
    rotate_vectors,
    spatially_filter,
)
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import om4_preprocessing
from ocean_preprocessing.utils import get_git_url_hash

logger = logging.getLogger("ocean_preprocessing")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy event loop warnings from distributed
logging.getLogger("distributed.core").setLevel(logging.WARNING)
logging.getLogger("distributed.nanny").setLevel(logging.WARNING)


Cluster = Literal["off", "local", "gateway", "kube", "coiled", "slurm"]


def init_cluster(cluster: Cluster, **cluster_opts):
    """Initialize and return a Dask distributed cluster client.

    Args:
        cluster: Type of cluster to create ('off', 'local', 'kube', 'slurm', or 'coiled').
        **cluster_opts: Additional keyword arguments passed to the cluster constructor.

    Returns:
        A Dask distributed Client connected to the cluster, or None if cluster='off'.

    Note:
        For Coiled clusters, automatically passes AWS and blosc environment variables to workers.
    """
    match cluster:
        case "off":
            return None
        case "local":
            from dask.distributed import LocalCluster as MyCluster
        case "kube":
            # TODO(#81): Experiment with deploying on these targets.
            from dask.distributed import KubeCluster as MyCluster
        case "slurm":
            # TODO(#81): Experiment with deploying on these targets.
            from dask.distributed import Slurm as MyCluster
        case "coiled":
            from coiled import Cluster as MyCluster  # type: ignore

            target_env_vars = [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "FSSPEC_S3_ENDPOINT_URL",
                "AWS_REQUEST_CHECKSUM_CALCULATION",
                "AWS_RESPONSE_CHECKSUM_VALIDATION",
                "BLOSC_NTHREADS",
                "BLOSC_NOLOCK",
                "BLOSC_TRACE",
                "NUMEXPR_NUM_THREADS",
            ]
            env_vars = {v: os.environ[v] for v in target_env_vars if v in os.environ}
            cluster_opts.update(dict(environ=env_vars))
        case _:
            raise ValueError(f"Invalid cluster option {cluster}.")

    cluster = MyCluster(**cluster_opts)
    return cluster.get_client()


FILTER_SCALE_BY_GRID = {
    "gaussian_grid_45_by_90": 72,  # 4°
    "gaussian_grid_90_by_180": 36,  # 2°
    "gaussian_grid_180_by_360": 18,  # 1°
    "gaussian_grid_360_by_720": 9,  # 0.5°
    "gaussian_grid_720_by_1440": 1,  # 0.25°
}


def spatial_filter_by_grid(
    ds: xr.Dataset, nc_grid_path: str, filter_scale: None | int
) -> xr.Dataset:
    """Apply spatial filtering with automatic filter scale detection.

    Args:
        ds: Dataset to filter. Must contain a 'wetmask' variable.
        nc_grid_path: Path to target grid file. Basename is used to determine filter scale.
        filter_scale: Optional explicit filter scale. If None, automatically determined from grid.

    Returns:
        Spatially filtered dataset.

    Raises:
        ValueError: If filter scale cannot be determined from grid path.
    """
    if filter_scale is None:
        basename = os.path.basename(nc_grid_path)
        base = os.path.splitext(basename)[0]
        if base not in FILTER_SCALE_BY_GRID:
            raise ValueError(
                f"We cannot estimate filter_scale from the input grid: {nc_grid_path}."
            )
        filter_scale = FILTER_SCALE_BY_GRID[base]

    if filter_scale == 1:
        warnings.warn(
            "Are you sure you want filtering on? `filter_scale=1`, either by guess or set manually. "
            "Typically, this means that you wouldn't want to filter at all! Please consider running "
            "--skip_spatial_filtering."
        )

    return spatially_filter(ds, ds.wetmask, filter_scale=filter_scale)


class CLI:
    """Data engineering routines to prepare ocean simulation datasets for ML.

    Two sub-commands are available:
        om4  -- This pre-processes the OM4 dataset, which is the ocean-only component of
          CMIP6, or the Climate Intercomparison Model 6.
        cm4  -- This pre-processes the CM4 dataset, which an ocean datasets from CMIP6
          that is coupled with the atmosphere.

    Each sub-command comes with standard arguments (explained below). These commands are
    designed to run on data authenticated by environment variable (or config file) via
    `fsspec`. For example, when using s3fs, or S3's file-system spec, you can set the
    following environment variables:
    ```
    export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
    # Check with M2LInES project management for how to get the OSN Access keys.
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    ```

    Args:
        output_path: Path where the processed dataset will be saved as a Zarr store.
            This path can be local (e.g., '/path/to/output.zarr') or remote
            (e.g., 's3://bucket/output.zarr', 'gs://bucket/output.zarr'). Data is
            written in Zarr format v2 with consolidated metadata.
        skip_validation: If True, skips schema validation checks on the processed
            dataset. Validation ensures data conforms to expected structure and types.
            Default is False (validation is performed).
        skip_spatial_filtering: If True, skips the spatial filtering step with the
            Gaussian kernel. Produces raw (unfiltered) output. Default is False
            (filtering is applied).
        skip_regridding: If True, skips horizontal regridding and keeps data on the
            native grid resolution. Useful for producing native-resolution outputs.
            Default is False (regridding is performed).
        skip_flattening: If True, keeps 3D variables (with depth dimension) in their
            original form. If False, flattens 3D variables by creating separate 2D
            variables for each depth level (e.g., thetao_0, thetao_1, etc.).
            Default is False (flattening is performed).
        account_for_partial_depths: If True, applies preprocessing to account for partial
            depth levels using native grid information. Requires native_grid_path to be
            provided. Default is False (partial depth processing is skipped).
        dry_run: If True, prints the output dataset structure without writing to disk.
            Useful for testing pipeline configuration. Default is False.
        small_run: If True, limits processing to the first 10 time steps only. Useful
            for quick testing and development. Default is False (processes all time steps).
        write_retries: Number of times the distributed scheduler retries a failed task
            during the final Zarr write. Guards against transient chunk-read failures
            such as the intermittent blosc `-1` decompression error from truncated S3
            reads (numcodecs#810). Only applies when running on a cluster. Default 5.
        cluster: Type of Dask cluster to use for distributed computation. Options are:
            'off' (no cluster, single-threaded), 'local' (LocalCluster), 'kube'
            (KubeCluster), 'slurm' (SlurmCluster), 'coiled' (Coiled cluster).
            Default is 'off'.
        cluster_opts: Additional keyword arguments passed to the Dask cluster
            constructor. For example, for LocalCluster: n_workers=4, threads_per_worker=2.
    """

    def __init__(
        self,
        output_path: str,
        skip_validation: bool = False,
        skip_spatial_filtering: bool = False,
        skip_regridding: bool = False,
        skip_flattening: bool = False,
        account_for_partial_depths: bool = False,
        dry_run: bool = False,
        small_run: bool = False,
        write_retries: int = 5,
        cluster: Cluster = "off",
        **cluster_opts,
    ):
        """Common arguments for the CLI."""
        self.output_path = output_path
        self.skip_validation = skip_validation
        self.skip_spatial_filtering = skip_spatial_filtering
        self.skip_regridding = skip_regridding
        self.skip_flattening = skip_flattening
        self.account_for_partial_depths = account_for_partial_depths
        self.dry_run = dry_run
        self.small_run = small_run
        self.write_retries = write_retries
        self.dask_client = init_cluster(cluster, **cluster_opts)

    def _collect(self, ds: xr.Dataset):
        """Finalize and write the processed dataset to disk.

        Args:
            ds: The processed dataset to write. Should already be chunked appropriately.

        Note:
            Respects dry_run and small_run flags.
        """
        if self.small_run:
            ds = ds.isel(time=slice(0, 10))
        if self.dry_run:
            logger.info(self.dask_client.compute(ds))
            return

        logger.info(f"writing dataset to {self.output_path}")

        delayed = ds.to_zarr(
            self.output_path,
            mode="w",
            consolidated=True,
            encoding={
                var_name: {"compressor": None} for var_name in ds.data_vars.keys()
            },  # Compression turned off
            compute=False,
        )
        # Reading blosc-compressed source chunks over S3 occasionally returns a
        # truncated buffer, surfacing as an intermittent
        # `RuntimeError: error during blosc decompression: -1` (numcodecs#810) that
        # otherwise kills the whole job near completion. The failure is transient --
        # re-fetching the chunk almost always succeeds -- so retry the failed task
        # rather than abort. `retries` is a distributed-scheduler feature; fall back
        # to a plain compute when running without a cluster.
        if self.dask_client is not None:
            self.dask_client.compute(delayed, retries=self.write_retries, sync=True)
        else:
            delayed.compute()
        logger.info("zarr write complete")

    def om4(
        self,
        zarr_data_path: str,
        native_grid_path: str,
        nc_mosaic_path: str,
        target_grid_path: str,
        ocean_static_path: str | None = None,
        spatial_filter_scale: None | int = None,
    ) -> None:
        """Process the OM4 oceans dataset (the ocean component of CMIP).

        This method performs the complete preprocessing pipeline for OM4 ocean data:

        1. Model-specific preprocessing (interpolate velocities to tracer grid)
        2. Account for partial depths (if --account-for-partial-depths)
        3. Validation of preprocessed dataset (unless --skip-validation)
        4. Optionally merge static ocean variables (if ocean_static_path provided)
        5. Vector rotation to zonal/meridional coordinates
        6. QC plots for rotated vectors (unless --skip-plots)
        7. Spatial filtering with Gaussian kernel (unless --skip-spatial-filtering)
        8. Horizontal regridding to target resolution (unless --skip-regridding)
        9. Type conversion to float32
        10. Restore variable attributes and add provenance metadata
        11. Optionally flatten 3D variables by depth level (unless --skip-flattening)
        12. Drop extraneous dimensions (we only need 'x', 'y', 'time' and maybe 'lev').
        13. Write output to Zarr store in Zarr v2 format.

        > NOTE: We pre-determine the spatial filtering scale based on the target grid!
        > We make informed guesses for what the scale should be. If you'd like to
        > override these, please use the --spatial-filter-scale option.

        Please also consider the CLI's common options (--skip-validation, --skip-regridding,
        --skip-flattening, --account-for-partial-depths, --dry-run, --small-run, etc.).

        Args:
            zarr_data_path: Path to the raw OM4 ocean data in Zarr format. Can be local
                (e.g., '/path/to/OM4_raw.zarr') or remote (e.g., 'gs://bucket/OM4_raw.zarr').
                This should contain ocean variables like thetao, so, uo, vo on the native
                tripolar grid.
            native_grid_path: Path to the native grid file in Zarr or NetCDF format. Used
                during preprocessing and for partial depth calculations (if enabled).
                Should contain grid metadata on the native resolution.
            nc_mosaic_path: Path to the ocean horizontal grid (mosaic) file in Zarr or
                NetCDF format (e.g., 'ocean_hgrid.zarr'). This file contains the supergrid
                information needed to extract rotation angles and coordinate bounds for
                the native grid.
            target_grid_path: Path to the target grid file in Zarr or NetCDF format (e.g.,
                'gaussian_grid_360_by_720.zarr'). This file defines the output grid resolution
                and is used for horizontal regridding (unless --skip-regridding). The basename
                is also used to determine the appropriate spatial filter scale.
            ocean_static_path: Optional path to a Zarr file containing static ocean variables
                on the native grid. If provided, variables 'wet' (renamed to 'sea_surface_fraction')
                and 'hfgeou' (geothermal heat flux) will be added to the processed dataset.
                Default is None (no static variables added).
            spatial_filter_scale: Optional integer to override the automatic spatial filter
                scale determination. When spatial filtering is performed (not --skip-spatial-filtering),
                this value will be used instead of the scale inferred from the target grid name.
                By default (None), the scale is automatically estimated from the target grid basename.
        """
        logger.info("preprocessing.")
        ds_processed = om4_preprocessing(
            zarr_data_path, native_grid_path, nc_mosaic_path
        )
        if self.small_run:
            logger.info("**small-run**: filtering data to 10 time steps.")
            ds_processed = ds_processed.isel(time=slice(0, 10))

        if self.account_for_partial_depths:
            logger.info("Apply processing to account for partial depths levels.")
            if native_grid_path.endswith(".zarr"):
                native_grid_ds = xr.open_zarr(native_grid_path, chunks={}).load()
            else:
                with fsspec.open(native_grid_path) as f:
                    native_grid_ds = xr.open_dataset(f).load()
            ds_processed = account_for_partial_depths(ds_processed, native_grid_ds)

        if not self.skip_validation:
            logger.info("validating preprocessing.")
            ds_processed_validate(ds_processed, deep=True)

        if ocean_static_path is not None:
            logger.info("adding static variables.")
            ocean_static_names = ["wet", "hfgeou"]
            ocean_static_renaming = {
                "xh": "x",
                "yh": "y",
                "wet": "sea_surface_fraction",
            }
            ds_static = xr.open_zarr(ocean_static_path, consolidated=True, chunks={})[
                ocean_static_names
            ].rename(ocean_static_renaming)
            ds_processed = xr.merge([ds_processed, ds_static])

        saved_attrs = {}
        for var in ds_processed.data_vars:
            saved_attrs[var] = ds_processed[var].attrs

        logger.info("rechunking for vector rotation.")
        # Rechunk to avoid chunk explosion in xr.dot()
        # Use larger spatial chunks to reduce task graph size
        ds_processed = ds_processed.chunk({"x": -1, "y": -1, "time": 1})

        logger.info("rotating vectors.")
        u_rotated, v_rotated = rotate_vectors(
            ds_processed.uo, ds_processed.vo, ds_processed.angle
        )
        tau_uo_rotated, tau_vo_rotated = rotate_vectors(
            ds_processed.tauuo, ds_processed.tauvo, ds_processed.angle
        )
        if not self.skip_validation:
            logger.info("validating rotated vectors (uo + vo)")
            rotated_vectors_qc_plots(
                ds_processed.uo, ds_processed.vo, u_rotated, v_rotated
            )
            logger.warning(
                "¡the next rotated vectors are labeled u and v, but they are actually tau_uo and tau_vo!"
            )
            rotated_vectors_qc_plots(
                ds_processed.tauuo, ds_processed.tauvo, tau_uo_rotated, tau_vo_rotated
            )

        ds_processed["uo"], ds_processed["vo"] = u_rotated, v_rotated
        ds_processed["tauuo"], ds_processed["tauvo"] = tau_uo_rotated, tau_vo_rotated

        time_blocks = {"x": -1, "y": -1, "time": 1}

        if not self.skip_spatial_filtering:
            filter_blocks = time_blocks.copy()
            if "lev" in ds_processed.dims:
                filter_blocks.update({"lev": 1})
            elif "ilev" in ds_processed.dims:
                filter_blocks.update({"ilev": 1})

            logger.info("rechunking for spatial filtering.")
            # Spatial filtering requires entire spatial domain in single chunks
            # Keep time chunked for memory efficiency
            ds_processed = ds_processed.chunk(filter_blocks)

            logger.info("apply spatial filtering.")
            ds_filtered = spatial_filter_by_grid(
                ds_processed, target_grid_path, spatial_filter_scale
            )
        else:
            ds_filtered = ds_processed

        if not self.skip_regridding:
            logger.info("regridding horizontally.")

            if target_grid_path.endswith(".zarr"):
                ds_target_grid = xr.open_zarr(target_grid_path, chunks={}).load()
            else:
                with fsspec.open(target_grid_path) as f:
                    ds_target_grid = xr.open_dataset(f).load()

            ds_target_grid = ds_target_grid.rename(
                {
                    "grid_x": "x_b",
                    "grid_y": "y_b",
                    "grid_xt": "x",
                    "grid_yt": "y",
                    "grid_lon": "lon_b",
                    "grid_lat": "lat_b",
                    "grid_lont": "lon",
                    "grid_latt": "lat",
                }
            )

            # Update time_blocks to also combine depth level into a single block.
            if "lev" in ds_processed.dims:
                time_blocks.update({"lev": -1})
            elif "ilev" in ds_processed.dims:
                time_blocks.update({"ilev": -1})

            # Ensure optimal chunking for regridding to prevent large task graphs
            logger.info("rechunking for regridding: time=1, full spatial domain")
            ds_filtered = ds_filtered.chunk(time_blocks)

            logger.info("loading grid coordinates for regridding.")
            coord_names = ["lon", "lat", "lon_b", "lat_b"]
            for co in coord_names:
                ds_filtered[co] = ds_filtered[co].astype("float64").compute()

            ds_regridded = horizontal_regrid(ds_filtered, ds_target_grid)
        else:
            ds_regridded = ds_filtered

        # TODO(alxmrs): Gate this by a flag
        logger.info("casting to float32")
        ds_input = ds_regridded.astype("float32")

        logger.info("fixing attributes")
        # Resume attributes
        for var, attr in saved_attrs.items():
            ds_input.attrs[var] = attr
        # Add provenance hash and metadata
        ds_input.attrs["m2lines/ocean_emulators_git_hash"] = get_git_url_hash()
        ds_input.attrs["m2lines/date_created"] = datetime.datetime.now().isoformat()
        ds_input.attrs["m2lines/cli_args"] = " ".join(sys.argv)
        # Label wetmask via attrs
        if len(ds_input["wetmask"].attrs) == 0:
            ds_input["wetmask"].attrs["long_name"] = "ocean mask"
            ds_input["wetmask"].attrs["units"] = "0 if land, 1 if ocean"

        dims_to_keep = ["x", "y", "time"]
        if not self.skip_flattening:
            logger.info("flattening variables by depth level.")
            ds = flatten_by_depth_level(ds_input)
        else:
            ds = ds_input
            if "lev" in ds.dims:
                dims_to_keep.append("lev")
            elif "ilev" in ds.dims:
                dims_to_keep.append("ilev")

        logger.info(f"dropping extraneous dimensions (keeping {dims_to_keep}).")
        ds = ds.drop_dims([x for x in list(ds.dims) if x not in dims_to_keep])

        logger.info("preparing final chunks for output zarr (time=1)")
        ds = ds.chunk({"time": 1})

        logger.info("collecting!")
        self._collect(ds)
        logger.info("done!")

    def cm4(self):
        """Process the CM4 oceans dataset (a coupled ocean model from CMIP)."""
        raise NotImplementedError("Not yet implemented, please come back later!")


if __name__ == "__main__":
    fire.Fire(CLI, name="ocean_preprocessing")
