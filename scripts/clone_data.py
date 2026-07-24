# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Script to clone remote Oceans Emulator data locally."""
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray[io]",
#   "zarr<3",
#   "dask",
#   "requests",
#   "aiohttp",
#   "gcsfs",
#   "numcodecs>=0.15",
#   "distributed",
#   "tenacity",
#   "samudra",
# ]
#
# [tool.uv.sources]
# samudra = { path = "../" }
# ///

import argparse
import os
import pathlib

import dask
import dask.diagnostics
import xarray as xr
from dask.distributed import LocalCluster
from tenacity import retry

from samudra.constants import DatasetSpec, build_om4_spec
from samudra.utils.data import compact_dataset, with_level_index_vars

DEFAULT_DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_onedeg/"


@retry
def robust_open_dataset(target: str, **kwargs) -> xr.Dataset:
    return xr.open_dataset(target, **kwargs)


def select_om4_variables(data: xr.Dataset, dataset_spec: DatasetSpec) -> xr.Dataset:
    """Select model variables from flattened OM4 data before downloading them.

    The public unfiltered OM4 stores encode depth values in variable names such
    as ``thetao_lev_2_5``. Normalize those names to Samudra's level-index form,
    select only the requested prognostic and boundary variables, and retain the
    full wet-mask representation needed by the data loader.
    """
    data = with_level_index_vars(data, dataset_spec=dataset_spec)
    requested = dataset_spec.prognostic_var_names + dataset_spec.boundary_var_names
    missing = sorted(set(requested).difference(data.variables))
    if missing:
        raise ValueError(
            "Selected OM4 variables are absent after normalizing depth names: "
            f"{missing}"
        )

    if dataset_spec.mask_all_levels_var in data.variables:
        masks = [dataset_spec.mask_all_levels_var]
    elif all(name in data.variables for name in dataset_spec.mask_vars):
        masks = list(dataset_spec.mask_vars)
    else:
        raise ValueError(
            "Selected OM4 data must contain wetmask or every level-wise mask"
        )
    return data[requested + masks]


def rechunk_for_output(data: xr.Dataset, time_steps: int) -> xr.Dataset:
    """Apply output Dask chunks without inheriting source Zarr chunk metadata."""
    if time_steps <= 0:
        raise ValueError("Output time chunk size must be positive")

    chunks = {"time": time_steps}
    if "lev" in data.dims:
        chunks["lev"] = data.sizes["lev"]
    data = data.chunk(chunks)

    # xarray gives an inherited Zarr ``encoding['chunks']`` precedence over the
    # new Dask chunks. Remove only chunk-layout metadata so compression, dtype,
    # fill values, and other source encodings are preserved.
    for variable in data.variables.values():
        variable.encoding.pop("chunks", None)
        variable.encoding.pop("chunksizes", None)
        variable.encoding.pop("preferred_chunks", None)
    return data


def main(args: argparse.Namespace) -> None:
    """Clones slice of Samudra data at the `dest_root` directory."""
    filter_keys = (args.prognostic_vars_key, args.boundary_vars_key)
    if any(filter_keys) and not all(filter_keys):
        raise ValueError(
            "--prognostic_vars_key and --boundary_vars_key must be provided together"
        )
    if all(filter_keys) and args.compact_variables:
        raise ValueError(
            "Variable-key filtering already writes normalized flattened variables "
            "and cannot be combined with --compact_variables"
        )
    dataset_spec = (
        build_om4_spec(
            prognostic_vars_key=args.prognostic_vars_key,
            boundary_vars_key=args.boundary_vars_key,
        )
        if all(filter_keys)
        else None
    )

    if args.local_cluster:
        cluster = LocalCluster()
        client = cluster.get_client()  # noqa: F841

    # Ensure the path/to/dest exists
    if not args.dest.startswith("gs://") and not args.dest.startswith("s3://"):
        pathlib.Path(args.dest).mkdir(parents=True, exist_ok=True)

    time_slice = slice(args.time_start, args.time_end)
    for name, dest_fmt in [
        ("OM4", "zarr"),
        ("OM4_means", "zarr"),
        ("OM4_stds", "zarr"),
    ]:
        dest = os.path.join(args.dest, name)
        source = (args.source or DEFAULT_DATA_ROOT) + name

        # Open Xarray Datasets with retries + exponential backoff.
        if name == "OM4":
            data = robust_open_dataset(source, engine="zarr", chunks={"time": 700})
            data = data.isel(time=time_slice)
        else:
            data = robust_open_dataset(source, engine="zarr", chunks={})

        if dataset_spec is not None:
            data = select_om4_variables(data, dataset_spec)

        if args.compact_variables:
            data = compact_dataset(data)

        with dask.diagnostics.ProgressBar():
            if dest_fmt.lower() == "zarr":
                if name == "OM4":
                    data = rechunk_for_output(data, args.write_time_chunks)
                data.to_zarr(dest + ".zarr")
            else:
                data.to_netcdf(dest + ".nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "clone_data",
        description="Make a copy of the OM4 dataset (~100 GiBs - ~2 TiBs).",
    )
    parser.add_argument(
        "dest", type=str, help="Root directory for the copy of datasets."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Alternative source root directory to copy data from. Defaults to 1° OM4 without Gaussian filtering.",
    )
    parser.add_argument(
        "--time_start",
        type=int,
        default=0,
        help="start index for data.isel() along time dimension.",
    )
    parser.add_argument(
        "--time_end",
        type=int,
        default=None,
        help="end index for data.isel() along time dimension.",
    )
    parser.add_argument(
        "--write_time_chunks",
        type=int,
        default=1,
        help="The number of time steps per output chunk. Default=1.",
    )
    parser.add_argument(
        "--prognostic_vars_key",
        type=str,
        default=None,
        help="Optional OM4 prognostic selector, for example thermo_dynamic_5.",
    )
    parser.add_argument(
        "--boundary_vars_key",
        type=str,
        default=None,
        help="Optional OM4 boundary selector, for example tau_hfds.",
    )
    parser.add_argument(
        "--compact_variables",
        action="store_true",
        help="Turn on a 'compact' data representation. This is now Zarr is more traditionally stored, but is sub-optimal in our data loader.",
    )
    parser.add_argument(
        "--local_cluster",
        action="store_true",
        help="Run pipeline on a local dask cluster. This should be faster due to multi-processing.",
    )

    main(parser.parse_args())
