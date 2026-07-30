# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build analysis-ready observation stores from the raw archive.

Ported from `YuanYuan98/Ocean_Emulator:viz_jupyter/prepare_obs_5day_aligned.py`
at commit `33d4ae2e608f57e357ba8529b747eb27be2eb041`, with one deliberate
change: the target timestamps come from the published **OM4 dataset** rather
than from a particular model rollout.

Yuan's script read the time axis out of one `predictions.zarr`, which welded the
resulting product to that run. OM4's 5-day cadence is the thing every Samudra
rollout inherits, so aligning to OM4 directly yields a product that serves any
rollout on that calendar, and makes the store reproducible from public inputs.

Two properties are preserved exactly, because the metrics depend on them:

- Each product stays on its **own native grid**. No spatial resampling happens
  here; the metrics interpolate the model onto the observation grid instead.
- Time alignment is a **centered 5-day rolling mean** sampled on the OM4
  timestamps, not an interpolation. The metrics refuse to interpolate in time,
  so the stored timestamps must match the model's exactly.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ocean_preprocessing.utils import get_git_url_hash

logger = logging.getLogger("ocean_preprocessing.obs")

DUACS_KEEP_VARS = ["ugos", "vgos", "ugosa", "vgosa", "adt", "sla"]
ARGO_TEMP_ALIASES = ["temp", "thetao", "temperature", "T"]
ARGO_SALT_ALIASES = ["so", "salinity", "salt", "sal", "psal", "S"]

# Provenance recorded on every store. These products are redistributed here for
# reproducible emulator evaluation, so attribution has to travel with the data
# rather than living only in a README.
PROVENANCE = {
    "duacs": {
        "source": "Copernicus Marine Service (CMEMS), DUACS L4 sea level",
        "source_product_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "source_url": "https://data.marine.copernicus.eu/",
        "attribution": "Generated using E.U. Copernicus Marine Service Information",
    },
    "oisst": {
        "source": "NOAA Optimum Interpolation SST (OISST) v2.1, AVHRR",
        "source_url": "https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
        "attribution": "NOAA National Centers for Environmental Information",
        "reference": (
            "Huang et al. (2021), Improvements of the Daily Optimum Interpolation "
            "Sea Surface Temperature (DOISST) Version 2.1, Journal of Climate"
        ),
    },
    "argo-iap": {
        "source": (
            "IAP gridded ocean temperature/salinity (CZ16), "
            "Institute of Atmospheric Physics, Chinese Academy of Sciences"
        ),
        "source_url": "http://www.ocean.iap.ac.cn/",
        "attribution": "Institute of Atmospheric Physics, Chinese Academy of Sciences",
        "reference": (
            "Cheng et al. (2017), Improved estimates of ocean heat content "
            "from 1960 to 2015, Science Advances"
        ),
    },
}


def _coerce_time(values: np.ndarray) -> pd.DatetimeIndex:
    """Convert a time coordinate to a DatetimeIndex, including cftime objects."""
    flat = np.asarray(values)
    if np.issubdtype(flat.dtype, np.datetime64):
        return pd.to_datetime(flat)
    if flat.size == 0:
        return pd.DatetimeIndex([])
    first = flat.flat[0]
    if hasattr(first, "year") and hasattr(first, "month"):
        return pd.to_datetime(
            [
                f"{v.year:04d}-{v.month:02d}-{v.day:02d} "
                f"{getattr(v, 'hour', 0):02d}:{getattr(v, 'minute', 0):02d}:"
                f"{getattr(v, 'second', 0):02d}"
                for v in flat
            ]
        )
    return pd.to_datetime(flat.astype(str))


def open_zarr(path: str, storage_options: dict | None = None) -> xr.Dataset:
    """Open a zarr store, reaching public S3 anonymously when no keys are set.

    The published observation inputs live in a public bucket, so the pipeline
    should be reproducible by someone with no credentials at all. When keys are
    present (as on the cluster) they are used instead, since the same bucket is
    readable either way.
    """
    if storage_options is None and str(path).startswith("s3://"):
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            storage_options = {"anon": True}
            logger.info("No AWS credentials found; reading %s anonymously", path)
    if storage_options:
        return xr.open_zarr(path, chunks={}, storage_options=storage_options)
    return xr.open_zarr(path, chunks={})


def om4_target_times(
    om4_path: str,
    start_date: str,
    end_date: str,
    storage_options: dict | None = None,
) -> pd.DatetimeIndex:
    """The OM4 5-day timestamps to align observations onto.

    Args:
        om4_path: Any OM4 store carrying the canonical 5-day time axis.
        start_date: First timestamp to keep, inclusive.
        end_date: Last timestamp to keep, inclusive.
        storage_options: fsspec options; inferred for public S3 when omitted.

    Returns:
        The OM4 timestamps inside the window.

    Raises:
        ValueError: If the window is empty, or the cadence is not 5-6 days.
            OM4's calendar is nominally 5-daily with occasional 6-day steps;
            anything else means the wrong store was passed, and silently
            aligning to it would produce an unusable product.
    """
    ds = open_zarr(om4_path, storage_options)
    times = _coerce_time(ds.time.values)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    # OM4 stamps sit at 12:00, so a date-only end bound would silently drop that
    # day's sample. Treat a midnight bound as "end of that day".
    if end == end.normalize():
        end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    window = times[(times >= start) & (times <= end)]
    if len(window) == 0:
        raise ValueError(
            f"No OM4 timestamps between {start_date} and {end_date} in {om4_path}"
        )
    deltas = set(np.unique((window[1:] - window[:-1]).days))
    if not deltas.issubset({5, 6}):
        raise ValueError(
            f"Expected a 5-day OM4 cadence, got day-deltas {sorted(deltas)} in {om4_path}"
        )
    logger.info(
        "OM4 target timeline: %d samples, %s to %s",
        len(window),
        window[0],
        window[-1],
    )
    return window


def _standardize_daily(ds: xr.Dataset) -> xr.Dataset:
    """Rename to lat/lon, coerce time, and put longitude on [0, 360)."""
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    ds = ds.rename(rename)
    ds = ds.assign_coords(time=_coerce_time(ds.time.values))
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds.sortby("lat")


def _align_centered_5day(ds: xr.Dataset, target_times: pd.DatetimeIndex) -> xr.Dataset:
    """Centered 5-day rolling mean, sampled on the target timestamps.

    `min_periods=5` means a window missing any of its five days yields NaN
    rather than a mean over fewer days -- a partial average is not the same
    quantity and would bias the comparison.
    """
    rolled = ds.rolling(time=5, center=True, min_periods=5).mean()
    rolled = rolled.assign_coords(
        time=("time", pd.DatetimeIndex(rolled.time.values).normalize())
    )
    aligned = rolled.sel(time=target_times.normalize())
    # Relabel to the exact OM4 stamps (which carry a 12:00 time of day), so the
    # metrics' exact-match check passes without any temporal interpolation.
    return aligned.assign_coords(time=("time", target_times.values))


def _chunk_for_write(ds: xr.Dataset) -> xr.Dataset:
    chunks = {}
    for dim, size in (("time", 24), ("lat", 180), ("lon", 360), ("depth_std", 16)):
        if dim in ds.dims:
            chunks[dim] = min(int(ds.sizes[dim]), size)
    return ds.chunk(chunks)


def _with_provenance(
    ds: xr.Dataset,
    product: str,
    *,
    alignment: str,
    om4_path: str | None = None,
) -> xr.Dataset:
    ds = ds.copy()
    ds.attrs.update(PROVENANCE[product])
    ds.attrs["m2lines/date_created"] = datetime.datetime.now().isoformat()
    ds.attrs["m2lines/cli_args"] = " ".join(sys.argv)
    ds.attrs["m2lines/ocean_emulators_git_hash"] = get_git_url_hash()
    ds.attrs["m2lines/time_alignment"] = alignment
    if om4_path is not None:
        ds.attrs["m2lines/time_axis_source"] = om4_path
    # The metrics compare on each product's own grid; recording this stops a
    # consumer from assuming the store was regridded to the model.
    ds.attrs["m2lines/horizontal_grid"] = "native product grid, not regridded"
    return ds


def _write(ds: xr.Dataset, output_path: str, dry_run: bool) -> None:
    if dry_run:
        logger.info("dry run, not writing %s:\n%s", output_path, ds)
        return
    logger.info("writing %s", output_path)
    _chunk_for_write(ds).to_zarr(output_path, mode="w", consolidated=True)
    logger.info("wrote %s", output_path)


ALIGNED = "centered 5-day rolling mean (min_periods=5) sampled on the OM4 5-day timestamps"
RAW_DAILY = "none; daily values as distributed"
MONTHLY = "none; monthly values as distributed"


def duacs(
    raw_dir: str,
    output_path: str,
    target_times: pd.DatetimeIndex,
    om4_path: str | None = None,
    dry_run: bool = False,
) -> None:
    """Build the 5-day aligned DUACS store from the per-year raw archive."""
    stores = sorted(Path(raw_dir).glob("duacs_*.zarr"))
    if not stores:
        raise FileNotFoundError(f"No raw DUACS year stores (duacs_*.zarr) under {raw_dir}")
    logger.info("DUACS: combining %d yearly stores", len(stores))

    ds = xr.concat(
        [xr.open_zarr(store, chunks={}) for store in stores],
        dim="time",
        coords="minimal",
        compat="override",
    ).sortby("time")

    keep = [name for name in DUACS_KEEP_VARS if name in ds.data_vars]
    missing = sorted(set(DUACS_KEEP_VARS) - set(keep))
    if missing:
        logger.info("  DUACS variables absent from the raw archive: %s", missing)
    ds = _standardize_daily(ds[keep])

    # Two days of margin on each side so the centered 5-day window is complete
    # at the first and last target timestamp.
    ds = ds.sel(
        time=slice(
            target_times.min().normalize() - pd.Timedelta(days=2),
            target_times.max().normalize() + pd.Timedelta(days=2),
        )
    )
    ds = _align_centered_5day(ds, target_times)
    _write(_with_provenance(ds, "duacs", alignment=ALIGNED, om4_path=om4_path), output_path, dry_run)


def _open_oisst(raw_dir: str, dates: pd.DatetimeIndex) -> xr.Dataset:
    files = [Path(raw_dir) / f"oisst-avhrr-v02r01.{d:%Y%m%d}.nc" for d in dates]
    absent = [path.name for path in files if not path.exists()]
    if absent:
        raise FileNotFoundError(
            f"Missing {len(absent)} OISST daily files, e.g. {absent[:5]}. "
            "Run the download stage first, or narrow the requested window."
        )
    ds = xr.open_mfdataset([str(path) for path in files], combine="by_coords")[["sst"]]
    return _standardize_daily(ds).squeeze(drop=True)


def oisst(
    raw_dir: str,
    output_path: str,
    target_times: pd.DatetimeIndex,
    om4_path: str | None = None,
    dry_run: bool = False,
) -> None:
    """Build the 5-day aligned OISST store."""
    dates = pd.date_range(
        target_times.min().normalize() - pd.Timedelta(days=2),
        target_times.max().normalize() + pd.Timedelta(days=2),
        freq="D",
    )
    ds = _align_centered_5day(_open_oisst(raw_dir, dates), target_times)
    _write(_with_provenance(ds, "oisst", alignment=ALIGNED, om4_path=om4_path), output_path, dry_run)


def oisst_daily(
    raw_dir: str,
    output_path: str,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> None:
    """Build the unsmoothed daily OISST store, used for temporal spectra."""
    dates = pd.date_range(pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize(), freq="D")
    ds = _open_oisst(raw_dir, dates)
    _write(_with_provenance(ds, "oisst", alignment=RAW_DAILY), output_path, dry_run)


def _argo_file_time(path: Path) -> pd.Timestamp | None:
    stem = path.stem
    if "_year_" not in stem or "_month_" not in stem:
        return None
    try:
        year = int(stem.split("_year_")[1].split("_")[0])
        month = int(stem.split("_month_")[1].split("_")[0])
    except (IndexError, ValueError):
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def _open_argo_field(raw_dir: str, field: str) -> xr.Dataset | None:
    paths = sorted(
        (path for path in (Path(raw_dir) / field).glob("*.nc")),
        key=lambda p: _argo_file_time(p) or pd.Timestamp.min,
    )
    if not paths:
        return None
    times = [_argo_file_time(path) for path in paths]
    if any(time is None for time in times):
        raise ValueError(f"Could not parse year/month from some ARGO {field} filenames")
    ds = xr.open_mfdataset(
        [str(path) for path in paths],
        combine="nested",
        concat_dim="time",
        decode_times=False,
    )
    return ds.assign_coords(time=("time", pd.DatetimeIndex(times)))


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def argo_iap(
    raw_dir: str,
    output_path: str,
    dry_run: bool = False,
) -> None:
    """Build the monthly ARGO-IAP store with temperature and, when present, salinity."""
    ds_temp = _open_argo_field(raw_dir, "temperature")
    if ds_temp is None:
        raise FileNotFoundError(f"No ARGO-IAP temperature files under {raw_dir}/temperature")
    temp_name = _find_var(ds_temp, ARGO_TEMP_ALIASES)
    if temp_name is None:
        raise KeyError(
            f"No recognised temperature variable in {list(ds_temp.data_vars)}; "
            f"expected one of {ARGO_TEMP_ALIASES}"
        )
    data_vars = {"temp": ds_temp[temp_name]}

    ds_salt = _open_argo_field(raw_dir, "salinity")
    if ds_salt is None:
        logger.info("  ARGO-IAP salinity absent; writing temperature only")
    else:
        salt_name = _find_var(ds_salt, ARGO_SALT_ALIASES)
        if salt_name is None:
            raise KeyError(f"No recognised salinity variable in {list(ds_salt.data_vars)}")
        if not pd.DatetimeIndex(ds_temp.time.values).equals(
            pd.DatetimeIndex(ds_salt.time.values)
        ):
            raise ValueError(
                "ARGO-IAP temperature and salinity cover different months; "
                "re-run the download so both fields span the same period"
            )
        data_vars["salt"] = ds_salt[salt_name]

    ds = xr.Dataset(data_vars).sortby("time")
    ds = _standardize_daily(ds)
    for name in list(ds.data_vars):
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype("float32")
    _write(_with_provenance(ds, "argo-iap", alignment=MONTHLY), output_path, dry_run)
