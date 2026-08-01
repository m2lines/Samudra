# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Download the raw observation products, as distributed by their providers.

Every downloader is restartable: files already present at a plausible size are
never requested, and anything that arrives truncated is deleted rather than
left to look complete. That matters because these are long jobs over thousands
of files, and one that has to start from zero after a network blip will never
finish.

Transfers go through `pypdl`, which fetches several files at once and splits
each across connections. Segmenting matters most for the IAP archive, whose
~40 MB monthly files arrive at a few MB/s over a single connection.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

import xarray as xr
from pypdl import Pypdl

logger = logging.getLogger("ocean_preprocessing.obs")

OISST_BASE_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr"
)
OISST_FILENAME = "oisst-avhrr-v02r01.{date:%Y%m%d}.nc"

DUACS_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
DUACS_VELOCITY_VARIABLES = ("ugos", "vgos", "ugosa", "vgosa")
DUACS_SSH_VARIABLES = ("adt", "sla")


@dataclasses.dataclass(frozen=True)
class ArgoField:
    """One ARGO-IAP field, with the mirrors and spellings it appears under."""

    name: str
    base_urls: tuple[str, ...]
    filename_templates: tuple[str, ...]


# IAP serves the same files from two hosts under inconsistent capitalisation, so
# each candidate is tried in turn before a month is considered missing.
ARGO_FIELDS = {
    "temperature": ArgoField(
        name="temperature",
        base_urls=(
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_Temperature_0p5_gridded_1month_netcdf",
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_temperature_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_Temperature_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_temperature_0p5_gridded_1month_netcdf",
        ),
        filename_templates=(
            "IAP_05_2000m_temperature_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_Temperature_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_temp_year_{year}_month_{month:02d}.nc",
        ),
    ),
    "salinity": ArgoField(
        name="salinity",
        base_urls=(
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_Salinity_0p5_gridded_1month_netcdf",
            "http://www.ocean.iap.ac.cn/ftp/cheng/CZ16_v0_IAP_salinity_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_Salinity_0p5_gridded_1month_netcdf",
            "http://159.226.119.60/cheng/CZ16_v0_IAP_salinity_0p5_gridded_1month_netcdf",
        ),
        filename_templates=(
            "IAP_05_2000m_salinity_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_Salinity_year_{year}_month_{month:02d}.nc",
            "IAP_05_2000m_salt_year_{year}_month_{month:02d}.nc",
        ),
    ),
}

# A truncated NetCDF is worse than a missing one: it opens, and then poisons the
# derived product silently. Anything below this is treated as a failed download.
MIN_PLAUSIBLE_BYTES = 1_000_000


def _is_complete(path: Path, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def _download_batch(
    tasks: list[tuple[str, Path]],
    *,
    workers: int,
    segments: int,
    retries: int,
    min_bytes: int,
    label: str,
) -> set[str]:
    """Download `(url, target)` pairs concurrently. Returns the URLs that failed.

    Uses pypdl, which downloads several files at once *and* splits each file
    across connections. The second part is what matters for the slower
    providers: the IAP archive delivers ~40 MB monthly files at a few MB/s on a
    single connection, so segmenting a file is a bigger win there than adding
    more concurrent files.

    Success is not taken on trust. A truncated file that pypdl reports as
    complete would be skipped forever by the size check on the next run, so
    every target is re-checked afterwards and anything implausibly small is
    deleted and reported as failed.
    """
    if not tasks:
        return set()
    for _, target in tasks:
        target.parent.mkdir(parents=True, exist_ok=True)

    downloader = Pypdl(max_concurrent=workers)
    downloader.start(
        tasks=[{"url": url, "file_path": str(target)} for url, target in tasks],
        segments=segments,
        retries=retries,
        display=False,
        block=True,
        overwrite=True,
    )

    failed = set()
    for entry in downloader.failed or []:
        # pypdl reports failures as the task dicts it was given.
        failed.add(entry["url"] if isinstance(entry, dict) else str(entry))

    for url, target in tasks:
        if url in failed:
            continue
        if not _is_complete(target, min_bytes):
            logger.debug("  %s: implausible result for %s", label, target.name)
            target.unlink(missing_ok=True)
            failed.add(url)

    logger.info("  %s: %d/%d fetched", label, len(tasks) - len(failed), len(tasks))
    return failed


def oisst(
    output_dir: Path | str,
    start_year: int = 1982,
    end_year: int = 2022,
    workers: int = 8,
    segments: int = 4,
    retries: int = 4,
    overwrite: bool = False,
) -> None:
    """Download NOAA OISST v2.1 daily NetCDF files.

    Args:
        output_dir: Directory to fill with `oisst-avhrr-v02r01.YYYYMMDD.nc`.
        start_year: First calendar year to fetch.
        end_year: Last calendar year to fetch, inclusive.
        workers: Files fetched concurrently.
        segments: Connections per file.
        retries: Attempts per file before giving up.
        overwrite: Re-download files that already look complete.
    """
    output_dir = Path(output_dir)
    days = xr.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    logger.info(
        "OISST: %d daily files, %d-%d -> %s",
        len(days),
        start_year,
        end_year,
        output_dir,
    )

    # Filter before handing work to the downloader rather than relying on its
    # own overwrite handling, so restart behaviour stays explicit: a file that
    # is already present at a plausible size is simply never requested.
    pending = []
    for day in days:
        filename = OISST_FILENAME.format(date=day)
        target = output_dir / filename
        if _is_complete(target, MIN_PLAUSIBLE_BYTES) and not overwrite:
            continue
        pending.append((f"{OISST_BASE_URL}/{day:%Y%m}/{filename}", target))

    logger.info(
        "OISST: %d already present, %d to fetch", len(days) - len(pending), len(pending)
    )
    failed = _download_batch(
        pending,
        workers=workers,
        segments=segments,
        retries=retries,
        min_bytes=MIN_PLAUSIBLE_BYTES,
        label="OISST",
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} OISST files could not be downloaded, e.g. "
            f"{sorted(failed)[:3]}. Re-run to retry only the missing days."
        )


def argo_iap(
    output_dir: Path | str,
    start_year: int = 1980,
    end_year: int = 2022,
    fields: tuple[str, ...] = ("temperature", "salinity"),
    workers: int = 4,
    segments: int = 8,
    retries: int = 3,
    overwrite: bool = False,
) -> None:
    """Download ARGO-IAP monthly gridded temperature and salinity.

    The IAP archive serves the same files from two hosts under inconsistent
    capitalisation, so each month is attempted against one naming candidate at
    a time: everything still missing after a round is retried with the next
    spelling. Batching by candidate rather than looping per file keeps the
    downloads concurrent while preserving the fallback.

    Args:
        output_dir: Directory to fill with the monthly NetCDF files.
        start_year: First calendar year to fetch.
        end_year: Last calendar year to fetch, inclusive.
        fields: Which of `temperature`, `salinity` to fetch.
        workers: Files fetched concurrently. Kept low; the IAP hosts are modest.
        segments: Connections per file. Worth more here than extra workers --
            these are ~40 MB files from a slow origin.
        retries: Attempts per file within a candidate round.
        overwrite: Re-download files that already look complete.
    """
    output_dir = Path(output_dir)
    unknown = set(fields) - set(ARGO_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown ARGO fields {sorted(unknown)}; expected {sorted(ARGO_FIELDS)}"
        )

    outstanding: dict[Path, tuple[ArgoField, int, int]] = {}
    total = 0
    for field_name in fields:
        config = ARGO_FIELDS[field_name]
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                total += 1
                canonical = config.filename_templates[0].format(year=year, month=month)
                target = output_dir / field_name / canonical
                if _is_complete(target, MIN_PLAUSIBLE_BYTES) and not overwrite:
                    continue
                outstanding[target] = (config, year, month)

    logger.info(
        "ARGO-IAP: %d monthly files, %d already present, %d to fetch -> %s",
        total,
        total - len(outstanding),
        len(outstanding),
        output_dir,
    )

    candidates = [
        (base, template)
        for config in (ARGO_FIELDS[f] for f in fields)
        for base in config.base_urls
        for template in config.filename_templates
    ]
    seen: set[tuple[str, str]] = set()
    rounds = [c for c in candidates if not (c in seen or seen.add(c))]

    for base_url, template in rounds:
        if not outstanding:
            break
        tasks = []
        for target, (config, year, month) in outstanding.items():
            if (
                base_url not in config.base_urls
                or template not in config.filename_templates
            ):
                continue
            tasks.append(
                (f"{base_url}/{template.format(year=year, month=month)}", target)
            )
        if not tasks:
            continue
        failed = _download_batch(
            tasks,
            workers=workers,
            segments=segments,
            retries=retries,
            min_bytes=MIN_PLAUSIBLE_BYTES,
            label=f"ARGO-IAP [{template.split('_year_')[0]}]",
        )
        for url, target in tasks:
            if url not in failed:
                outstanding.pop(target, None)

    if outstanding:
        names = sorted(p.name for p in outstanding)[:3]
        raise RuntimeError(
            f"{len(outstanding)} ARGO-IAP files could not be downloaded from any "
            f"mirror, e.g. {names}. Re-run to retry only the missing months."
        )


def _yearly_ranges(start: str, end: str) -> list[tuple[date, date]]:
    """Split `[start, end]` into per-calendar-year spans, clamped at both ends."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError(f"Invalid date range: {start} > {end}")
    # The interior boundaries are the January 1sts inside the span; the first
    # and last spans are clamped to the dates actually requested, so a range
    # starting mid-October yields a short first year.
    edges = [first] + [
        stamp.date()
        for stamp in xr.date_range(start, end, freq="YS")
        if stamp.date() > first
    ]
    return [
        (edge, edges[i + 1] - timedelta(days=1) if i + 1 < len(edges) else last)
        for i, edge in enumerate(edges)
    ]


def duacs(
    output_dir: Path | str,
    start_date: str = "2014-10-18",
    end_date: str = "2022-12-26",
    variables: tuple[str, ...] = DUACS_VELOCITY_VARIABLES,
    include_ssh: bool = False,
    credentials_file: Path | str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Download daily DUACS from Copernicus Marine, one zarr per calendar year.

    Split by year deliberately: the full record is a few hundred GB, and a
    single store means any interruption loses everything. Per-year stores make
    the job restartable and let the prepare stage open them as one collection.

    Requires a free Copernicus Marine account. Set
    `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`,
    or pass `credentials_file`, or run `copernicusmarine login` once.

    Args:
        output_dir: Directory to hold the per-year zarr stores.
        start_date: First day to fetch, `YYYY-MM-DD`.
        end_date: Last day to fetch, inclusive.
        variables: DUACS variables to subset.
        include_ssh: Also fetch `adt` and `sla`.
        credentials_file: Toolbox-compatible credentials file, if not using env vars.
        overwrite: Re-download years that already exist.
        dry_run: Ask Copernicus Marine to validate the request without downloading.
    """
    output_dir = Path(output_dir)
    if not _has_copernicus_credentials(credentials_file):
        raise SystemExit(
            "Copernicus Marine credentials are required for DUACS. Set "
            "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD "
            "in the environment, pass credentials_file, or run `copernicusmarine login`. "
            "A free account is available at https://data.marine.copernicus.eu/register"
        )

    wanted = tuple(variables) + (DUACS_SSH_VARIABLES if include_ssh else ())
    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = _yearly_ranges(start_date, end_date)
    logger.info(
        "DUACS: %d yearly stores, %s to %s, variables=%s -> %s",
        len(ranges),
        start_date,
        end_date,
        ",".join(wanted),
        output_dir,
    )

    for first, last in ranges:
        final = output_dir / f"duacs_{first:%Y%m%d}_{last:%Y%m%d}.zarr"
        # Download to a temporary name and rename on success, so an interrupted
        # year is never mistaken for a complete one on the next run.
        tmp = final.with_suffix(".tmp.zarr")
        # Clear any leftover partial *before* the skip check, not after. A year
        # interrupted on one run and completed on the next would otherwise keep
        # its stale tmp store forever, and the prepare stage would concatenate
        # that partial data alongside the good year.
        if not dry_run and tmp.exists():
            logger.info("  discarding stale partial download: %s", tmp.name)
            shutil.rmtree(tmp)

        if final.exists() and not overwrite and not dry_run:
            logger.info("  exists, skipping: %s", final.name)
            continue

        if not dry_run and overwrite and final.exists():
            shutil.rmtree(final)

        cmd = [
            "copernicusmarine",
            "subset",
            "-i",
            DUACS_DATASET_ID,
            "--start-datetime",
            first.isoformat(),
            "--end-datetime",
            last.isoformat(),
            "--file-format",
            "zarr",
            "--output-directory",
            str(output_dir),
            "--output-filename",
            final.name if dry_run else tmp.name,
            "--disable-progress-bar",
        ]
        for variable in wanted:
            cmd.extend(["-v", variable])
        if credentials_file is not None:
            cmd.extend(["--credentials-file", str(credentials_file)])
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.append("--overwrite")

        logger.info("  %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        if not dry_run:
            if not tmp.exists():
                raise FileNotFoundError(f"Copernicus Marine wrote no output at {tmp}")
            tmp.rename(final)
            logger.info("  wrote %s", final.name)

    logger.info("DUACS: done")


def _has_copernicus_credentials(credentials_file: Path | str | None) -> bool:
    if credentials_file is not None and Path(credentials_file).exists():
        return True
    if os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and os.environ.get(
        "COPERNICUSMARINE_SERVICE_PASSWORD"
    ):
        return True
    home = Path.home()
    return any(
        path.exists()
        for path in [
            home / ".copernicusmarine" / ".copernicusmarine-credentials",
            home / ".netrc",
            home / "_netrc",
        ]
    )
