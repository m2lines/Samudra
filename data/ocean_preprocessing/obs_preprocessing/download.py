# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Download the raw observation products, as distributed by their providers.

Every downloader is restartable: a file that already exists at a plausible size
is skipped, and partial downloads land on a `.part` path that is only renamed
into place once complete. That matters because these are long jobs over
thousands of files, and a job that has to start from zero after a network blip
will never finish.
"""

from __future__ import annotations

import calendar
import dataclasses
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

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


def _fetch(
    url: str,
    target: Path,
    *,
    timeout: float,
    retries: int,
    retry_wait: float,
    min_bytes: int,
) -> bool:
    """Fetch one URL to `target` atomically. Returns True on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if getattr(response, "status", 200) >= 400:
                    return False
                with tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
            size = tmp.stat().st_size
            if size < min_bytes:
                raise OSError(f"downloaded file is implausibly small: {size} bytes")
            tmp.replace(target)
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            logger.debug(
                "  attempt %d/%d failed for %s: %s", attempt, retries, url, exc
            )
            if attempt < retries:
                time.sleep(retry_wait)
    return False


def _run_pool(tasks, worker, workers: int, label: str) -> tuple[int, list[str]]:
    """Run `worker` over `tasks` concurrently, returning (successes, failures)."""
    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            ok, name = future.result()
            if ok:
                done += 1
            else:
                failed.append(name)
            total = done + len(failed)
            if total % 250 == 0:
                logger.info("  %s: %d/%d", label, total, len(tasks))
    return done, failed


def oisst(
    output_dir: Path | str,
    start_year: int = 1982,
    end_year: int = 2022,
    workers: int = 8,
    timeout: float = 120.0,
    retries: int = 4,
    retry_wait: float = 5.0,
    overwrite: bool = False,
) -> None:
    """Download NOAA OISST v2.1 daily NetCDF files.

    Args:
        output_dir: Directory to fill with `oisst-avhrr-v02r01.YYYYMMDD.nc`.
        start_year: First calendar year to fetch.
        end_year: Last calendar year to fetch, inclusive.
        workers: Concurrent downloads.
        timeout: Per-request timeout in seconds.
        retries: Attempts per file before giving up.
        retry_wait: Seconds between attempts.
        overwrite: Re-download files that already look complete.
    """
    output_dir = Path(output_dir)
    days = [
        date(year, month, day)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]
    logger.info(
        "OISST: %d daily files, %d-%d -> %s",
        len(days),
        start_year,
        end_year,
        output_dir,
    )

    def worker(day: date) -> tuple[bool, str]:
        filename = OISST_FILENAME.format(date=day)
        target = output_dir / filename
        if _is_complete(target, MIN_PLAUSIBLE_BYTES) and not overwrite:
            return True, filename
        url = f"{OISST_BASE_URL}/{day:%Y%m}/{filename}"
        ok = _fetch(
            url,
            target,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
            min_bytes=MIN_PLAUSIBLE_BYTES,
        )
        return ok, filename

    done, failed = _run_pool(days, worker, workers, "OISST")
    logger.info("OISST: %d present, %d failed", done, len(failed))
    if failed:
        raise RuntimeError(
            f"{len(failed)} OISST files could not be downloaded, e.g. {failed[:5]}. "
            "Re-run to retry only the missing days."
        )


def argo_iap(
    output_dir: Path | str,
    start_year: int = 1980,
    end_year: int = 2022,
    fields: tuple[str, ...] = ("temperature", "salinity"),
    workers: int = 4,
    timeout: float = 300.0,
    retries: int = 3,
    retry_wait: float = 5.0,
    overwrite: bool = False,
) -> None:
    """Download ARGO-IAP monthly gridded temperature and salinity.

    Args:
        output_dir: Directory to fill with the monthly NetCDF files.
        start_year: First calendar year to fetch.
        end_year: Last calendar year to fetch, inclusive.
        fields: Which of `temperature`, `salinity` to fetch.
        workers: Concurrent downloads. Kept low; the IAP hosts are modest.
        timeout: Per-request timeout in seconds.
        retries: Attempts per mirror/spelling before moving on.
        retry_wait: Seconds between attempts.
        overwrite: Re-download files that already look complete.
    """
    output_dir = Path(output_dir)
    unknown = set(fields) - set(ARGO_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown ARGO fields {sorted(unknown)}; expected {sorted(ARGO_FIELDS)}"
        )

    months = [
        (field, year, month)
        for field in fields
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]
    logger.info("ARGO-IAP: %d monthly files -> %s", len(months), output_dir)

    def worker(task: tuple[str, int, int]) -> tuple[bool, str]:
        field_name, year, month = task
        config = ARGO_FIELDS[field_name]
        canonical = config.filename_templates[0].format(year=year, month=month)
        target = output_dir / field_name / canonical
        if _is_complete(target, MIN_PLAUSIBLE_BYTES) and not overwrite:
            return True, canonical
        # Try every mirror/spelling combination; providers are inconsistent and a
        # miss on one spelling does not mean the month is unavailable.
        for base_url in config.base_urls:
            for template in config.filename_templates:
                url = f"{base_url}/{template.format(year=year, month=month)}"
                if _fetch(
                    url,
                    target,
                    timeout=timeout,
                    retries=retries,
                    retry_wait=retry_wait,
                    min_bytes=MIN_PLAUSIBLE_BYTES,
                ):
                    return True, canonical
        return False, canonical

    done, failed = _run_pool(months, worker, workers, "ARGO-IAP")
    logger.info("ARGO-IAP: %d present, %d failed", done, len(failed))
    if failed:
        raise RuntimeError(
            f"{len(failed)} ARGO-IAP files could not be downloaded, e.g. {failed[:5]}. "
            "Re-run to retry only the missing months."
        )


def _yearly_ranges(start: str, end: str) -> list[tuple[date, date]]:
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    if start_date > end_date:
        raise ValueError(f"Invalid date range: {start} > {end}")
    ranges, current = [], start_date
    while current <= end_date:
        year_end = min(date(current.year, 12, 31), end_date)
        ranges.append((current, year_end))
        current = year_end + timedelta(days=1)
    return ranges


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
        if final.exists() and not overwrite and not dry_run:
            logger.info("  exists, skipping: %s", final.name)
            continue

        # Download to a temporary name and rename on success, so an interrupted
        # year is never mistaken for a complete one on the next run.
        tmp = final.with_suffix(".tmp.zarr")
        if not dry_run and tmp.exists():
            shutil.rmtree(tmp)
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
