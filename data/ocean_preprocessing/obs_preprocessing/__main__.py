# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""CLI for fetching and preparing observation products.

    python -m ocean_preprocessing.obs_preprocessing download oisst --output_dir=...
    python -m ocean_preprocessing.obs_preprocessing prepare all --raw_root=... --output_root=...
"""

import logging
import os

# Match the sibling OM4 CLI: single-threaded blosc avoids decompression errors
# when several workers read the same compressed store.
os.environ.setdefault("BLOSC_NTHREADS", "1")
os.environ.setdefault("BLOSC_NOLOCK", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import fire  # noqa: E402

from ocean_preprocessing.obs_preprocessing import download as download_mod  # noqa: E402
from ocean_preprocessing.obs_preprocessing import prepare as prepare_mod  # noqa: E402

logger = logging.getLogger("ocean_preprocessing.obs")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# The OM4 store whose 5-day calendar the observations are aligned onto. Public
# and anonymous, so preparing the observation products needs no credentials.
DEFAULT_OM4_PATH = "s3://m2lines-pubs/Samudra/v2026-07/om4_onedeg/OM4.zarr"
# Covers the full OM4/observation overlap. DUACS begins 2014-10-18; two days of
# margin on each side keep the centered 5-day windows complete.
DEFAULT_START = "2014-10-20"
DEFAULT_END = "2022-12-24"


class Download:
    """Fill the raw observation archive. Restartable: complete files are skipped."""

    def oisst(
        self,
        output_dir: str,
        start_year: int = 1982,
        end_year: int = 2022,
        workers: int = 8,
        overwrite: bool = False,
    ) -> None:
        """Download NOAA OISST v2.1 daily NetCDF. No credentials needed."""
        download_mod.oisst(
            output_dir,
            start_year=start_year,
            end_year=end_year,
            workers=workers,
            overwrite=overwrite,
        )

    def argo_iap(
        self,
        output_dir: str,
        start_year: int = 1980,
        end_year: int = 2022,
        workers: int = 4,
        overwrite: bool = False,
    ) -> None:
        """Download ARGO-IAP monthly temperature and salinity. No credentials needed."""
        download_mod.argo_iap(
            output_dir,
            start_year=start_year,
            end_year=end_year,
            workers=workers,
            overwrite=overwrite,
        )

    def duacs(
        self,
        output_dir: str,
        start_date: str = "2014-10-18",
        end_date: str = "2022-12-26",
        include_ssh: bool = False,
        credentials_file: str | None = None,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Download daily DUACS as per-year zarrs. Needs Copernicus Marine credentials."""
        download_mod.duacs(
            output_dir,
            start_date=start_date,
            end_date=end_date,
            include_ssh=include_ssh,
            credentials_file=credentials_file,
            overwrite=overwrite,
            dry_run=dry_run,
        )


class Prepare:
    """Derive the analysis-ready stores from the raw archive."""

    def _times(self, om4_path: str, start_date: str, end_date: str):
        return prepare_mod.om4_target_times(om4_path, start_date, end_date)

    def duacs(
        self,
        raw_root: str,
        output_root: str,
        om4_path: str = DEFAULT_OM4_PATH,
        start_date: str = DEFAULT_START,
        end_date: str = DEFAULT_END,
        dry_run: bool = False,
    ) -> None:
        """Build the 5-day aligned DUACS store."""
        times = self._times(om4_path, start_date, end_date)
        prepare_mod.duacs(
            str(Path(raw_root) / "duacs"),
            str(Path(output_root) / "duacs.zarr"),
            times,
            om4_path=om4_path,
            dry_run=dry_run,
        )

    def oisst(
        self,
        raw_root: str,
        output_root: str,
        om4_path: str = DEFAULT_OM4_PATH,
        start_date: str = DEFAULT_START,
        end_date: str = DEFAULT_END,
        include_daily: bool = True,
        dry_run: bool = False,
    ) -> None:
        """Build the 5-day aligned OISST store, and optionally the raw daily one."""
        times = self._times(om4_path, start_date, end_date)
        raw_dir = str(Path(raw_root) / "oisst")
        prepare_mod.oisst(
            raw_dir,
            str(Path(output_root) / "oisst.zarr"),
            times,
            om4_path=om4_path,
            dry_run=dry_run,
        )
        if include_daily:
            prepare_mod.oisst_daily(
                raw_dir,
                str(Path(output_root) / "oisst_daily.zarr"),
                start_date=start_date,
                end_date=end_date,
                dry_run=dry_run,
            )

    def argo_iap(self, raw_root: str, output_root: str, dry_run: bool = False) -> None:
        """Build the monthly ARGO-IAP store."""
        prepare_mod.argo_iap(
            str(Path(raw_root) / "argo-iap"),
            str(Path(output_root) / "argo-iap.zarr"),
            dry_run=dry_run,
        )

    def all(
        self,
        raw_root: str,
        output_root: str,
        om4_path: str = DEFAULT_OM4_PATH,
        start_date: str = DEFAULT_START,
        end_date: str = DEFAULT_END,
        dry_run: bool = False,
    ) -> None:
        """Build every store the observation metrics need."""
        Path(output_root).mkdir(parents=True, exist_ok=True)
        self.oisst(raw_root, output_root, om4_path, start_date, end_date, dry_run=dry_run)
        self.argo_iap(raw_root, output_root, dry_run=dry_run)
        self.duacs(raw_root, output_root, om4_path, start_date, end_date, dry_run=dry_run)


class CLI:
    """Fetch and prepare observation products for emulator evaluation.

    Two stages, run in order:

    ```bash
    # 1. Fill the raw archive (OISST and ARGO need no credentials)
    python -m ocean_preprocessing.obs_preprocessing download oisst --output_dir=$RAW/oisst
    python -m ocean_preprocessing.obs_preprocessing download argo_iap --output_dir=$RAW/argo-iap
    export COPERNICUSMARINE_SERVICE_USERNAME=... COPERNICUSMARINE_SERVICE_PASSWORD=...
    python -m ocean_preprocessing.obs_preprocessing download duacs --output_dir=$RAW/duacs

    # 2. Derive the analysis-ready stores
    python -m ocean_preprocessing.obs_preprocessing prepare all \\
        --raw_root=$RAW --output_root=$OUT
    ```

    The prepare stage reads the OM4 time axis to decide which timestamps to
    align onto, defaulting to the public store, so it needs no credentials
    either. Pass `--dry_run` to print the resulting datasets without writing.
    """

    def __init__(self):
        self.download = Download()
        self.prepare = Prepare()


def main() -> None:
    fire.Fire(CLI, name="ocean_preprocessing.obs_preprocessing")


if __name__ == "__main__":
    main()
