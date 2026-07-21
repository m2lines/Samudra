# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Download the filtered OM4 data required by the Colab quickstart."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DOWNLOAD_FINGERPRINT = "om4-t0-250-thermo_dynamic_5-tau_hfds-timechunk10-v2"
EXPECTED_STORES = ("OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def download_is_current(destination: Path) -> bool:
    """Return whether a complete quickstart download can be reused."""
    marker = destination / ".quickstart_download_complete"
    return (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == DOWNLOAD_FINGERPRINT
        and all((destination / store).is_dir() for store in EXPECTED_STORES)
    )


def clear_incomplete_download(destination: Path) -> None:
    """Remove only quickstart-owned stores before retrying a download."""
    (destination / ".quickstart_download_complete").unlink(missing_ok=True)
    for store in EXPECTED_STORES:
        shutil.rmtree(destination / store, ignore_errors=True)


def download(destination: Path) -> None:
    """Download the minimum data slice used by the quickstart configuration."""
    destination.mkdir(parents=True, exist_ok=True)
    if download_is_current(destination):
        print(f"Using the completed download at {destination}")
        return

    clear_incomplete_download(destination)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "clone_data.py"),
        str(destination),
        "--time_start",
        "0",
        "--time_end",
        "250",
        "--write_time_chunks",
        "10",
        "--prognostic_vars_key",
        "thermo_dynamic_5",
        "--boundary_vars_key",
        "tau_hfds",
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(REPO_ROOT / "src"), existing_pythonpath)
        if path is not None
    )
    subprocess.run(command, check=True, env=environment)
    marker = destination / ".quickstart_download_complete"
    marker.write_text(DOWNLOAD_FINGERPRINT + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    download(parser.parse_args().destination)
