# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-only pipeline tests and a noninteractive Copernicus credential check."""

import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ocean_preprocessing.obs_preprocessing.download import _has_copernicus_credentials


def main() -> None:
    print("Python:", sys.executable, flush=True)
    for name in ["copernicusmarine", "pypdl", "xarray", "numpy", "zarr"]:
        print(name, importlib.metadata.version(name), flush=True)
    repo = Path(__file__).parents[3]
    with tempfile.TemporaryDirectory(prefix="obs-check-") as temporary:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(repo / "data/tests/test_obs_full_range.py"),
                "-q",
                "--import-mode=importlib",
                "-p",
                "no:cacheprovider",
                f"--basetemp={temporary}/pytest",
            ],
            env=os.environ | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            check=True,
        )
    credentials = os.environ.get("COPERNICUS_CREDENTIALS_FILE")
    if not _has_copernicus_credentials(credentials):
        raise SystemExit("Copernicus credentials are missing from the job environment")
    command = ["copernicusmarine", "login", "--check-credentials-valid"]
    if credentials:
        command += ["--credentials-file", credentials]
    result = subprocess.run(
        command, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120
    )
    if result.returncode:
        # Do not echo authentication output or secrets into the Slurm log.
        raise SystemExit(
            "Copernicus credential validation failed; check the Torch login environment"
        )
    print("Copernicus credentials: valid", flush=True)


if __name__ == "__main__":
    main()
