# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Provenance shared by every campaign and mutation-gate artifact."""

import importlib.metadata
import subprocess
from pathlib import Path


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_environment(path: Path) -> None:
    """Record every installed distribution, one `name==version` per line."""
    path.write_text(
        "\n".join(
            sorted(
                f"{package.metadata['Name']}=={package.version}"
                for package in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
