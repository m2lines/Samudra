# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("samudra")
except PackageNotFoundError:  # not installed (e.g. running from a bare source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
