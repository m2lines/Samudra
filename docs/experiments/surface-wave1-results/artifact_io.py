# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read original metric bytes from plain files or lossless gzip copies."""

import gzip
import io
from pathlib import Path


def artifact_exists(path: Path) -> bool:
    return path.exists() or Path(str(path) + ".gz").exists()


def read_artifact(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(Path(str(path) + ".gz").read_bytes())


def open_artifact(path: Path) -> io.StringIO:
    return io.StringIO(read_artifact(path).decode())
