# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the exporter against the published, frozen training implementation."""

import os
import subprocess
import sys
import tarfile
from pathlib import Path


def test_signed_stream_against_frozen_direct_forcing_model(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    archive = (
        repo
        / "docs/experiments/stochastic-transitions/sources/signed-dense/code.tar.gz"
    )
    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(archive) as saved:
        saved.extractall(source, filter="data")
    source = source / "code"
    target = tmp_path / "test_signed_annual_checks.py"
    target.write_text((repo / "tests/fixtures/signed_annual_checks.py").read_text())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source / "src")
    environment["SIGNED_EXPORTER"] = str(
        repo / "scripts/export_stochastic_signed_annual_viewer.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--confcutdir",
            str(tmp_path),
            str(target),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
