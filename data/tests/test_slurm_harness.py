# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

VARIANT_SCRIPT = Path(__file__).parents[2] / "scripts" / "om4_data_variant.sh"


def _resolve_variant(variant: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"DATA_VARIANT": variant}
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; resolve_om4_data_variant || exit $?; '
            'printf "%s|%s|%s" "$DATA_VARIANT" "$OM4_SOURCE_STORE" '
            '"$OM4_OUTPUT_SUFFIX"',
            "bash",
            str(VARIANT_SCRIPT),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("averaged", "averaged|om4_5daily.zarr|"),
        ("snapshots", "snapshots|om4_5daily_snapshots.zarr|_snapshots"),
    ],
)
def test_om4_data_variant(variant, expected):
    result = _resolve_variant(variant)

    assert result.returncode == 0
    assert result.stdout == expected


def test_om4_data_variant_rejects_unknown_value():
    result = _resolve_variant("instantaneous")

    assert result.returncode == 2
    assert "unknown DATA_VARIANT='instantaneous'" in result.stderr
