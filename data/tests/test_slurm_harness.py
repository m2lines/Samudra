# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

VARIANT_SCRIPT = Path(__file__).parents[2] / "scripts" / "om4_data_variant.sh"
SLURM_HARNESSES = [
    VARIANT_SCRIPT.parent / "slurm_preprocess_om4.sbatch",
    VARIANT_SCRIPT.parent / "slurm_make_norm_om4.sbatch",
]


def _resolve_variant(variant: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"DATA_VARIANT": variant}
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; resolve_om4_data_variant || exit $?; '
            'printf "%s|%s|%s|%s" "$DATA_VARIANT" "$OM4_SOURCE_STORE" '
            '"$OM4_OUTPUT_SUFFIX" "$OM4_WFO_SOURCE_STORE"',
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
        ("averaged", "averaged|om4_5daily.zarr||"),
        (
            "averaged_with_wfo",
            "averaged_with_wfo|om4_5daily.zarr|_with_wfo|om4_5daily_snapshots.zarr",
        ),
        ("snapshots", "snapshots|om4_5daily_snapshots.zarr|_snapshots|"),
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


@pytest.mark.parametrize("harness", SLURM_HARNESSES)
def test_slurm_harness_resolves_sidecar_from_repo_checkout(harness):
    script = harness.read_text()

    # sbatch executes a copy under Slurm's spool directory, so BASH_SOURCE
    # cannot locate a helper stored beside the original script.
    assert 'dirname -- "${BASH_SOURCE[0]}"' not in script
    assert 'VARIANT_SCRIPT="${REPO_DIR%/}/scripts/om4_data_variant.sh"' in script
    assert script.index('REPO_DIR="${REPO_DIR:-') < script.index(
        'source "${VARIANT_SCRIPT}"'
    )


def test_preprocessing_harness_runs_selected_data_subproject():
    script = SLURM_HARNESSES[0].read_text()

    # An editable conda install may reference another checkout. Running from
    # REPO_DIR/data ensures Python imports the revision selected for this job.
    assert 'cd "${REPO_DIR}/data"' in script
    assert script.index('cd "${REPO_DIR}/data"') < script.index(
        "\npython -m ocean_preprocessing om4"
    )
