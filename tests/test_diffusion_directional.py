# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scientific invariants of the image-grid directional diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "scripts/report_diffusion_directional.py"
_SPEC = importlib.util.spec_from_file_location("diffusion_directional", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def test_axis_and_diagonal_gratings_have_opposite_fourfold_signs():
    y, x = np.indices((64, 64))
    axis = np.sin(2 * np.pi * 10 * x / 64)
    diagonal = np.sin(2 * np.pi * 7 * (x + y) / 64)
    power, moment = report.band_moments(np.stack((axis, diagonal)))
    coefficient = report.normalized_moment(power, moment)
    assert coefficient[0, 1] > 0.95
    assert coefficient[1, 1] < -0.95
    np.testing.assert_allclose(power[:, 1], 0.5, atol=0.002)


def test_amplitude_scaling_changes_power_not_orientation():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(20, 30))
    power, moment = report.band_moments(field)
    bigger_power, bigger_moment = report.band_moments(7 * field)
    np.testing.assert_allclose(bigger_power, 49 * power, rtol=1e-12)
    np.testing.assert_allclose(
        report.normalized_moment(power, moment),
        report.normalized_moment(bigger_power, bigger_moment),
        atol=1e-12,
    )


def test_planes_constants_and_zero_have_finite_zero_power():
    y, x = np.indices((20, 30))
    fields = np.stack((np.zeros_like(x), np.ones_like(x), 3 * x - 5 * y + 4))
    power, moment = report.band_moments(fields)
    np.testing.assert_array_equal(power, 0)
    np.testing.assert_array_equal(moment, 0)
    assert np.isnan(report.normalized_moment(power, moment)).all()


def test_plane_removal_preserves_superposed_signal():
    y, x = np.indices((32, 32))
    signal = np.sin(2 * np.pi * (x + y) / 8)
    clean = report.band_moments(signal)
    trended = report.band_moments(signal + x + 3 * y + 10)
    for actual, expected in zip(trended, clean, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_missing_values_fail_instead_of_filling():
    field = np.ones((20, 30))
    field[0, 0] = np.nan
    with pytest.raises(ValueError, match="fully observed"):
        report.band_moments(field)


def test_member_power_does_not_use_spectrum_of_member_mean():
    y, x = np.indices((32, 32))
    wave = np.sin(2 * np.pi * (x + y) / 8)
    members = np.stack((wave, -wave))
    member_power, _ = report.band_moments(members)
    mean_power, _ = report.band_moments(members.mean(0))
    assert member_power.mean(0).sum() > 0.4
    np.testing.assert_array_equal(mean_power, 0)


@pytest.mark.parametrize("corruption", ["checksum", "reference", "mask"])
def test_reader_rejects_unmatched_inputs(tmp_path, monkeypatch, corruption):
    import json

    monkeypatch.setattr(
        report.spectra, "SPATIAL_REGIONS", (("box", slice(0, 19), slice(0, 19)),)
    )
    origins = [f"2014-{month:02d}" for month in range(1, 10)]
    for arm in ("control", "modified"):
        directory = tmp_path / arm / "validation"
        directory.mkdir(parents=True)
        files = {}
        for origin in origins:
            reference = np.ones((2, 20, 20))
            mask = np.ones((2, 20, 20), dtype=bool)
            if arm == "modified" and origin == origins[0]:
                if corruption == "reference":
                    reference[0, 0, 0] = 2
                if corruption == "mask":
                    mask[0, 0, 0] = False
            filename = f"members-{origin}.npz"
            np.savez(
                directory / filename,
                lat=np.arange(20),
                lon=np.arange(20),
                mask=mask,
                day30_surface_reference=reference,
                day30=np.ones((8, 2, 20, 20)),
            )
            files[filename] = report.sha(directory / filename)
        if arm == "modified" and corruption == "checksum":
            files[f"members-{origins[0]}.npz"] = "not-the-file-checksum"
        (directory / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    metrics=dict(origins=origins),
                    files=files,
                    checkpoint_sha256="test",
                    step=0,
                    counts={},
                    sampling_steps=32,
                )
            )
        )
    with pytest.raises((ValueError, AssertionError)):
        report.collect(tmp_path, ["control", "modified"])
