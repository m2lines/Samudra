# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Control pairing and normalized saved-member spatial scores."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.stochastic_variogram import variogram_score


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("report_stochastic_variogram")


def test_pairing_uses_each_architectures_control(report):
    frame = pd.DataFrame(
        [
            dict(variant=name, step=2000, score=value)
            for name, value in zip(report.VARIANTS, [10.0, 9.0, 2.0, 1.0], strict=True)
        ]
    )
    result = report.paired_rows(frame, ["step"], ["score"])
    assert result.control.tolist() == [10.0, 2.0]
    assert result.difference.tolist() == [-1.0, -1.0]
    assert result.relative_change_percent.tolist() == pytest.approx([-10, -50])
    with pytest.raises(ValueError, match="Incomplete"):
        report.paired_rows(frame.iloc[:3], ["step"], ["score"])


def test_module_overrides_are_restored_on_failure(report):
    original = report.shared.VARIANTS
    with pytest.raises(RuntimeError):
        with report.configured_reports():
            assert report.shared.VARIANTS == report.VARIANTS
            raise RuntimeError("test")
    assert report.shared.VARIANTS is original


def test_normalization_fingerprint_and_channel_order(report, tmp_path):
    grid = dict(
        lat=np.arange(12),
        lon=np.arange(16),
        names=np.array(report.CHANNELS),
        mask=np.ones((6, 12, 16), bool),
    )
    path = tmp_path / "grid.npz"
    np.savez(path, **grid, mean=np.zeros(6), std=np.ones(6))
    proof = dict(grid_sha256=report.shared.fingerprint(path))
    mean, std, _ = report.normalization(path, grid, proof)
    np.testing.assert_array_equal(mean, 0)
    np.testing.assert_array_equal(std, 1)
    with pytest.raises(ValueError, match="fingerprint"):
        report.normalization(path, grid, dict(grid_sha256="wrong"))
    wrong_grid = dict(grid, names=grid["names"][::-1])
    with pytest.raises(AssertionError):
        report.normalization(path, wrong_grid, proof)


def test_saved_case_matches_exact_training_normalization(report):
    rng = np.random.default_rng(42)
    members = rng.normal(size=(8, 6, 12, 16)).astype(np.float32)
    gold = rng.normal(size=(6, 12, 16)).astype(np.float32)
    mask = np.ones_like(gold, bool)
    mask[:, 3, 5] = False
    members[:, :, 3, 5] = np.nan
    lat = np.linspace(-70, 70, 12).astype(np.float32)
    mean = np.arange(6, dtype=np.float32)
    std = np.arange(1, 7, dtype=np.float32)
    rows = report.score_case(members, gold, mask, lat, mean, std, report.CHANNELS)
    assert len(rows) == 35
    actual = [row for row in rows if row["channel"] == "all-six" and row["lag"] == 0][0]
    expected = variogram_score(
        torch.from_numpy(
            (members - mean[None, :, None, None]) / std[None, :, None, None]
        )[:, None],
        torch.from_numpy((gold - mean[:, None, None]) / std[:, None, None])[None],
        torch.from_numpy(mask),
        torch.from_numpy(np.cos(np.deg2rad(lat))[:, None]),
        report.LAGS,
    )
    assert actual["fair"] == float(expected["fair"])
    channel_scores = [
        row["fair"] for row in rows if row["channel"] != "all-six" and row["lag"] == 0
    ]
    assert actual["fair"] == pytest.approx(np.mean(channel_scores), abs=1e-6)


def test_report_rejects_nonfinite_prediction_at_wet_cell(report):
    members = np.zeros((8, 6, 12, 16), dtype=np.float32)
    members[0, 0, 5, 5] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        report.score_case(
            members,
            np.zeros((6, 12, 16), np.float32),
            np.ones((6, 12, 16), bool),
            np.arange(12),
            np.zeros(6),
            np.ones(6),
            report.CHANNELS,
        )


def training_fixture(report, root):
    for variant in report.VARIANTS:
        folder = root / variant
        folder.mkdir()
        rows = []
        for step in range(1, 5):
            native = step % 2 == 0
            treatment = variant in report.PAIRS
            components = [2.0, 4.0, 3.2 if treatment and native else 3.0]
            row = dict(
                step=step,
                source="om4" if native else "observation",
                counts=dict(om4=step // 2, observation=(step + 1) // 2),
                components=components,
                grad_norm=1.0,
                lr=0.0001,
                loss=0.5 * components[0] + 0.5 * components[1] + 0.1 * components[2]
                if native
                else 5.0,
            )
            if treatment:
                for key, value in dict(
                    fair=0.2,
                    plugin=0.25,
                    finite_ensemble_correction=0.05,
                    weighted_auxiliary=0.02,
                    weighted_base_initializer_crps=0.3,
                ).items():
                    row[f"velocity_init_variogram/{key}"] = value if native else 0.0
            rows.append(row)
        (folder / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_training_decomposes_auxiliary_from_comparable_loss(report, tmp_path):
    training_fixture(report, tmp_path)
    frame, sources = report.training_rows(tmp_path, expected_updates=4)
    assert len(frame) == 16 and len(sources) == 4
    native = frame[frame.source == "om4"]
    np.testing.assert_allclose(native.om4_weighted_initializer_crps, 0.3)
    np.testing.assert_allclose(native.om4_weighted_native_forecast, 2.0)
    treatment = native[native.variant.isin(report.PAIRS)]
    np.testing.assert_allclose(treatment.om4_weighted_variogram, 0.02)
    np.testing.assert_allclose(treatment.om4_variogram_correction, 0.05)
    assert frame[frame.source == "observation"].om4_weighted_variogram.isna().all()


@pytest.mark.parametrize(
    "failure", ["counts", "order", "nonfinite", "obs_auxiliary", "decomposition"]
)
def test_training_rejects_wrong_exposure_or_loss(report, tmp_path, failure):
    training_fixture(report, tmp_path)
    path = tmp_path / "B-learned-variogram" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "counts":
        rows[-1]["counts"]["om4"] = 3
    elif failure == "order":
        rows.reverse()
    elif failure == "nonfinite":
        rows[0]["grad_norm"] = float("nan")
    elif failure == "obs_auxiliary":
        rows[0]["velocity_init_variogram/weighted_auxiliary"] = 1.0
    else:
        rows[1]["velocity_init_variogram/weighted_base_initializer_crps"] = 1.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, AssertionError)):
        report.training_rows(tmp_path, expected_updates=4)
