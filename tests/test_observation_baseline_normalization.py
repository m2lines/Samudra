# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Global normalization must include polar observations and exclude dry/missing cells."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from samudra.observations.archive import (
    NORMALIZATION_POLICY,
    NORMALIZATION_VERSION,
    SPLITS,
    fit_statistics,
    validate_normalization_policy,
)


def load_script(name):
    path = Path(__file__).parents[1] / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_months(root, polar_delta=0):
    train = root / "train"
    train.mkdir(parents=True)
    latitude = np.array([-75.0, 0.0, 75.0])
    mask = np.ones((77, 3, 2), dtype=bool)
    mask[:, 1, 1] = False
    mask[39, 0, 0] = False  # A channel-specific dry interior cell.
    grid = dict(
        lat=latitude,
        lon=np.array([0.0, 180.0]),
        mask=mask,
        mean=np.zeros(77),
        std=np.ones(77),
    )
    np.savez(root / "grid.npz", **grid)
    for index, month in enumerate(pd.period_range(*SPLITS["train"], freq="M")):
        fields = {}
        for name, channels, times in (
            ("surface", 2, 2),
            ("atmosphere", 8, 2),
            ("interior", 28, 1),
        ):
            values = (
                np.arange(channels)[None, :, None, None]
                + np.arange(times)[:, None, None, None] * 0.5
                + np.array([8.0 + polar_delta, 1.0, -4.0 + polar_delta])[
                    None, None, :, None
                ]
                + np.array([0.0, 0.25])[None, None, None, :]
                + index * 0.01
            ).astype(np.float32)
            values[..., 1, 1] = 1e6  # Finite dry values must not enter any moments.
            if name == "interior":
                values[:, 1, 0, 0] = -1e6  # Depth-specific dry cell must be excluded.
                values = values[0]
            else:
                values[0, 0, 2, 1] = np.nan  # Finite support varies with channel/time.
            fields[name] = values
        dates = [
            month.start_time.isoformat(),
            (month.start_time + pd.Timedelta(days=5)).isoformat(),
        ]
        np.savez(train / f"{month}.npz", **fields, midpoints=np.array(dates))
    return grid


def independent_weighted_statistics(root, grid, name):
    indices = list(range(38, 52)) + list(range(57, 71))
    support = {
        "surface": grid["mask"][[38, 76]],
        "atmosphere": grid["mask"][38][None],
        "interior": grid["mask"][indices],
    }[name]
    raw = []
    for path in sorted((root / "train").glob("*.npz")):
        with np.load(path) as sample:
            value = sample[name]
        raw.append(value[None] if name == "interior" else value)
    values = np.concatenate(raw, axis=0).astype(np.float64)
    weights = np.broadcast_to(
        np.cos(np.deg2rad(grid["lat"]))[:, None] * support, values.shape
    )
    means, deviations = [], []
    for channel in range(values.shape[1]):
        present = np.isfinite(values[:, channel]) & (weights[:, channel] > 0)
        selected, area = values[:, channel][present], weights[:, channel][present]
        mean = np.average(selected, weights=area)
        means.append(mean)
        deviations.append(np.sqrt(np.average((selected - mean) ** 2, weights=area)))
    return np.array(means), np.array(deviations)


def test_global_statistics_match_independent_finite_channel_wet_reduction(tmp_path):
    grid = prepared_months(tmp_path)
    target = tmp_path / "statistics.npz"
    fit_statistics(tmp_path / "train", grid, target)
    with np.load(target) as fitted:
        assert fitted["training_month_count"] == 243
        for name in ("surface", "atmosphere", "interior"):
            mean, std = independent_weighted_statistics(tmp_path, grid, name)
            np.testing.assert_allclose(
                fitted[f"{name}_mean"], mean, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(fitted[f"{name}_std"], std, rtol=1e-6, atol=1e-6)


def test_polar_observations_change_fitted_statistics_and_shared_state_scales(tmp_path):
    module = load_script("data/prepare_observation_normalization.py")
    outputs = []
    for name, delta in (("base", 0), ("polar_changed", 100)):
        root = tmp_path / name
        prepared_months(root, polar_delta=delta)
        module.prepare_normalization(root)
        with np.load(root / "normalization.npz") as fitted:
            outputs.append((fitted["state_mean"].copy(), fitted["state_std"].copy()))
        metadata = json.loads((root / "normalization.json").read_text())
        validate_normalization_policy(metadata)
        assert metadata["normalization_version"] == NORMALIZATION_VERSION
        assert metadata["normalization_policy"]["spatial_domain"] == "global"
    base_mean, base_std = outputs[0]
    changed_mean, changed_std = outputs[1]
    # A +/-60-degree fit would be unchanged: every changed cell is at +/-75.
    assert changed_mean[38] - base_mean[38] > 20
    assert changed_std[38] > base_std[38]
    np.testing.assert_array_equal(changed_mean[:38], np.zeros(38))
    np.testing.assert_array_equal(changed_std[:38], np.ones(38))
    np.testing.assert_array_equal(changed_mean[52:57], np.repeat(changed_mean[51], 5))
    np.testing.assert_array_equal(changed_std[71:76], np.repeat(changed_std[70], 5))


@pytest.mark.parametrize(
    "change", ["unversioned", "old_version", "latitude_cutoff", "different_split"]
)
def test_normalization_policy_rejects_old_or_incompatible_artifacts(change):
    metadata: dict[str, Any] = {
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_policy": copy.deepcopy(NORMALIZATION_POLICY),
        "training_split": list(SPLITS["train"]),
    }
    if change == "unversioned":
        metadata.pop("normalization_version")
    elif change == "old_version":
        metadata["normalization_version"] = 1
    elif change == "latitude_cutoff":
        metadata["normalization_policy"]["spatial_domain"] = "abs(latitude)<=60"
    else:
        metadata["training_split"] = ["1993-05", "2014-07"]
    with pytest.raises(ValueError, match="regenerate"):
        validate_normalization_policy(metadata)


def test_independent_qualification_check_uses_global_moments_not_historical_stats(
    tmp_path,
):
    prepare = load_script("data/prepare_observation_normalization.py")
    qualify = load_script("scripts/qualify_observation_baseline.py")
    grid = prepared_months(tmp_path)
    prepare.prepare_normalization(tmp_path)
    with np.load(tmp_path / "statistics.npz") as fitted:
        stats = dict(fitted)
    data = SimpleNamespace(
        normalization=tmp_path,
        grid=grid,
        stats=stats,
        paths=lambda split: sorted((tmp_path / split).glob("*.npz")),
    )
    checks = qualify.verify_global_moments(data)
    assert min(checks["surface"]["polar_weight_fraction_by_channel"]) > 0
    # A plausible but latitude-restricted surface mean cannot pass the check.
    data.stats["surface_mean"] = stats["surface_mean"].copy() + 1
    with pytest.raises(AssertionError):
        qualify.verify_global_moments(data)


def test_fit_rejects_incomplete_training_cohort_before_writing_artifact(tmp_path):
    grid = prepared_months(tmp_path)
    (tmp_path / "train" / "1993-05.npz").unlink()
    target = tmp_path / "statistics.npz"
    with pytest.raises(ValueError, match="complete May 1993-July 2013"):
        fit_statistics(tmp_path / "train", grid, target)
    assert not target.exists()
