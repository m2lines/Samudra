# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Global normalization must include polar observations and exclude dry/missing cells."""

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from samudra.observations.archive import (
    NORMALIZATION_POLICY,
    NORMALIZATION_VERSION,
    SPLITS,
    fit_statistics,
    validate_normalization_policy,
)
from samudra.observations.constants import STATE_NAMES


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
        names=np.array(STATE_NAMES),
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


def om4_statistics(root):
    root.mkdir(parents=True, exist_ok=True)
    means = {
        name: np.float32(-20 + index * 0.75) for index, name in enumerate(STATE_NAMES)
    }
    stds = {
        name: np.float32(0.3 + index * 0.1) for index, name in enumerate(STATE_NAMES)
    }
    for suffix, values in (("means", means), ("stds", stds)):
        xr.Dataset(
            {name: xr.DataArray(value) for name, value in values.items()}
        ).to_zarr(root / f"OM4_{suffix}.zarr", mode="w", consolidated=True)
    return means, stds


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
    om4 = tmp_path / "om4"
    om4_means, om4_stds = om4_statistics(om4)
    for name, delta in (("base", 0), ("polar_changed", 100)):
        root = tmp_path / name
        prepared_months(root, polar_delta=delta)
        module.prepare_normalization(root, om4=om4)
        with (
            xr.open_zarr(root / "observation_means.zarr", chunks=None) as means,
            xr.open_zarr(root / "observation_stds.zarr", chunks=None) as stds,
        ):
            outputs.append(
                (
                    np.array([means[name].item() for name in STATE_NAMES]),
                    np.array([stds[name].item() for name in STATE_NAMES]),
                )
            )
        metadata = json.loads((root / "normalization.json").read_text())
        validate_normalization_policy(metadata)
        assert metadata["normalization_version"] == NORMALIZATION_VERSION
        assert metadata["normalization_policy"]["spatial_domain"] == "global"
    base_mean, base_std = outputs[0]
    changed_mean, changed_std = outputs[1]
    # A +/-60-degree fit would be unchanged: every changed cell is at +/-75.
    assert changed_mean[38] - base_mean[38] > 20
    assert changed_std[38] > base_std[38]
    for field in module.OM4_STATE_NAMES:
        index = STATE_NAMES.index(field)
        assert changed_mean[index] == base_mean[index] == om4_means[field]
        assert changed_std[index] == base_std[index] == om4_stds[field]
    assert (
        changed_mean[STATE_NAMES.index("thetao_14")]
        != changed_mean[STATE_NAMES.index("thetao_13")]
    )


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
        metadata["normalization_version"] = 2
    elif change == "latitude_cutoff":
        metadata["normalization_policy"]["spatial_domain"] = "abs(latitude)<=60"
    else:
        metadata["training_split"] = ["1993-05", "2014-07"]
    with pytest.raises(ValueError, match="regenerate"):
        validate_normalization_policy(metadata)


def test_fit_rejects_incomplete_training_cohort_before_writing_artifact(tmp_path):
    grid = prepared_months(tmp_path)
    (tmp_path / "train" / "1993-05.npz").unlink()
    target = tmp_path / "statistics.npz"
    with pytest.raises(ValueError, match="complete May 1993-July 2013"):
        fit_statistics(tmp_path / "train", grid, target)
    assert not target.exists()


def test_named_zarr_normalization_loads_into_training_samples(tmp_path):
    import torch

    from samudra.constants import build_om4_layout
    from samudra.observations.samples import Samples

    assert list(STATE_NAMES) == build_om4_layout().prognostic_var_names
    module = load_script("data/prepare_observation_normalization.py")
    prepared_months(tmp_path)
    om4 = tmp_path / "om4"
    om4_means, om4_stds = om4_statistics(om4)
    module.prepare_normalization(tmp_path, om4=om4)
    samples = Samples(tmp_path, torch.device("cpu"))
    with xr.open_zarr(tmp_path / "observation_means.zarr", chunks=None) as means:
        np.testing.assert_array_equal(
            samples.grid["mean"], [means[name].item() for name in STATE_NAMES]
        )
    assert samples.grid["mean"][STATE_NAMES.index("uo_0")] == om4_means["uo_0"]
    assert samples.grid["std"][STATE_NAMES.index("vo_18")] == om4_stds["vo_18"]
    metadata = json.loads((tmp_path / "normalization.json").read_text())
    assert any(
        "observation_means.zarr/thetao_0/" in name for name in metadata["sha256"]
    )


def test_normalization_records_om4_source_and_rejects_changed_input(tmp_path):
    import zarr  # type: ignore[import-untyped]

    from samudra.observations.training import verify_normalization

    module = load_script("data/prepare_observation_normalization.py")
    root, om4 = tmp_path / "obs", tmp_path / "om4"
    prepared_months(root)
    om4_statistics(om4)
    module.prepare_normalization(root, om4=om4)
    verify_normalization(root)
    metadata = json.loads((root / "normalization.json").read_text())
    sources = metadata["field_sources"]
    assert sources["uo_0"] == sources["vo_18"] == "OM4"
    assert sources["thetao_14"] == sources["so_18"] == "OM4"
    assert sources["thetao_13"] == sources["so_13"] == "IAP"
    assert sources["thetao_0"] == "OISST" and sources["zos"] == "DUACS"
    assert len(sources) == len(STATE_NAMES)
    assert metadata["om4_normalization_inputs"]["means"] == str(
        (om4 / "OM4_means.zarr").resolve()
    )
    assert any("OM4_means.zarr/uo_0/" in path for path in metadata["sha256"])
    with xr.open_zarr(root / "observation_stds.zarr", chunks=None) as store:
        assert store["so_18"].attrs["normalization_source"] == "OM4"
    store = zarr.open_group(str(om4 / "OM4_means.zarr"), mode="a")
    store["uo_0"][...] = 17
    with pytest.raises(ValueError, match="Normalization input changed"):
        verify_normalization(root)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_store",
        "missing_field",
        "nonscalar",
        "nonfinite_mean",
        "nonfinite_std",
        "zero_std",
        "negative_std",
    ],
)
def test_om4_fallback_requires_complete_valid_statistics(tmp_path, failure):
    import shutil

    import zarr  # type: ignore[import-untyped]

    module = load_script("data/prepare_observation_normalization.py")
    om4 = tmp_path / "om4"
    om4_statistics(om4)
    if failure == "missing_store":
        shutil.rmtree(om4 / "OM4_stds.zarr")
    else:
        suffix = (
            "means"
            if failure in ("missing_field", "nonscalar", "nonfinite_mean")
            else "stds"
        )
        store_path = om4 / f"OM4_{suffix}.zarr"
        store = zarr.open_group(str(store_path), mode="a")
        if failure == "missing_field":
            del store["thetao_14"]
        elif failure == "nonscalar":
            del store["uo_0"]
            store["uo_0"] = np.array([1.0, 2.0])
        else:
            store["uo_0"][...] = {
                "nonfinite_mean": np.nan,
                "nonfinite_std": np.inf,
                "zero_std": 0,
                "negative_std": -1,
            }[failure]
        zarr.consolidate_metadata(str(store_path))
    with pytest.raises(ValueError, match="OM4"):
        module.prepare_normalization(tmp_path / "unused-observations", om4=om4)
