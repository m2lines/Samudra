# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline reporting cannot change cohorts, truth, geometry or archived bytes."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from samudra.experiments.observation_common_collect import (
    COMPONENTS,
    CONTROL,
    Audit,
    Candidate,
    annual_rows,
    annual_truth,
    pool_components,
)
from samudra.experiments.observation_common_grid import CommonGrid
from samudra.experiments.observation_pilot import digest


def geometry():
    grid = dict(
        lat=np.linspace(-85, 85, 18),
        lon=np.arange(16) * 22.5,
        mask=np.ones((77, 18, 16), dtype=bool),
    )
    source = {**grid, "mask": grid["mask"].copy()}
    source["mask"][:, 5, 5] = False
    return CommonGrid(source, grid)


def write_json(path, value):
    path.write_text(json.dumps(value))


def candidate_files(tmp_path, format, g, arrays, origins):
    root = tmp_path / format
    root.mkdir()
    signature = dict(checkpoint_sha256="a" * 64, split="test", global_observations=True)
    write_json(root / "evaluation-input.json", signature)
    write_json(
        root / "selected.json", dict(sha256="a" * 64, metrics=dict(origins=origins))
    )
    complete: dict[str, Any] = dict(origins=len(origins), split="test")
    if format == "legacy":
        np.savez(root / "selected-predictions.npz", **arrays, origins=origins)
    else:
        complete["common_grid_sidecars"] = {}
        for i, origin in enumerate(origins):
            name = f"selected/{origin}.npz"
            path = root / "common-grid" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                prediction=arrays["prediction"][i],
                predicted_ohc=arrays["predicted_ohc"][i],
                origins=[origin],
                months=np.asarray([], dtype=str),
                **g.geometry_arrays(),
            )
            complete["common_grid_sidecars"][name] = digest(path)
    write_json(root / "COMPLETE.json", complete)
    return dict(
        name=format,
        format=format,
        evaluation=str(root),
        label="selected",
        checkpoint_sha256="a" * 64,
        artifacts={
            p.relative_to(root).as_posix(): digest(p)
            for p in root.rglob("*")
            if p.is_file()
        },
    )


def monthly_arrays():
    rng = np.random.default_rng(6)
    return dict(
        prediction=rng.normal(size=(3, 6, 2, 18, 16)),
        reference=rng.normal(size=(3, 6, 4, 18, 16)),
        predicted_ohc=rng.normal(size=(3, 2, 18, 16)),
        reference_ohc=rng.normal(size=(3, 2, 18, 16)),
    )


def test_formats_score_identically_on_canonical_truth_and_fixed_geometry(tmp_path):
    g, arrays = geometry(), monthly_arrays()
    origins = ["2015-01", "2015-02", "2015-03"]
    kwargs = dict(reference=arrays["reference"], reference_ohc=arrays["reference_ohc"])
    result = []
    for format in ("legacy", "common"):
        model = candidate_files(tmp_path, format, g, arrays, origins)
        candidate = Candidate(model, "monthly", "test", origins, g, Audit())
        prediction, ohc = candidate.monthly(**kwargs)
        result.append(
            g.score(
                prediction,
                ohc,
                **kwargs,
                climatology_surface=arrays["prediction"] * 0.5,
                climatology_ohc=arrays["predicted_ohc"] * 0.5,
            )
        )
    assert result[0] == result[1]
    # A candidate cannot make a bad forecast disappear by changing truth validity.
    candidate = Candidate(model, "monthly", "test", origins, g, Audit())
    bad = arrays["prediction"].copy()
    bad[0, 5, 0, 8, 8] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        g.score(
            bad,
            arrays["predicted_ohc"],
            **kwargs,
            climatology_surface=arrays["prediction"],
            climatology_ohc=arrays["predicted_ohc"],
        )


def test_candidate_rejects_unpinned_tampered_and_different_geometry(tmp_path):
    g, arrays = geometry(), monthly_arrays()
    origins = ["2015-01", "2015-02", "2015-03"]
    model = candidate_files(tmp_path, "common", g, arrays, origins)
    candidate = Candidate(model, "monthly", "test", origins, g, Audit())
    del model["artifacts"]["common-grid/selected/2015-01.npz"]
    with pytest.raises(ValueError, match="Missing pinned"):
        candidate.sidecar(origins[0], [])
    path = candidate.root / "common-grid/selected/2015-01.npz"
    model["artifacts"]["common-grid/selected/2015-01.npz"] = "0" * 64
    with pytest.raises(ValueError, match="checksum differs"):
        candidate.sidecar(origins[0], [])
    model["artifacts"]["common-grid/selected/2015-01.npz"] = digest(path)
    g.surface_support[8, 8] = False
    with pytest.raises(AssertionError, match="surface_support"):
        candidate.sidecar(origins[0], [])


@pytest.mark.parametrize("format", ["common", "legacy"])
def test_monthly_own_persistence_is_checkpoint_bound_and_not_evolved(tmp_path, format):
    g, arrays = geometry(), monthly_arrays()
    origins = ["2015-01", "2015-02", "2015-03"]
    model = candidate_files(tmp_path, format, g, arrays, origins)
    root = tmp_path / format
    held = arrays["prediction"] * 0 + 7
    held_ohc = arrays["predicted_ohc"] * 0 + 9
    write_json(root / "selected-inferred-persistence.json", dict(origins=origins))
    if format == "legacy":
        np.savez(
            root / "selected-inferred-persistence.npz",
            **{**arrays, "prediction": held, "predicted_ohc": held_ohc},
            origins=origins,
        )
    else:
        complete = json.loads((root / "COMPLETE.json").read_text())
        for i, origin in enumerate(origins):
            name = f"selected-inferred-persistence/{origin}.npz"
            path = root / "common-grid" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                prediction=held[i],
                predicted_ohc=held_ohc[i],
                origins=[origin],
                months=np.asarray([], dtype=str),
                **g.geometry_arrays(),
            )
            complete["common_grid_sidecars"][name] = digest(path)
        write_json(root / "COMPLETE.json", complete)
    model["artifacts"] = {
        p.relative_to(root).as_posix(): digest(p)
        for p in root.rglob("*")
        if p.is_file()
    }
    model["forecast"] = "initialized-persistence"
    actual = Candidate(model, "monthly", "test", origins, g, Audit()).monthly(
        reference=arrays["reference"], reference_ohc=arrays["reference_ohc"]
    )
    np.testing.assert_array_equal(actual[0], held)
    np.testing.assert_array_equal(actual[1], held_ohc)
    model["checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        Candidate(model, "monthly", "test", origins, g, Audit())


def test_legacy_truth_mask_cannot_silently_remove_a_wet_cell(tmp_path):
    g, arrays = geometry(), monthly_arrays()
    origins = ["2015-01", "2015-02", "2015-03"]
    model = candidate_files(tmp_path, "legacy", g, arrays, origins)
    candidate = Candidate(model, "monthly", "test", origins, g, Audit())
    expected = arrays["reference_ohc"].copy()
    expected[0, 0, 8, 8] = np.nan
    with pytest.raises(AssertionError, match="canonical truth"):
        candidate.monthly(reference=arrays["reference"], reference_ohc=expected)
    # Differences outside the prespecified intersection do not affect scoring.
    expected = arrays["reference_ohc"].copy()
    expected[:, :, 5, 5] = np.nan
    candidate.monthly(reference=arrays["reference"], reference_ohc=expected)
    with pytest.raises(ValueError, match="cohort"):
        Candidate(model, "monthly", "test", origins[:2], g, Audit())


def test_endpoint_blend_pools_mse_and_uses_common_pooled_denominators():
    origins = ["2015-01-01", "2018-01-01", "2021-01-01"]

    def record(origin, value):
        return dict(
            leads={
                str(day): dict(sst_rmse=value, adt_rmse=value, velocity_rmse=value)
                for day in (30, 365)
            },
            monthly_ohc={
                origin[:4] + month: {"0_700": value, "700_2000": value}
                for month in ("-01", "-12")
            },
        )

    records = {
        CONTROL: {o: record(o, 2) for o in origins},
        "model": {o: record(o, v) for o, v in zip(origins, [0, 0, 6], strict=True)},
    }
    rows = annual_rows(records, origins)
    row = next(r for r in rows if r["model"] == "model" and r["origin"] == "pooled")
    assert row["rmse_score"] == pytest.approx(np.sqrt(12) / 2)
    assert all(row[k] == pytest.approx(np.sqrt(12)) for k in COMPONENTS)
    del records["model"][origins[-1]]
    with pytest.raises(ValueError, match="Incomplete candidate"):
        annual_rows(records, origins)
    with pytest.raises(ValueError, match="Nonfinite"):
        pool_components([dict(sst_rmse=np.nan)])


@pytest.mark.parametrize("persistence", [False, True])
def test_annual_formats_rescore_identical_fields_and_reject_missing_month(
    tmp_path, monkeypatch, persistence
):
    from samudra.experiments import observation_annual

    monkeypatch.setattr(observation_annual, "spatial_error", lambda *args: None)
    g, origin = geometry(), "2014-01-01"
    months = [f"2014-{i:02d}" for i in range(1, 13)]
    rng = np.random.default_rng(13)
    prediction = rng.normal(size=(73, 2, 18, 16))
    ohc = rng.normal(size=(12, 2, 18, 16))
    truth = rng.normal(size=(73, 4, 18, 16))
    ohc_truth = rng.normal(size=(12, 2, 18, 16))
    results = []
    for format in ("legacy", "common"):
        root = tmp_path / format
        root.mkdir()
        signature = dict(
            checkpoint_sha256="b" * 64,
            split="validation",
            global_observations=True,
            include_persistence=persistence,
        )
        complete = dict(
            inputs=signature, origins={origin: dict(arrays=f"{origin}.npz")}
        )
        if format == "common":
            directory = root / "common-grid"
            directory.mkdir()
            path = directory / f"{origin}{'-persistence' if persistence else ''}.npz"
            np.savez(
                path,
                prediction=prediction * 0 + 7 if persistence else prediction,
                predicted_ohc=np.broadcast_to(ohc[0], ohc.shape)
                if persistence
                else ohc,
                months=months,
                origins=[origin],
                **g.geometry_arrays(),
            )
            complete["common_grid_sidecars"] = {path.name: digest(path)}
        else:
            np.savez(
                root / f"{origin}.npz",
                surface=prediction,
                predicted_ohc=ohc,
                reference=truth,
                reference_ohc=ohc_truth,
                months=months,
                persistence_surface=prediction * 0 + 7,
                persistence_ohc=ohc[0],
            )
        write_json(root / "COMPLETE.json", complete)
        write_json(root / "input.json", signature)
        model = dict(
            name=format,
            format=format,
            evaluation=str(root),
            checkpoint_sha256="b" * 64,
            forecast="initialized-persistence" if persistence else "evolved",
            artifacts={
                p.relative_to(root).as_posix(): digest(p)
                for p in root.rglob("*")
                if p.is_file()
            },
        )
        candidate = Candidate(model, "annual", "validation", [origin], g, Audit())
        arrays = candidate.annual(
            origin, months, reference=truth, reference_ohc=ohc_truth
        )
        np.testing.assert_array_equal(
            arrays[0], prediction * 0 + 7 if persistence else prediction
        )
        np.testing.assert_array_equal(
            arrays[1], np.broadcast_to(ohc[0], ohc.shape) if persistence else ohc
        )
        results.append(
            g.score_annual(
                *arrays, reference=truth, reference_ohc=ohc_truth, months=months
            )
        )
        with pytest.raises(AssertionError):
            candidate.annual(
                origin, months[:-1], reference=truth, reference_ohc=ohc_truth
            )
    assert results[0] == results[1]
    bad = ohc.copy()
    bad[11, 1, 8, 8] = np.nan
    with pytest.raises(ValueError, match="fixed common OHC support"):
        g.score_annual(
            prediction, bad, reference=truth, reference_ohc=ohc_truth, months=months
        )


def test_annual_truth_uses_canonical_payloads_and_checks_calendar(tmp_path):
    origin = "2015-01-01"
    root = tmp_path / origin
    root.mkdir()
    records = []
    for index in range(92):
        start = pd.Timestamp(origin) + pd.Timedelta(days=5 * (index - 19))
        path = root / f"bin-{index:02d}.npz"
        np.savez(
            path,
            surface=np.full((2, 2, 3), index),
            velocity=np.ones((2, 2, 3)),
            atmosphere=np.zeros((8, 2, 3)),
            start=np.array(str(start)),
            end=np.array(str(start + pd.Timedelta(days=5))),
            midpoint=np.array(str(start + pd.Timedelta(days=2.5))),
        )
        records.append(dict(file=path.name, sha256=digest(path)))
    interiors = []
    for month in range(1, 13):
        path = root / f"2015-{month:02d}.npz"
        np.savez(path, ohc=np.full((2, 2, 3), month))
        interiors.append(dict(file=path.name, sha256=digest(path)))
    manifest: dict[str, Any] = dict(
        origin=origin,
        history_bins=19,
        forecast_bins=73,
        bin_days=5,
        records=records,
        monthly_interiors=interiors,
    )
    write_json(root / "COMPLETE.json", manifest)
    audit = Audit()
    truth, ohc, months, midpoint_months = annual_truth(tmp_path, origin, audit)
    assert truth.shape == (73, 4, 2, 3)
    assert np.all(truth[0, :2] == 19)
    assert np.all(ohc[-1] == 12)
    assert months[-1] == "2015-12" and midpoint_months[-1] == 11
    assert len(audit.files) == 105
    manifest["monthly_interiors"].reverse()
    write_json(root / "COMPLETE.json", manifest)
    with pytest.raises(ValueError, match="calendar-month"):
        annual_truth(tmp_path, origin, Audit())
    (root / records[0]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum differs"):
        annual_truth(tmp_path, origin, Audit())
