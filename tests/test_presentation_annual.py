# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "presentation_annual",
    Path(__file__).parents[1] / "scripts/collect_presentation_annual.py",
)
assert spec is not None and spec.loader is not None
annual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annual)


def test_annual_pooling_preserves_equal_case_squared_error():
    result = annual.pooled([{"sst_rmse": 0.0}, {"sst_rmse": 2.0}])
    assert result["sst_rmse"] == pytest.approx(2**0.5)
    assert result["sst_rmse"] != 1.0  # Averaging case RMSE would understate error.


def test_pooling_cannot_hide_a_failed_forecast():
    with pytest.raises(ValueError, match="Nonfinite"):
        annual.pooled([{"sst_rmse": 1.0}, {"sst_rmse": float("nan")}])


def test_fixed_lead_ohc_compares_calendar_month_not_instantaneous_day():
    record = {
        "leads": {"30": {"sst_rmse": 1}, "365": {"sst_rmse": 2}},
        "monthly_ohc": {
            "2018-01": {"0_700": 3, "700_2000": 4},
            "2018-12": {"0_700": 5, "700_2000": 6},
        },
    }
    short = annual.endpoint(record, "2018-01-01", 30)
    long = annual.endpoint(record, "2018-01-01", 365)
    assert short["ohc_0_700_rmse"] == 3
    assert long["ohc_0_700_rmse"] == 5
    assert long["sst_rmse"] == 2


def test_output_hash_audit_requires_every_result_and_detects_changes(tmp_path):
    from samudra.experiments.annual_artifacts import annual_output_hashes

    complete = {
        "origins": {
            "2018-01-01": {
                "metrics": "2018-01-01.json",
                "arrays": "2018-01-01.npz",
                "persistence": "2018-01-01-persistence.json",
            }
        }
    }
    names = [
        "input.json",
        "COMPLETE.json",
        "2018-01-01.json",
        "2018-01-01.npz",
        "2018-01-01-persistence.json",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"original")
    before = annual_output_hashes(tmp_path, complete)
    assert len(before) == 5
    (tmp_path / "2018-01-01.npz").write_bytes(b"changed")
    assert annual_output_hashes(tmp_path, complete) != before
    (tmp_path / "2018-01-01-persistence.json").unlink()
    with pytest.raises(ValueError, match="Missing or invalid"):
        annual_output_hashes(tmp_path, complete)


@pytest.mark.parametrize("native", [False, True])
def test_collector_selects_archive_geometry_and_observation_normalization(
    tmp_path, monkeypatch, native
):
    calls: list[object] = []

    def samples(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            use_observation_normalization=lambda: calls.append("normalized")
        )

    monkeypatch.setattr(annual, "Samples", samples)
    if native:
        (tmp_path / "iap-grid.npz").touch()
    annual.collector_samples({"observations": str(tmp_path)})
    assert calls == [
        (
            (tmp_path, "cpu", "zero"),
            {
                "global_observations": True,
                **({"iap_grid": tmp_path / "iap-grid.npz"} if native else {}),
            },
        ),
        "normalized",
    ]
    with pytest.raises(ValueError, match="grid is missing"):
        annual.collector_samples(
            {"observations": str(tmp_path), "iap_grid": str(tmp_path / "missing.npz")}
        )


def native_data():
    """Model and IAP shapes deliberately differ; IAP has 14 T/S depths."""
    values = np.broadcast_to(
        np.array([[[2.0, 4.0]], [[6.0, 8.0]]]), (12, 2, 1, 2)
    ).copy()
    values[11] += 10

    def reject_model_state(*args):
        raise AssertionError("Native IAP climatology must not construct a model state")

    return SimpleNamespace(
        dual_grid=True,
        grid={"mask": np.ones((77, 2, 3), dtype=bool)},
        iap_grid={"lat": np.array([0.0]), "lon": np.array([0.0, 180.0])},
        ohc_support=torch.tensor([[[True, False]], [[True, True]]]),
        ohc_area=torch.tensor([[1.0, 3.0]]),
        interior_climatology=np.zeros((12, 28, 1, 2)),
        stats={
            "ohc_climatology": values,
            "surface_climatology": np.zeros((12, 2, 2, 3)),
        },
        ohc=reject_model_state,
    )


def test_native_climatology_preserves_iap_cells_and_fixed_support():
    data = native_data()
    values = annual.seasonal_ohc(data)
    np.testing.assert_equal(values[0], [[[2.0, np.nan]], [[6.0, 8.0]]])
    np.testing.assert_equal(values[11], [[[12.0, np.nan]], [[16.0, 18.0]]])
    data.stats["ohc_climatology"] = np.zeros((12, 2, 2, 3))
    with pytest.raises(ValueError, match="IAP support differ"):
        annual.seasonal_ohc(data)


def test_legacy_climatology_keeps_state_reconstruction_and_surface_override():
    states = []

    def ohc(state):
        states.append(state.clone())
        return state[:, :2].numpy()

    mask = torch.ones((1, 77, 2, 3))
    mask[:, :, 0, 0] = 0
    data = SimpleNamespace(
        dual_grid=False,
        grid={"mask": mask[0].numpy()},
        ts_indices=list(range(38, 52)) + list(range(57, 71)),
        interior_climatology=np.full((12, 28, 2, 3), 7.0),
        stats={"surface_climatology": np.full((12, 2, 2, 3), 9.0)},
        mean=torch.full((1, 1, 77, 1, 1), 3.0),
        std=torch.full((1, 1, 77, 1, 1), 2.0),
        mask=mask,
        tensor=lambda x: torch.as_tensor(x, dtype=torch.float32),
        ohc=ohc,
    )
    assert set(annual.seasonal_ohc(data)) == {0, 11}
    expected = torch.zeros_like(mask)
    expected[:, data.ts_indices] = 2
    expected[:, [38, 76]] = 3
    for state in states:
        torch.testing.assert_close(state, expected * mask, rtol=0, atol=0)
    assert len(states) == 2


@pytest.fixture
def native_annual_archive(tmp_path, monkeypatch):
    data = native_data()
    origins = ["2015-01-01", "2018-01-01", "2022-01-01"]
    model = {"name": "U-global-quarter", "checkpoint_sha256": "fixed-checkpoint"}
    config = dict(
        origins=origins, models=[model], observations="unused", annual_data="unused"
    )
    annual.atomic_json(config, tmp_path / "paths.json")
    folder = tmp_path / model["name"]
    folder.mkdir()
    origin_files: dict[str, dict[str, str]] = {}
    complete = {
        "origins": origin_files,
        "inputs": {"checkpoint_sha256": model["checkpoint_sha256"]},
    }
    raws = {}
    for index, origin in enumerate(origins):
        midpoint = pd.date_range(origin, periods=92, freq="5D").to_numpy()
        raw = dict(
            midpoint=midpoint,
            surface=np.zeros((92, 2, 2, 3)),
            velocity=np.zeros((92, 2, 2, 3)),
        )
        raws[origin] = raw
        metrics: dict[str, dict[str, dict[str, int]]] = {"leads": {}, "monthly_ohc": {}}
        for day, month, multiplier in ((30, "01", 1), (365, "12", 2)):
            factor = multiplier * (index + 1)
            metrics["leads"][str(day)] = dict(
                sst_rmse=factor, velocity_rmse=2 * factor, adt_rmse=3 * factor
            )
            metrics["monthly_ohc"][origin[:4] + "-" + month] = {
                "0_700": 4 * factor,
                "700_2000": 5 * factor,
            }
        item = dict(
            metrics=origin + ".json",
            arrays=origin + ".npz",
            persistence=origin + "-persistence.json",
        )
        origin_files[origin] = item
        for kind in ("metrics", "persistence"):
            annual.atomic_json(metrics, folder / item[kind])
        np.savez(
            folder / item["arrays"],
            months=[f"{origin[:4]}-{month:02d}" for month in range(1, 13)],
            surface=np.zeros((73, 2, 2, 3)),
            reference=np.zeros((73, 4, 2, 3)),
            reference_ohc=np.full((12, 2, 1, 2), index, dtype=float),
            iap_lat=data.iap_grid["lat"],
            iap_lon=data.iap_grid["lon"],
        )
    annual.atomic_json({}, folder / "input.json")
    annual.atomic_json(complete, folder / "COMPLETE.json")
    annual.atomic_json(
        dict(
            config_sha256=annual.digest(tmp_path / "paths.json"),
            model=model,
            outputs=annual.annual_output_hashes(folder, complete),
        ),
        folder / "VERIFIED.json",
    )
    monkeypatch.setattr(annual, "collector_samples", lambda config: data)
    monkeypatch.setattr(annual, "read_origin", lambda path: ({}, raws[path.name]))

    def surface_metrics(prediction, truth, samples):
        assert samples is data
        assert prediction.shape == (73, 2, 2, 3)
        assert truth.shape == (73, 4, 2, 3)
        return {
            "leads": {
                str(day): dict(sst_rmse=2.0, velocity_rmse=4.0, adt_rmse=3.0)
                for day in (30, 365)
            }
        }

    monkeypatch.setattr(annual, "surface_metrics", surface_metrics)
    return tmp_path, data


def test_native_annual_pools_three_origins_with_iap_weighted_control(
    native_annual_archive,
):
    root, _ = native_annual_archive
    output = root / "collected"
    annual.collect(root, output)
    rows = pd.read_csv(output / "rmse-scores.csv")
    # Independently pool three origins, retaining different physical IAP cell areas.
    factor = np.sqrt((1 + 4 + 9) / 3)
    for day, shift, multiplier in ((30, 0, 1), (365, 10, 2)):
        shallow = np.sqrt(sum((2 + shift - index) ** 2 for index in range(3)) / 3)
        deep = np.sqrt(
            sum(
                ((6 + shift - index) ** 2 + 3 * (8 + shift - index) ** 2) / 4
                for index in range(3)
            )
            / 3
        )
        selected = rows[(rows.model == "U-global-quarter") & (rows.lead_days == day)]
        pooled = selected[selected.origin == "pooled"].iloc[0]
        assert pooled.sst_rmse == pytest.approx(multiplier * factor)
        expected = multiplier * factor * np.mean([1 / 2, 2 / 4, 4 / shallow, 5 / deep])
        assert pooled.rmse_score == pytest.approx(expected)
        # Individual rows use the same pooled control, not an origin-specific denominator.
        assert selected[selected.origin == "2015-01-01"].iloc[
            0
        ].rmse_score == pytest.approx(expected / factor)
        climate = rows[
            (rows.model == "Training seasonal climatology")
            & (rows.lead_days == day)
            & (rows.origin == "pooled")
        ].iloc[0]
        assert climate.ohc_0_700_rmse == pytest.approx(shallow)
        assert climate.ohc_700_2000_rmse == pytest.approx(deep)
        assert climate.rmse_score == pytest.approx(1.0)
    results = json.loads((output / "results.json").read_text())
    assert "native IAP cells" in results["control_definition"]
    assert len(rows) == 3 * 4 * 2  # Model, initialized persistence, climatology.


def test_native_annual_rejects_exported_iap_geometry_mismatch(native_annual_archive):
    root, data = native_annual_archive
    data.iap_grid["lat"] = np.array([1.0])
    with pytest.raises(AssertionError):
        annual.collect(root, root / "rejected")
    assert not (root / "rejected").exists()
