# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-grid routing, truthful availability and legacy control regressions."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from data import Catalog, edges, raster_rows

DEPTH_EDGES = [
    0,
    5,
    15,
    30,
    50,
    80,
    130,
    200,
    300,
    450,
    650,
    900,
    1200,
    1600,
    2100,
    2700,
    3500,
    4500,
    5500,
    6500,
]


def write_catalog(root, native=False):
    root.mkdir()
    y, x, depth_count = (4, 8, 19) if native else (2, 4, 14)
    interfaces = np.asarray(DEPTH_EDGES[: depth_count + 1])
    files = {}

    def save(name, value):
        array = np.asarray(value)
        np.save(root / name, array)
        files[name] = dict(shape=list(array.shape), dtype=str(array.dtype))
        return name

    meta = dict(
        schema_version=1,
        models={},
        records={},
        interior_examples={},
        origins=["2015-01-01"],
        lat=np.linspace(-90 + 90 / y, 90 - 90 / y, y).tolist(),
        lon=np.linspace(180 / x, 360 - 180 / x, x).tolist(),
        depths=((interfaces[:-1] + interfaces[1:]) / 2).tolist(),
        depth_interfaces=interfaces.tolist(),
        files=files,
        report_url="https://example.test/report",
    )
    surface_mask = save("surface-mask.npy", np.ones((2, y, x), bool))
    interior_mask = save("interior-mask.npy", np.ones((depth_count, 4, y, x), bool))
    heat_mask = save("heat-mask.npy", np.ones((2, 2, 4), bool))
    meta.update(
        surface_mask=surface_mask, interior_mask=interior_mask, heat_mask=heat_mask
    )
    reference = save("surface-reference.npy", np.full((73, 2, y, x), 3, np.float32))
    heat_reference = save("heat-reference.npy", np.full((12, 2, 2, 4), 4, np.float32))
    record = dict(
        models={},
        surface_reference=reference,
        heat_reference=heat_reference,
        heat_months=[f"2015-{m:02}" for m in range(1, 13)],
    )
    example = dict(
        origin="2015-01-01",
        label="2015-01-01 (obs)",
        source="obs",
        description="Saved observation initialization",
        models={},
        references={},
    )
    names = (
        ["quarter-uglobal-selected", "quarter-obs-only-selected"]
        if native
        else ["obs08000", "diffusion"]
    )
    for index, name in enumerate(names):
        meta["models"][name] = dict(
            label=name,
            short_label=name,
            lineage=dict(
                task_counts={"om4": 0, "observation": 5300},
                global_step=5300,
                checkpoint_sha256="saved-checkpoint",
            ),
        )
        values = np.full((73, 2, y, x), index + 10 if native else index + 1, np.float32)
        row = dict(
            surface=save(name + "-surface.npy", values),
            surface_persistence=save(name + "-surface-persistence.npy", values * 0),
        )
        if name != "diffusion":
            example["models"][name] = save(
                name + "-interior.npy",
                np.full(
                    (depth_count, 4, y, x),
                    index + 10 if native else index + 1,
                    np.float32,
                ),
            )
            row["heat"] = save(
                name + "-heat.npy", np.full((12, 2, 2, 4), index + 2, np.float32)
            )
            row["heat_persistence"] = row["heat"]
        if native:
            row["forecast_interior"] = save(
                name + "-forecast.npy",
                np.arange(6, dtype=np.float32)[:, None, None, None, None]
                + np.full((6, depth_count, 4, y, x), index + 20, np.float32),
            )
        elif name == "obs08000":
            row["temperature"] = save(
                "temperature.npy", np.full((12, depth_count, y, x), 7, np.float32)
            )
            row["temperature_persistence"] = row["temperature"]
            record["temperature_reference"] = row["temperature"]
            example["references"]["observations"] = example["models"][name]
            example["references"]["climatology"] = example["models"][name]
        else:
            meta["models"][name]["description"] = "Diffusion saved surface only"
        record["models"][name] = row
    meta["records"]["2015-01-01"] = record
    meta["interior_examples"]["2015-01-01"] = example
    if native:
        meta.update(
            heat_lat=[-45, 45],
            heat_lon=[45, 135, 225, 315],
            forecast_interior_leads_days=[5, 15, 30, 90, 180, 365],
        )
    (root / "catalog.json").write_text(json.dumps(meta))
    return meta


@pytest.fixture
def registry(tmp_path):
    root, native = tmp_path / "base", tmp_path / "native"
    meta = write_catalog(root)
    write_catalog(native, native=True)
    meta["extra_catalogs"] = ["../native"]
    meta["extra_catalog_hashes"] = {
        "../native": hashlib.sha256((native / "catalog.json").read_bytes()).hexdigest()
    }
    (root / "catalog.json").write_text(json.dumps(meta))
    return Catalog(root)


def test_registry_preserves_legacy_arrays_and_native_geometry(registry):
    assert len(registry.all_models) == 4
    legacy = registry.for_model("obs08000")
    native = registry.for_model("quarter-uglobal-selected")
    assert legacy.values("surface", "obs08000", "2015-01-01")[0].shape == (73, 2, 2, 4)
    assert native.values("surface", "quarter-uglobal-selected", "2015-01-01")[
        0
    ].shape == (73, 2, 4, 8)
    assert tuple(map(len, native.geometry("heat"))) == (2, 4)
    assert tuple(map(len, native.geometry("surface"))) == (4, 8)
    assert native.available_modes("quarter-uglobal-selected") == {
        "surface",
        "interior",
        "forecast_interior",
        "heat",
    }
    assert legacy.available_modes("diffusion") == {"surface"}
    assert native.depth_interfaces[-1] == 6500
    assert native.models_for("surface", "2015-01-01") == {
        "quarter-uglobal-selected",
        "quarter-obs-only-selected",
    }


def test_native_persistence_uses_exact_initializer_at_every_saved_lead(registry):
    native = registry.for_model("quarter-uglobal-selected")
    pred, reference = native.values(
        "forecast_interior", "quarter-uglobal-selected", "2015-01-01", "persistence"
    )
    assert pred.shape == reference.shape == (6, 19, 4, 4, 8)
    assert np.all(reference == 10)
    assert np.all(pred[-1] == 25)
    assert not reference.flags.writeable
    with pytest.raises(KeyError):
        native.values("temperature", "quarter-uglobal-selected", "2015-01-01")
    with pytest.raises(ValueError, match="same catalog geometry"):
        registry.values("surface", "quarter-uglobal-selected", "2015-01-01", "obs08000")


def test_registry_refuses_changed_child_metadata(registry):
    path = registry.children[0].root / "catalog.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        Catalog(registry.root)


def test_row_strips_preserve_every_cell_irregular_latitude_and_missing_masks():
    lat, lon = np.array([-80, -20, 35, 83]), np.array([45, 135, 225, 315])
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    wet = np.ones((4, 4), bool)
    wet[0, 0] = False
    values[1, 2] = np.nan
    result = raster_rows(values, wet, lat, lon, ["#000000", "#ff0000"], 0, 15)
    np.testing.assert_array_equal(np.concatenate(result["values"]), values)
    np.testing.assert_allclose(result["y"], edges(lat, (-90, 90))[:-1])
    np.testing.assert_allclose(result["dh"], np.diff(edges(lat, (-90, 90))))
    pixels = np.concatenate(result["image"]).view(np.uint8).reshape(4, 4, 4)
    np.testing.assert_array_equal(pixels[0, 0], [213, 219, 224, 255])
    np.testing.assert_array_equal(pixels[1, 2], [255, 255, 255, 255])
    np.testing.assert_array_equal(pixels[-1, -1], [255, 0, 0, 255])
    with pytest.raises(ValueError, match="uniform longitude"):
        raster_rows(values, wet, lat, [45, 120, 225, 315], ["#000000"], 0, 15)


def test_actual_widget_callbacks_route_views_without_changing_selected_model(
    registry, monkeypatch
):
    monkeypatch.setenv("SAMUDRA_VIEWER_DATA", str(registry.root))
    monkeypatch.delenv("SAMUDRA_VIEWER_CATALOGS", raising=False)
    path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("viewer_multires_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    viewer = module.viewer
    viewer.mode.value = "heat"
    assert viewer.monthly_profiles
    viewer.model.value = "quarter-uglobal-selected"
    assert viewer.mode.value == "heat"
    assert not viewer.monthly_profiles
    assert "not saved" in viewer.context.object
    assert len(viewer.lat) == 2
    assert set(viewer.reference.options.values()) == {
        "observations",
        "persistence",
        "quarter-uglobal-selected",
        "quarter-obs-only-selected",
    }
    viewer.mode.value = "forecast_interior"
    assert viewer.saved_lead.options == [5, 15, 30, 90, 180, 365]
    assert len(viewer.lat) == 4
    viewer.saved_lead.value = 365
    viewer.reference.value = "persistence"
    assert np.all(viewer.prediction == 25)
    assert np.all(viewer.comparison == 10)
    assert len(viewer.profile_source.data["depth"]) == 19
    assert list(viewer.line_source.data["x"]) == [5, 15, 30, 90, 180, 365]
    viewer.mode.value = "interior"
    viewer.variable.value = 3
    assert len(viewer.profile_source.data["prediction"]) == 19
    viewer.model.value = "obs08000"
    assert viewer.mode.value == "interior"
    assert len(viewer.catalog.depths) == 14
    viewer.mode.value = "heat"
    assert viewer.monthly_profiles
    viewer.model.value = "diffusion"
    assert viewer.model.value == "diffusion"
    assert viewer.mode.value == "surface"
    assert "not saved" in viewer.context.object
    assert "forecast_interior" not in viewer.mode.options.values()
    viewer.model.value = "obs08000"
    assert viewer.prediction.shape == (73, 2, 2, 4)
