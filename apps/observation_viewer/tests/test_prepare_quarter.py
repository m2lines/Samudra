# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify physical axis order, sparse lead availability, native geometry and gates."""

import json
from pathlib import Path

import numpy as np
import pytest
from prepare_quarter import LEADS, ORIGINS, Writer, interior_indices, prepare, sha


def fixture_config(root):
    obs = root / "observations"
    obs.mkdir()
    annual = root / "annual"
    annual.mkdir()
    names = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + [
        "zos"
    ]
    mask = np.ones((77, 2, 3), bool)
    mask[:, 0, 0] = False
    mask[1::19, 1, 2] = False
    np.savez(
        obs / "grid.npz",
        names=names,
        lat=[-20, 20],
        lon=[0, 120, 240],
        mask=mask,
        depth=np.arange(19) + 0.5,
        depth_thickness=np.ones(19),
    )
    np.savez(obs / "iap-grid.npz", lat=[0.0], lon=[60.0, 180.0])
    manifests = {}
    for origin in ORIGINS:
        p = annual / (origin + ".json")
        p.write_text("{}\n")
        manifests[p.name] = sha(p)
    config = dict(
        origins=ORIGINS,
        observations=str(obs),
        annual_data=str(annual),
        grid_sha256=sha(obs / "grid.npz"),
        iap_grid_sha256=sha(obs / "iap-grid.npz"),
        native_shape=[2, 3],
        heat_shape=[1, 2],
        producer="eval-pin",
        sample_manifest_sha256="data-pin",
        models=[],
    )
    for mi, key in enumerate(["quarter-uglobal-selected", "quarter-obs-only-selected"]):
        d = root / key
        d.mkdir()
        training = d / "manifest.json"
        training.write_text(json.dumps(dict(code_commit="training-pin")))
        outputs = {}
        for oi, origin in enumerate(ORIGINS):
            initial = (
                np.broadcast_to(
                    np.arange(2 * 77, dtype=np.float32).reshape(2, 77, 1, 1),
                    (2, 77, 2, 3),
                ).copy()
                + mi * 1000
                + oi * 100
            )
            state = (
                np.broadcast_to(
                    np.arange(6 * 77, dtype=np.float32).reshape(6, 77, 1, 1),
                    (6, 77, 2, 3),
                ).copy()
                + mi * 1000
                + oi * 100
            )
            surface = (
                np.broadcast_to(
                    np.arange(73, dtype=np.float32).reshape(73, 1, 1, 1), (73, 2, 2, 3)
                ).copy()
                + mi * 100
            )
            for si, day in enumerate(LEADS):
                surface[day // 5 - 1] = state[si, [38, 76]]
            heat = np.ones((12, 2, 1, 2), np.float32) * (mi + 1) * 1e9
            heat[:, :, :, 0] = np.nan
            reference = np.ones((73, 4, 2, 3), np.float32) * oi
            reference[:, :, 0, 1] = np.nan
            path = d / (origin + ".npz")
            np.savez(
                path,
                surface=surface,
                reference=reference,
                initial=initial,
                state_at_leads=state,
                leads_days=LEADS,
                months=[f"{origin[:4]}-{m:02}" for m in range(1, 13)],
                predicted_ohc=heat,
                reference_ohc=heat / (mi + 1),
                persistence_ohc=heat[0],
                persistence_surface=np.broadcast_to(
                    initial[-1, [38, 76]], (73, 2, 2, 3)
                ),
                iap_lat=[0.0],
                iap_lon=[60.0, 180.0],
            )
            outputs[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        complete = dict(
            origins={o: {} for o in ORIGINS},
            inputs=dict(
                training_manifest_sha256=sha(training),
                checkpoint_sha256=key,
                include_persistence=True,
                global_observations=True,
                sample_manifest_sha256="data-pin",
                data_manifests=manifests,
            ),
        )
        cp = d / "COMPLETE.json"
        cp.write_text(json.dumps(complete))
        rp = Path(str(d) + "-RESULT.json")
        rp.write_text(
            json.dumps(
                dict(
                    passed=True,
                    evaluation_producer="eval-pin",
                    complete_sha256=sha(cp),
                    outputs=outputs,
                )
            )
        )
        config["models"].append(
            dict(
                key=key,
                label=key,
                description="fixture",
                evaluation=str(d),
                result_sha256=sha(rp),
                checkpoint_sha256=key,
                training_manifest=str(training),
            )
        )
    p = root / "config.json"
    p.write_text(json.dumps(config))
    return p


def test_full_native_bundle_preserves_saved_times_depths_and_units(tmp_path):
    config = fixture_config(tmp_path)
    out = tmp_path / "bundle"
    meta = prepare(config, out)
    assert "monthly_temperature" not in meta
    assert meta["forecast_interior_leads_days"] == [5, 15, 30, 90, 180, 365]
    assert len(meta["depths"]) == 19 and len(meta["depth_interfaces"]) == 20
    assert meta["heat_lat"] == [0.0] and meta["lat"] == [-20, 20]
    origin = ORIGINS[0]
    key = "quarter-uglobal-selected"
    record = meta["records"][origin]
    row = record["models"][key]
    initial = np.load(out / row["interior"])
    future = np.load(out / row["forecast_interior"])
    assert initial.shape == (19, 4, 2, 3) and future.shape == (6, 19, 4, 2, 3)
    np.testing.assert_array_equal(
        initial[3, :, 1, 1], np.array([38 + 3, 57 + 3, 3, 19 + 3]) + 77
    )
    np.testing.assert_array_equal(
        future[5, 3, :, 1, 1], np.array([38 + 3, 57 + 3, 3, 19 + 3]) + 5 * 77
    )
    assert np.isnan(initial[:, :, 0, 0]).all()
    assert np.isnan(future[:, :, :, 0, 0]).all()
    assert np.isnan(np.load(out / record["surface_reference"])[:, :, 0, 1]).all()
    np.testing.assert_array_equal(np.load(out / row["heat"])[:, :, :, 1], 1)
    assert meta["interior_examples"][origin]["references"] == {}
    assert meta["interior_examples"][origin]["models"][key] == row["interior"]
    for name, info in meta["files"].items():
        assert sha(out / name) == info["sha256"]
    assert json.loads((out / "COMPLETE.json").read_text())["catalog_sha256"] == sha(
        out / "catalog.json"
    )
    with pytest.raises(ValueError, match="immutable"):
        prepare(config, out)


def test_source_mutation_fails_before_publishing_bundle(tmp_path):
    config = fixture_config(tmp_path)
    path = tmp_path / "quarter-uglobal-selected" / f"{ORIGINS[0]}.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="byte count"):
        prepare(config, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_incomplete_cohort_rejected_even_with_valid_receipt_hash(tmp_path):
    config_path = fixture_config(tmp_path)
    c = json.loads(config_path.read_text())
    m = c["models"][0]
    d = Path(m["evaluation"])
    p = d / "COMPLETE.json"
    v = json.loads(p.read_text())
    v["origins"].pop(ORIGINS[-1])
    p.write_text(json.dumps(v))
    r = Path(str(d) + "-RESULT.json")
    v = json.loads(r.read_text())
    v["complete_sha256"] = sha(p)
    r.write_text(json.dumps(v))
    m["result_sha256"] = sha(r)
    config_path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="cohort"):
        prepare(config_path, tmp_path / "bundle")


def test_writer_rejects_infinity_and_existing_file(tmp_path):
    w = Writer(tmp_path)
    with pytest.raises(ValueError, match="Infinite"):
        w.write("bad", np.array([[np.inf]]))
    with pytest.raises(ValueError, match="overwrite"):
        w.write("bad", np.array([[1.0]]))


def test_channel_lookup_uses_names_not_assumed_source_order():
    names = [f"{v}_{i}" for v in ("so", "vo", "thetao", "uo") for i in range(19)] + [
        "zos"
    ]
    ids = interior_indices(names)
    assert [names[i] for i in ids[7]] == ["thetao_7", "so_7", "uo_7", "vo_7"]
