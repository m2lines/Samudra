# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Check annual-model provenance, physical layout and original bundle preservation."""

import json

import numpy as np
import pytest
from export_interior import sha
from extend_annual_models import extend
from model_info import description

from data import Catalog


def fixture(tmp_path):
    base, source = tmp_path / "base", tmp_path / "source"
    base.mkdir()
    source.mkdir()
    names = [f"{v}_{i}" for v in ["uo", "vo", "thetao", "so"] for i in range(19)] + [
        "zos"
    ]
    shape = (2, 3)
    depths = np.arange(19, dtype=float)
    files = {}

    def saved(name, values):
        np.save(base / name, values)
        files[name] = dict(
            sha256=sha(base / name), shape=list(values.shape), dtype=str(values.dtype)
        )
        return name

    sm = saved("surface-mask.npy", np.ones((2, *shape), bool))
    im = saved("interior-mask.npy", np.ones((14, 4, *shape), bool))
    hm = saved("heat-mask.npy", np.ones((2, *shape), bool))
    sr = saved("surface-reference.npy", np.ones((73, 2, *shape), np.float32))
    raw_heat = np.full((12, 2, *shape), 31_234_567_890, np.float32)
    hr = saved("heat-reference.npy", raw_heat.astype(np.float64) / 1e9)
    old = saved("original.npy", np.ones((73, 2, *shape), np.float32) * 5)
    origin = "2015-01-01"
    meta = dict(
        schema_version=1,
        lat=[-30, 30],
        lon=[0, 120, 240],
        depths=depths[:14].tolist(),
        files=files,
        models={"old": {}},
        origins=[origin],
        surface_mask=sm,
        interior_mask=im,
        heat_mask=hm,
        interior_examples={origin: dict(models={}, references={})},
        records={
            origin: dict(
                models={"old": dict(surface=old)},
                heat_months=[f"2015-{i:02d}" for i in range(1, 13)],
                surface_reference=sr,
                heat_reference=hr,
            )
        },
    )
    (base / "catalog.json").write_text(json.dumps(meta))
    root = source / "remote"
    (root / "U-global").mkdir(parents=True)
    np.savez(
        root / "grid.npz",
        lat=meta["lat"],
        lon=meta["lon"],
        depth=depths,
        names=names,
        mask=np.ones((77, *shape), bool),
    )
    (root / "manifest.json").write_text(json.dumps(dict(code_commit="producer")))
    (root / "best.json").write_text(
        json.dumps(
            dict(
                checkpoint_sha256="weights",
                global_step=3800,
                task_counts={"om4": 1972, "observation": 1828},
            )
        )
    )
    (root / "U-global/COMPLETE.json").write_text(
        json.dumps(dict(inputs=dict(checkpoint_sha256="weights")))
    )
    initial = np.broadcast_to(
        np.arange(77, dtype=np.float32)[None, :, None, None], (2, 77, *shape)
    )
    np.savez(
        root / "annual.npz",
        surface=np.ones((73, 2, *shape), np.float32) * 7,
        persistence_surface=np.ones((73, 2, *shape), np.float32) * 2,
        initial=initial,
        months=meta["records"][origin]["heat_months"],
        reference=np.ones((73, 4, *shape), np.float32),
        predicted_ohc=np.ones((12, 2, *shape), np.float64) * 3e9,
        reference_ohc=raw_heat,
        persistence_ohc=np.ones((2, *shape), np.float64) * 4e9,
    )
    inv = dict(
        models={
            "U-global": dict(
                root="/remote",
                grid="/remote/grid.npz",
                manifest="/remote/manifest.json",
                best="/remote/best.json",
                selected=dict(checkpoint_sha256="weights"),
                origins={origin: "/remote/annual.npz"},
            )
        },
        files={},
    )
    for p in root.rglob("*"):
        if p.is_file():
            inv["files"]["/" + str(p.relative_to(source))] = dict(
                sha256=sha(p), bytes=p.stat().st_size
            )
    (source / "inventory.json").write_text(json.dumps(inv))
    return base, source


def test_extension_preserves_old_data_maps_channels_and_units(tmp_path):
    base, source = fixture(tmp_path)
    before = sha(base / "catalog.json")
    output = tmp_path / "out"
    extend(base, source, output)
    c = Catalog(output)
    assert sha(base / "catalog.json") == before
    assert sha(output / "original.npy") == sha(base / "original.npy")
    pred, ref = c.values("surface", "U-global", "2015-01-01", "persistence")
    assert np.all(pred == 7) and np.all(ref == 2)
    heat, persist = c.values("report_heat", "U-global", "2015-01-01", "persistence")
    assert np.all(heat == 3) and np.all(persist == 4)
    interior = c.array(c.meta["interior_examples"]["2015-01-01"]["models"]["U-global"])
    np.testing.assert_array_equal(interior[0, :, 0, 0], [38, 57, 0, 19])
    assert "temperature" not in c.meta["records"]["2015-01-01"]["models"]["U-global"]
    assert "old" in c.meta["models"]
    with pytest.raises(ValueError, match="new bundle"):
        extend(base, source, output)


def test_extension_rejects_changed_source_before_creating_bundle(tmp_path):
    base, source = fixture(tmp_path)
    (source / "remote/manifest.json").write_text("{}")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="checksum"):
        extend(base, source, output)
    assert not output.exists()


def test_descriptions_separate_model_checkpoint_and_forecast():
    info = dict(
        label="A-global",
        lineage=dict(global_step=4000, task_counts={"om4": 2000, "observation": 2000}),
    )
    text = description("A-global", info)
    assert "axial attention" in text and "64.2M" in text
    assert "4,000" in text and "73 five-day" in text
    assert "Constant learning rate, no cooldown" in text
    old = description(
        "scratch16000",
        dict(
            label="Obs only",
            lineage=dict(
                global_step=16000, task_counts={"om4": 0, "observation": 16000}
            ),
        ),
    )
    assert "exclusively on observations" in old and "Fixed-exposure" in old
