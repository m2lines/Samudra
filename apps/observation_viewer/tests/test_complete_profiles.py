# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify calendar, references, heat/profile consistency and bundle preservation."""

import json

import numpy as np
import pytest
from complete_monthly_profiles import complete
from export_interior import sha
from extend_monthly_temperature import integrated_heat

from data import Catalog


def fixture(tmp_path, bad_heat=False):
    base, export = tmp_path / "base", tmp_path / "export"
    base.mkdir()
    export.mkdir()
    origin = "2015-01-01"
    shape = (12, 14, 2, 3)
    files = {}

    def save(name, a):
        name += ".npy"
        np.save(base / name, a)
        files[name] = dict(
            shape=list(a.shape),
            dtype=str(a.dtype),
            sha256=sha(base / name),
            bytes=(base / name).stat().st_size,
        )
        return name

    mask = np.ones((2, 2, 3), bool)
    im = save("interior-mask", np.ones((14, 4, 2, 3), bool))
    hm = save("heat-mask", mask)
    temp = np.full(shape, 10, np.float32)
    ref = save("reference", np.full(shape, 30, np.float32))
    oldtemp = save("old-t", temp)
    heat = save("old-heat", integrated_heat(temp, mask))
    initial = save("initial", np.full((14, 4, 2, 3), 10, np.float32))
    months = [f"2015-{i:02d}" for i in range(1, 13)]
    m = dict(
        schema_version=1,
        models={
            "old": {"lineage": {"checkpoint_sha256": "old"}},
            "new": {"lineage": {"checkpoint_sha256": "new"}},
        },
        origins=[origin],
        lat=[-30, 30],
        lon=[0, 120, 240],
        depths=list(range(14)),
        files=files,
        interior_mask=im,
        heat_mask=hm,
        monthly_temperature=dict(report_comparison={}),
        records={
            origin: dict(
                heat_months=months,
                temperature_reference=ref,
                models={
                    "old": dict(temperature=oldtemp, heat=heat, interior=initial),
                    "new": dict(heat=heat, interior=initial),
                },
            )
        },
    )
    (base / "catalog.json").write_text(json.dumps(m))
    (export / "export_annual_profiles.py").write_text("# exporter")
    (export / "runtime-contract.json").write_text("{}")
    config = dict(
        base_catalog_sha256=sha(base / "catalog.json"), models=[dict(name="new")]
    )
    (export / "paths.json").write_text(json.dumps(config))
    signature = dict(
        config_sha256=sha(export / "paths.json"),
        script_sha256=sha(export / "export_annual_profiles.py"),
    )
    (export / "audit-output").mkdir()
    (export / "audit-output/AUDITED.json").write_text(json.dumps(signature))
    dest = export / "exports/new"
    dest.mkdir(parents=True)
    new = np.full(shape, 20, np.float32)
    held = np.full(shape[1:], 15, np.float32)
    np.savez(
        dest / (origin + ".npz"),
        temperature=new,
        reference=np.full(shape, 30, np.float32),
        heat=integrated_heat(new, mask) + (1 if bad_heat else 0),
        temperature_persistence=held,
        heat_persistence=integrated_heat(held[None], mask)[0],
        months=np.array(months),
    )
    file = dest / (origin + ".npz")
    receipt = dict(
        **signature,
        model="new",
        lineage=m["models"]["new"]["lineage"],
        files={file.name: dict(sha256=sha(file), bytes=file.stat().st_size)},
        cases={
            origin: dict(heat_difference_from_report_gj_m2={"rms": 0, "max_abs": 0})
        },
    )
    (dest / "COMPLETE.json").write_text(json.dumps(receipt))
    return base, export, m


def test_complete_profiles_retains_originals_and_adds_consistent_persistence(tmp_path):
    base, export, meta = fixture(tmp_path)
    output = tmp_path / "complete"
    complete(base, export, output)
    c = Catalog(output)
    record = c.meta["records"]["2015-01-01"]
    for model, value in [("old", 10), ("new", 20)]:
        temperature, persist = c.values(
            "temperature", model, "2015-01-01", "persistence"
        )
        assert np.all(temperature == value)
        heat, held_heat = c.values("heat", model, "2015-01-01", "persistence")
        mask = c.array(c.meta["heat_mask"])
        np.testing.assert_array_equal(integrated_heat(temperature, mask), heat)
        np.testing.assert_array_equal(integrated_heat(persist, mask), held_heat)
    for name, info in meta["files"].items():
        assert sha(output / name) == info["sha256"]
    assert (
        record["models"]["new"]["report_heat"]
        == meta["records"]["2015-01-01"]["models"]["new"]["heat"]
    )
    assert c.meta["monthly_temperature"]["complete_models"] == ["old", "new"]


def test_profile_heat_disagreement_is_rejected(tmp_path):
    base, export, _ = fixture(tmp_path, bad_heat=True)
    with pytest.raises(AssertionError):
        complete(base, export, tmp_path / "bad")
