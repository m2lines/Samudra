# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Viewer conversion preserves trajectories, lineage, masks and source distinctions."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_stochastic_viewer",
    Path(__file__).parents[1] / "scripts/prepare_stochastic_viewer.py",
)
assert spec is not None and spec.loader is not None
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def make_export(root, learned=False, om4=False):
    root.mkdir()
    names = np.array(
        [f"{v}_{d}" for v in ["uo", "vo", "thetao", "so"] for d in range(19)] + ["zos"]
    )
    mask = np.ones((77, 2, 3), bool)
    mask[:, 0, 0] = False
    common = dict(
        channel_names=names,
        mask=mask,
        lat=np.array([-30.0, 30.0]),
        lon=np.array([0.0, 120.0, 240.0]),
        depth=np.arange(19) + 0.5,
        depth_interfaces=np.arange(20),
        units=np.array(["m/s"] * 38 + ["degree_C"] * 19 + ["salinity"] * 19 + ["m"]),
    )
    receipt: dict[str, Any] = dict(
        complete=True,
        arm="B" if learned else "C",
        weights="ema",
        step=2000 if learned else 5000,
        checkpoint="test-checkpoint",
        checkpoint_sha256="test-hash",
        origins={},
    )
    if learned:
        receipt.update(
            variant="B-noise-processing",
            counts=dict(om4=1000, observation=1000),
            contract=dict(noise_processing_blocks=3, noise_mode="iid"),
        )
    for month in ["2013-11", "2014-01", "2014-07"]:
        start: np.ndarray = (
            np.datetime64(month + "-01")
            - np.timedelta64(5, "D")
            + np.arange(7) * np.timedelta64(5, "D")
        )
        metadata = dict(
            common,
            starts=start.astype(str),
            ends=(start + np.timedelta64(5, "D")).astype(str),
            midpoints=(start.astype("datetime64[h]") + np.timedelta64(60, "h")).astype(
                str
            ),
            leads=np.arange(0, 31, 5),
            surface_reference=np.ones((7, 2, 2, 3), np.float32),
        )
        files = {}
        for readout, offset in [("mean8", 0), ("draw1", 1000), ("draw2", 2000)]:
            fields = np.broadcast_to(
                np.arange(77)[None, :, None, None]
                + np.arange(7)[:, None, None, None] * 100
                + offset,
                (7, 77, 2, 3),
            ).astype(np.float32)
            path = root / f"{month}-{readout}.npz"
            np.savez(path, **metadata, fields=fields)
            files[readout] = dict(sha256=viewer.sha(path))
        receipt["origins"][month] = dict(seed=1729, files=files, om4=None)
    if om4:
        folder = root / "om4-initialization"
        folder.mkdir()
        entries = {}
        for month in ["2014-01", "2014-04", "2014-05"]:
            end = np.asarray([month + "-01"], dtype="datetime64[D]")
            start = end - np.timedelta64(5, "D")
            metadata = dict(
                common,
                starts=start.astype(str),
                ends=end.astype(str),
                midpoints=(
                    start.astype("datetime64[h]") + np.timedelta64(60, "h")
                ).astype(str),
                leads=np.array([0]),
            )
            files = {}
            for readout, offset in [
                ("mean8", 0),
                ("draw1", 1000),
                ("draw2", 2000),
                ("gold", 3000),
            ]:
                path = folder / f"{month}-{readout}.npz"
                np.savez(
                    path, **metadata, fields=np.full((1, 77, 2, 3), offset, np.float32)
                )
                files[readout] = dict(sha256=viewer.sha(path))
            entries[month] = dict(
                files=files, weights="ema", checkpoint_sha256="test-hash", members=8
            )
        receipt["om4_initializations"] = entries
        (folder / "COMPLETE.json").write_text(
            json.dumps(dict(complete=True, origins=entries))
        )
    (root / "COMPLETE.json").write_text(json.dumps(receipt))
    return receipt


def test_existing_c_default_ids_and_lineage_preserved(tmp_path):
    source = tmp_path / "export"
    make_export(source)
    result = viewer.prepare(source, tmp_path / "catalog", viewer.REPORT)
    assert set(result["models"]) == {
        "stochastic-C-final-" + r for r in ["mean8", "draw1", "draw2"]
    }
    for info in result["models"].values():
        assert info["lineage"]["global_step"] == 5000
        assert info["lineage"]["task_counts"] == dict(om4=2000, observation=3000)
        assert "Both initializer and processor are stochastic" in info["description"]


def test_learned_noise_readouts_temporal_slices_masks_and_descriptions(tmp_path):
    source = tmp_path / "export"
    make_export(source, learned=True)
    output = tmp_path / "catalog"
    result = viewer.prepare(source, output, profile="learned-noise-2k")
    prefix = "stochastic-B-noise-processing-2k-"
    assert set(result["models"]) == {prefix + r for r in ["mean8", "draw1", "draw2"]}
    assert result["report_url"] == viewer.NOISE_REPORT
    record = result["records"]["2014-01-01"]
    for readout, offset in [("mean8", 0), ("draw1", 1000), ("draw2", 2000)]:
        model = record["models"][prefix + readout]
        surface = np.load(output / model["surface"])
        interior = np.load(output / model["forecast_interior"])
        assert surface.shape == (6, 2, 2, 3) and interior.shape == (6, 19, 4, 2, 3)
        assert (
            surface[0, 0, 1, 1] == 138 + offset and surface[-1, 1, 1, 1] == 676 + offset
        )
        assert interior[0, 0, 2, 1, 1] == 100 + offset
        assert np.isnan(surface[:, :, 0, 0]).all()
        info = result["models"][prefix + readout]
        assert info["lineage"]["global_step"] == 2000
        assert info["lineage"]["task_counts"] == dict(om4=1000, observation=1000)
        assert (
            "deterministic residual" in info["description"]
            and "No cooldown or replay" in info["description"]
        )
        assert "8→32→3" in info["description"]
    assert len(result["interior_examples"]["2014-01-01"]["description"]) < 180
    assert all(
        viewer.sha(output / name) == info["sha256"]
        for name, info in result["files"].items()
    )


def test_om4_initializers_have_gold_and_no_invented_forecasts(tmp_path):
    source = tmp_path / "export"
    make_export(source, learned=True, om4=True)
    output = tmp_path / "catalog"
    result = viewer.prepare(source, output, profile="learned-noise-2k")
    assert len(result["origins"]) == 6
    key = "2014-04-01-om4"
    r = result["interior_examples"][key]
    assert r["source"] == "om4" and r["references"].keys() == {"om4"}
    assert np.nanmean(np.load(output / r["references"]["om4"])) == 3000
    assert all(
        set(x) == {"interior"} for x in result["records"][key]["models"].values()
    )
    assert "surface_reference" not in result["records"][key]


@pytest.mark.parametrize(
    "change",
    [
        dict(arm="C"),
        dict(step=5000),
        dict(counts=dict(om4=2000, observation=0)),
        dict(contract=dict(noise_mode="multiscale", noise_processing_blocks=3)),
    ],
)
def test_profile_rejects_wrong_checkpoint_or_noise_configuration(tmp_path, change):
    source = tmp_path / "export"
    receipt = make_export(source, learned=True)
    receipt.update(change)
    (source / "COMPLETE.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        viewer.prepare(source, tmp_path / "catalog", profile="learned-noise-2k")
    assert not (tmp_path / "catalog").exists()


def test_rejects_changed_export_checksum(tmp_path):
    source = tmp_path / "export"
    make_export(source, learned=True)
    with (source / "2014-01-draw1.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(AssertionError):
        viewer.prepare(source, tmp_path / "catalog", profile="learned-noise-2k")
