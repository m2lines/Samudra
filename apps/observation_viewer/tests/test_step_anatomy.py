# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from step_anatomy import Anatomy, load


@pytest.fixture
def anatomy(tmp_path):
    names = [f"{v}_{i}" for v in ["uo", "vo", "thetao", "so"] for i in range(19)] + [
        "zos"
    ]
    np.savez(
        tmp_path / "grid.npz", names=names, mean=np.arange(77), std=np.full(77, 2.0)
    )
    cases = {}
    for source, offset in [("om4", 0.0), ("obs_pre", 1.0), ("obs", 2.0)]:
        root = tmp_path / source
        root.mkdir()
        cases[source] = dict(
            step=2000 if source != "obs" else 3000, checkpoint_sha256="test"
        )
        mask = np.zeros((19, 77, 4, 8), bool)
        mask[:, [38, 76]] = True
        np.savez(
            root / "inputs.npz",
            fields=np.full(mask.shape, offset),
            available=mask,
            forcing=np.zeros((26, 3, 4, 8)),
            contexts=np.zeros((26, 5, 4, 8)),
            latents=np.zeros((8, 2, 128, 1, 2)),
            forcing_mean=np.zeros(3),
            forcing_std=np.ones(3),
        )
        np.savez(
            root / "atmosphere.npz",
            normalized=np.zeros((26, 8, 4, 8)),
            mean=np.zeros(8),
            std=np.ones(8),
        )
        for stage in ["initial", "day5", "monthly", "initial-full", "day5-full"]:
            d = root / stage
            d.mkdir()
            support = np.ones((77, 4, 8), bool)
            if source != "om4" and stage in ["initial", "day5"]:
                support[:] = False
                support[[38, 76]] = True
            np.savez(
                d / "final.npz",
                prediction=np.full(support.shape, offset),
                gold=np.zeros(support.shape),
                support=support,
            )
            for call in [0, 1, 62]:
                np.savez(
                    d / f"call-{call:02d}.npz",
                    clean=np.full(support.shape, offset + call),
                    noisy=np.full(support.shape, offset + 80),
                    sigma=80.0,
                )
            (d / "trace.json").write_text(
                json.dumps(dict(calls=[dict(role="predictor", sigma=80)] * 63))
            )
    (tmp_path / "COMPLETE.json").write_text(json.dumps(dict(cases=cases)))
    yield Anatomy(tmp_path)
    load.cache_clear()


def test_shared_scales_and_missing_observation_outputs(anatomy):
    order = [i for i, _ in anatomy.panels]
    scales = anatomy.scales.copy()
    anatomy.source.value = "obs_pre"
    assert anatomy.scales == scales
    assert [i for i, _ in anatomy.panels] == order
    assert not anatomy.current[2][0].any()
    assert "Not requested" in anatomy.render(0)
    anatomy.source.value = "obs"
    assert anatomy.scales == scales
    anatomy.units.value = "Normalized"
    assert anatomy.transform(3.0, 38) == 3.0
    anatomy.units.value = "Physical"
    assert anatomy.transform(3.0, 38) == 44.0
    assert anatomy.transform(3.0, 38, difference=True) == 6.0


def test_trace_and_context_are_explicit(anatomy):
    anatomy.source.value = "obs_pre"
    anatomy.stage.value = "initial-full"
    assert "not OBS truth" in anatomy.render(0)
    anatomy.readout.value = "Diffusion step: input + prediction"
    anatomy.step.value = 1
    assert anatomy.current[0][0, 0, 0] == 1
    assert "Noisy input" in anatomy.render(0)
    assert "Denoiser prediction" in anatomy.render(0)
    anatomy.corrector.value = True
    assert anatomy.current[0][0, 0, 0] == 2
    anatomy.stage.value = "inputs"
    assert not anatomy.current[2][0].any()
    assert anatomy.current[2][38].all()


def test_nn_input_scaling_uses_normalized_values(anatomy, monkeypatch):
    import step_anatomy

    anatomy.source.value = "obs_pre"
    anatomy.readout.value = "Diffusion step: input + prediction"
    anatomy.step.value = 1
    images = []

    def capture(values, *a, **kw):
        images.append(np.array(values))
        return "image"

    monkeypatch.setattr(step_anatomy, "image_html", capture)
    anatomy.render(38)
    np.testing.assert_allclose(images[0], 81 / np.sqrt(6401))
    anatomy.nn_input.value = False
    images.clear()
    anatomy.render(38)
    np.testing.assert_allclose(images[0], 81 * 2 + 38)


def test_field_groups_follow_support_and_preserve_selection(anatomy):
    anatomy.field.value = 0
    anatomy.source.value = "obs_pre"
    assert list(anatomy.field.groups["Available"].values()) == [38, 76]
    assert 0 in anatomy.field.groups["Not present in this view"].values()
    assert anatomy.field.value == 0
    anatomy.stage.value = "initial-full"
    assert list(anatomy.field.groups) == ["Available"]
    assert len(anatomy.field.groups["Available"]) == 77


def test_denoiser_rmse_and_sparkline(tmp_path):
    from step_anatomy import denoiser_rmse, rmse_sparkline

    gold = np.zeros((2, 2, 2))
    support = np.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
    np.savez(tmp_path / "final.npz", gold=gold, support=support)
    clean = np.array([[[3, 4], [999, 999]], [[999, 999], [999, 999]]])
    np.savez(tmp_path / "call-00.npz", clean=clean)
    np.savez(tmp_path / "call-01.npz", clean=clean * 2)
    calls, errors = denoiser_rmse(str(tmp_path))
    np.testing.assert_allclose(errors[:, 0], [np.sqrt(12.5), np.sqrt(50)])
    assert np.isnan(errors[:, 1]).all()
    svg = rmse_sparkline(calls, errors[:, 0], 1, "m/s")
    assert "circle" in svg and "call 1" in svg and "m/s" in svg
    assert rmse_sparkline(calls, errors[:, 1], 1, "m/s") == ""


def test_model_switch_and_shared_era5_om4(anatomy, tmp_path):
    import shutil

    shared = tmp_path / "shared"
    shared.mkdir()
    shutil.copy(anatomy.root / "grid.npz", shared / "grid.npz")
    cases = {}
    for key, original in [
        ("om4", "om4"),
        ("obs_pre", "obs_pre"),
        ("om4_post", "om4"),
        ("obs", "obs"),
    ]:
        shutil.copytree(anatomy.root / original, shared / key)
        cases[key] = dict(
            source="om4" if key.startswith("om4") else "obs",
            label=key + " shared",
            forcing_kind="shared_era5",
            step=5000,
            checkpoint_sha256="shared-test",
        )
    (shared / "COMPLETE.json").write_text(
        json.dumps(dict(cases=cases, model_description="ERA5 for both sources."))
    )
    catalog = tmp_path / "models.json"
    catalog.write_text(
        json.dumps(
            {"models": {"shared-era5": {"label": "Shared ERA5", "path": "shared"}}}
        )
    )
    viewer = Anatomy(anatomy.root, catalog=catalog)
    viewer.field.value = 76
    viewer.stage.value = "day5-full"
    viewer.model.value = "shared-era5"
    assert viewer.field.value == 76
    assert viewer.stage_for("om4_post") == "day5"
    assert "ERA5 for both sources." in viewer.caption.object
    assert "Eight ERA5 input channels" in viewer.auxiliary.object
    assert "tauuo (N/m²)" not in viewer.auxiliary.object
    viewer.source.value = "om4_post"
    assert viewer.current[2][0].all()
    viewer.model.value = "original-a"
    assert viewer.source.value == "om4"
    assert "tauuo (N/m²)" in viewer.auxiliary.object
    restored = Anatomy(anatomy.root, catalog=catalog, model="shared-era5")
    assert restored.model.value == "shared-era5"
