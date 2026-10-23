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
    anatomy.readout.value = "Noisy sampler input"
    anatomy.step.value = 1
    assert anatomy.current[0][0, 0, 0] == 81
    anatomy.readout.value = "Denoiser clean estimate"
    anatomy.corrector.value = True
    assert anatomy.current[0][0, 0, 0] == 2
    anatomy.stage.value = "inputs"
    assert not anatomy.current[2][0].any()
    assert anatomy.current[2][38].all()
