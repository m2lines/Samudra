# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Any

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from samudra.skyline.data import CHANNELS, OceanSource, audit_support
from samudra.skyline.model import ModelConfig, PointTransformer, Vocabulary
from samudra.skyline.precision import audit_precision
from samudra.skyline.run import FrameStream, evaluate, resume_checkpoint


def test_preemption_resume_policy(tmp_path):
    latest = tmp_path / "latest.pt"
    config = {"resume": False, "preemptible": True}
    assert not resume_checkpoint(config, latest, 0)
    assert not resume_checkpoint(config, latest, 1)
    latest.write_bytes(b"checkpoint")
    assert resume_checkpoint(config, latest, 1)
    with pytest.raises(FileExistsError):
        resume_checkpoint(config, latest, 0)
    with pytest.raises(FileExistsError):
        resume_checkpoint(dict(config, preemptible=False), latest, 1)
    assert resume_checkpoint(dict(config, resume=True), latest, 0)
    latest.unlink()
    with pytest.raises(FileNotFoundError):
        resume_checkpoint(dict(config, resume=True), latest, 1)


from samudra.skyline.score import score_values


@pytest.fixture
def ocean(tmp_path):
    dates = [
        cftime.DatetimeJulian(y, m, d)
        for y, m, days in [
            (1975, 1, [3, 8, 13]),
            (2013, 10, [5, 10, 15]),
            (2014, 10, [10, 15, 20]),
        ]
        for d in days
    ]
    a = np.arange(9 * 5 * 8, dtype=np.float32).reshape(9, 5, 8) / 100
    fields: dict[str, Any] = {
        n: (("time", "y", "x"), a + i) for i, n in enumerate(CHANNELS)
    }
    wet = np.ones((5, 8), dtype=bool)
    wet[2, 3] = False
    fields.update({f"mask_{z}": (("y", "x"), wet) for z in range(19)})
    fields["areacello"] = (("y", "x"), np.ones((5, 8)))
    xr.Dataset(
        fields,
        coords={"time": dates, "y": np.linspace(-80, 80, 5), "x": np.arange(8) * 45},
    ).to_zarr(tmp_path / "OM4.zarr", consolidated=True)
    for name, value in [("means", 1.0), ("stds", 2.0)]:
        xr.Dataset({n: xr.DataArray(value) for n in CHANNELS}).to_zarr(
            tmp_path / f"OM4_{name}.zarr", consolidated=True
        )
    source = OceanSource(str(tmp_path), 2)
    yield source
    source.close()


def test_split_and_cadence(ocean):
    assert {k: v.tolist() for k, v in ocean.origins.items()} == {
        "train": [1],
        "validation": [4],
        "test": [7],
    }


def test_crop_wrap_padding_masks_and_target_isolation(ocean):
    ocean.locations = [np.array([0])] * 19
    ocean.probabilities = [np.array([1.0])] * 19
    frame = ocean.read(1)
    batch = ocean.sample(frame, np.random.default_rng(2), 2, 3, [0, 18])
    assert batch["values"].shape == (2, 18, 80)
    assert not batch["valid"][:, :3].any()
    assert not batch["valid"][:, 9:12].any()
    assert (batch["values"][:, :3] == 0).all()
    assert batch["values"][0, 3, 0] == frame["input"][0, 0, 0, 7]
    assert batch["current"][0] == frame["input"][1, 38, 0, 0]
    assert batch["target"][0] == frame["target"][0, 0, 0]
    changed = dict(frame, target=frame["target"] + 10)
    other = ocean.sample(changed, np.random.default_rng(2), 2, 3, [0, 18])
    for key in batch:
        if key != "target":
            torch.testing.assert_close(batch[key], other[key])
    torch.testing.assert_close(other["target"], batch["target"] + 10)


def test_histogram_density_and_midpoint_assignment():
    vocab = Vocabulary(np.full(19, 10.0), 17)
    z = torch.tensor([0, 0])
    mid = vocab.edges[0, 5]
    assert vocab.encode(torch.tensor([mid - 1e-4, mid + 1e-4]), z).tolist() == [4, 5]
    depths = torch.zeros(17, dtype=torch.long)
    logits = torch.arange(17, dtype=torch.float32).repeat(17, 1) / 5
    std = torch.full((19,), 3.0)
    density = vocab.log_density(logits, vocab.centers[0], depths, std).exp()
    widths = vocab.edges[0].diff() * std[0]
    torch.testing.assert_close((density * widths).sum(), torch.tensor(1.0))
    outside = vocab.log_density(
        torch.zeros(1, 17), torch.tensor([100.0]), torch.tensor([0]), std
    )
    assert torch.isneginf(outside).all()


@pytest.mark.parametrize("objective", ["absolute_ce", "residual_ce", "regression"])
def test_forward_backward_and_masked_context(ocean, objective):
    torch.set_num_threads(2)
    batch = ocean.sample(ocean.read(1), np.random.default_rng(4), 2, 3)
    model = PointTransformer(
        ModelConfig(width=32, layers=2, heads=4, bins=17, objective=objective)
    )
    logits = model(batch)
    assert logits.shape == (2, 1 if objective == "regression" else 17)
    logits.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    altered = {k: v.clone() for k, v in batch.items()}
    altered["target"] += 1e4
    altered["values"][~batch["valid"]] = 1000
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(altered))


def test_frame_stream_resume():
    class Source:
        origins = {"train": np.array([1, 2, 3, 4])}

        def read(self, t):
            return {"origin": t}

    a = FrameStream(Source(), 42)
    a.advance()
    a.used = 8
    b = FrameStream(Source(), 42, copy.deepcopy(a.state()))
    assert a.state() == b.state()
    for _ in range(12):
        a.advance()
        b.advance()
        assert a.frame == b.frame
        assert a.state() == b.state()
    a.close()
    b.close()


def test_candidate_scores_do_not_require_future_truth(ocean):
    # Last record has no next truth, but still has a valid observed history.
    frame = ocean.read(8, with_target=False)
    assert frame["target"] is None
    batch = ocean.sample(
        frame, np.random.default_rng(0), 2, 3, [0, 18], [(0, 0), (4, 7)]
    )
    assert torch.isnan(batch["target"]).all()
    model = PointTransformer(ModelConfig(width=32, layers=1, heads=4, bins=17)).eval()
    vocab = Vocabulary(np.full(19, 100.0), 17)
    means, stds = torch.zeros(19), torch.ones(19)
    scores = score_values(model, vocab, batch, torch.tensor([10.0, 20.0]), means, stds)
    assert torch.isfinite(scores["log_density_per_c"]).all()
    assert not scores["outside_support"].any()
    with pytest.raises(ValueError, match="five-day"):
        ocean.read(3, with_target=False)
    with pytest.raises(ValueError, match="wet cell"):
        ocean.sample(frame, np.random.default_rng(0), 1, 3, [0], [(2, 3)])


def test_validation_locations_match_between_objectives(ocean, tmp_path):
    config = {
        "validation_dates": 1,
        "validation_seed": 42,
        "validation_points_per_depth": 1,
        "microbatch": 2,
        "crop": 3,
    }
    locations = []
    for objective in ("absolute_ce", "regression"):
        model = PointTransformer(
            ModelConfig(width=16, layers=1, heads=2, bins=17, objective=objective)
        )
        vocab = Vocabulary(np.full(19, 100.0), 17)
        path = tmp_path / objective
        path.mkdir()
        result = evaluate(model, vocab, ocean, config, torch.device("cpu"), path, 1)
        assert len(result["per_depth"]) == 19
        rows = [
            json.loads(line)
            for line in (path / "predictions-0000001.jsonl").read_text().splitlines()
        ]
        locations.append([(r["origin"], r["row"], r["col"], r["depth"]) for r in rows])
    assert locations[0] == locations[1]


def test_baseline_cache_scores_identical_targets(ocean, tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text(json.dumps({"source": ocean.receipt()}))
    np.save(baseline / "temperature-0004.npy", ocean.read(4)["target"])
    config = {
        "validation_dates": 1,
        "validation_seed": 42,
        "validation_points_per_depth": 1,
        "microbatch": 2,
        "crop": 3,
        "baseline_cache": str(baseline),
    }
    model = PointTransformer(ModelConfig(width=16, layers=1, heads=2, bins=17))
    vocab = Vocabulary(np.full(19, 100.0), 17)
    result = evaluate(model, vocab, ocean, config, torch.device("cpu"), tmp_path, 1)
    assert all(
        m["samudra_rmse_c"] == 0 and m["samudra_nmse"] == 0 for m in result["per_depth"]
    )


def test_support_fit_and_precision_audit(ocean, tmp_path):
    support_path = audit_support(ocean, tmp_path / "support")
    support = json.loads(support_path.read_text())
    assert support["complete_training_scan"]
    assert support["dates_scanned"] == 3
    result = audit_precision(
        ocean,
        {"support": str(support_path), "model": {"bins": 2048}},
        tmp_path / "precision",
    )
    assert result["validation_dates"] == 1
    assert len(result["per_depth"]) == 19
    assert all(
        row["persistence_rmse_c"] == pytest.approx(0.4, abs=1e-5)
        for row in result["per_depth"]
    )
