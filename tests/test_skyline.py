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

from samudra.skyline.coverage import (
    corrected_mean,
    geometric_groups,
    proposal_probabilities,
    sample_coverage,
)
from samudra.skyline.data import CHANNELS, OceanSource, audit_support
from samudra.skyline.model import ModelConfig, PointTransformer, Vocabulary
from samudra.skyline.precision import audit_precision
from samudra.skyline.run import FrameStream, evaluate, resume_checkpoint
from samudra.skyline.stage import merge_validation


def test_coverage_corrects_loss_and_gradient_exactly():
    labels = np.array([0, 0, 0, 0, 1])
    q = torch.tensor(proposal_probabilities(labels), dtype=torch.float64)
    assert float(q[-1]) > float(q[0])
    weights = 1 / (len(labels) * q)
    assert float(weights.max()) <= 2
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([-2.0, 0.0, 1.0, 3.0, 8.0], dtype=torch.float64)
    losses = (parameter - targets).square()
    base = losses.mean()
    proposal_expectation = (q * weights * losses).sum()
    torch.testing.assert_close(base, proposal_expectation)
    (base_grad,) = torch.autograd.grad(base, parameter, retain_graph=True)
    (proposal_grad,) = torch.autograd.grad(proposal_expectation, parameter)
    torch.testing.assert_close(base_grad, proposal_grad)
    # Weights must be constants for differentiation, even when scores depend on
    # the current model. Self-normalizing a realized batch would change the loss.
    weights = torch.tensor([0.2, 1.8], requires_grad=True)
    x = torch.tensor([2.0, 5.0], requires_grad=True)
    corrected_mean(x, weights).backward()
    assert weights.grad is None
    torch.testing.assert_close(x.grad, torch.tensor([0.1, 0.9]))


def test_geometric_coverage_retains_distant_state():
    x = np.array([[0.0], [0.01], [0.02], [0.03], [20.0]])
    labels = geometric_groups(x, np.zeros(5, dtype=int), 2)
    assert labels[-1] not in labels[:-1]
    q = proposal_probabilities(labels)
    assert q[-1] > q[0]
    np.testing.assert_allclose(q.sum(), 1)


def test_coverage_target_isolation_and_exact_rng_replay(ocean):
    config = dict(candidate_pool=128, clusters_per_depth=3, base_fraction=0.5)
    frame = ocean.read(1)
    rng = np.random.default_rng(13)
    state = copy.deepcopy(rng.bit_generator.state)
    first = sample_coverage(ocean, frame, rng, 8, 3, config)
    rng.bit_generator.state = state
    second = sample_coverage(
        ocean, dict(frame, target=frame["target"] + 100), rng, 8, 3, config
    )
    for key in first:
        if key != "target":
            torch.testing.assert_close(first[key], second[key])
    torch.testing.assert_close(second["target"], first["target"] + 100)
    assert (first["importance"] > 0).all()
    assert (first["importance"] <= 2).all()


def test_validation_merge_uses_squared_errors_and_counts():
    parts = []
    for origin, count, error in [(1, 2, 3.0), (2, 6, 5.0)]:
        parts.append(
            dict(
                origins=[origin],
                per_depth=[
                    dict(
                        depth=z,
                        depth_m=z + 1.0,
                        count=count,
                        rmse_c=error,
                        persistence_rmse_c=2.0,
                        nmse=error**2,
                        bias_c=error,
                        bin_nll=None,
                        in_support_density_nll=None,
                        outside_support=None,
                        quantization_rmse_c=None,
                        samudra_rmse_c=None,
                        samudra_nmse=None,
                    )
                    for z in range(19)
                ],
                mean_depth_bin_nll=None,
            )
        )
    result = merge_validation(parts)
    assert result["origins"] == [1, 2]
    assert result["per_depth"][0]["count"] == 8
    assert result["per_depth"][0]["rmse_c"] == pytest.approx(np.sqrt(21))
    assert result["mean_depth_nmse"] == 21


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
from samudra.skyline.utilization import GpuMonitor, UtilizationWindow


def test_utilization_guard_includes_idle_intervals_and_warmup():
    window = UtilizationWindow(minimum_percent=60, warmup_seconds=10, window_seconds=20)
    for t in range(0, 10, 5):
        assert window.add(t, 0) is None
    for t in range(10, 30, 5):
        assert window.add(t, 100) is None
    assert window.add(30, 100) == 100
    # GPU-idle checkpoint/I/O intervals count, not just optimizer execution.
    for t in range(35, 60, 5):
        mean = window.add(t, 0)
    assert mean == 0
    assert mean < window.minimum
    with pytest.raises(ValueError):
        window.add(60, float("nan"))


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_gpu_monitor_targets_assigned_uuid_and_fails_closed(
    tmp_path, monkeypatch, prefix
):
    from types import SimpleNamespace

    assigned = "bef7083f-59fb-c045-0bf1-89e763150c98"
    monitor = GpuMonitor(tmp_path, prefix + assigned, 0, {})

    def query(args, **kwargs):
        assert args[1:3] == ["-i", "GPU-" + assigned]
        monitor.stop.set()
        return SimpleNamespace(stdout="91, 12000, 350\n")

    monkeypatch.setattr("samudra.skyline.utilization.subprocess.run", query)
    monitor.run()
    assert json.loads(monitor.path.read_text())["utilization_percent"] == 91
    assert monitor.reason is None
    monitor.stop.clear()

    def failed_query(*args, **kwargs):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr("samudra.skyline.utilization.subprocess.run", failed_query)
    monitor.run()
    assert monitor.reason is not None
    assert "telemetry unavailable" in monitor.reason


@pytest.mark.parametrize("utilization", [0, 55])
def test_low_utilization_is_logged_without_stopping(tmp_path, monkeypatch, utilization):
    from types import SimpleNamespace

    monitor = GpuMonitor(
        tmp_path, "GPU-test", 0, {"warmup_seconds": 0, "window_seconds": 10}
    )
    monitor.window.add(0, utilization)
    monitor.window.add(5, utilization)
    clock = iter([100, 110])
    monkeypatch.setattr(
        "samudra.skyline.utilization.time.monotonic", lambda: next(clock)
    )

    def query(*args, **kwargs):
        monitor.stop.set()
        return SimpleNamespace(stdout=f"{utilization}, 12000, 350\n")

    monkeypatch.setattr("samudra.skyline.utilization.subprocess.run", query)
    monitor.run()
    record = json.loads(monitor.path.read_text())
    assert record["rolling_utilization_percent"] == utilization
    assert record["stop_reason"] is None
    assert monitor.reason is None


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
    ocean.location_cdfs = [np.array([1.0])] * 19
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
        for row in rows:
            metric = result["per_depth"][row["depth"]]
            assert metric["rmse_c"] == pytest.approx(
                abs(row["prediction_c"] - row["truth_c"]), abs=2e-5
            )
            assert (
                row["nll"] == pytest.approx(metric["bin_nll"])
                if row["nll"] is not None
                else metric["bin_nll"] is None
            )
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
    records = [
        json.loads(line)
        for line in (tmp_path / "predictions-0000001.jsonl").read_text().splitlines()
    ]
    assert all(row["samudra_c"] == row["truth_c"] for row in records)


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
