# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Pending-batch resume, prefetch equivalence, and global-batch DDP semantics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_observation_baseline_harness import (  # type: ignore[import-not-found]
    TinySources,
    make_state,
    recipe,
    validate,
)

from samudra.observations import training as baseline
from samudra.observations.pipeline import map_tensors, pin_host_tensor
from samudra.observations.qualify import equal_tree


def test_staging_copies_shared_forecast_and_reconstruction_tensors_once():
    tensor = torch.arange(12).reshape(2, 2, 3)
    sample = {"fields": tensor, "forcing": tensor, "raw": {"duplicate": tensor}}
    calls = []

    def copy(value):
        calls.append(value)
        return value.clone()

    result = map_tensors([{"sample": sample, "auxiliary": sample}], copy)[0]
    assert len(calls) == 1
    assert result["sample"]["fields"] is result["auxiliary"]["forcing"]
    assert "raw" not in result["sample"]
    torch.testing.assert_close(result["sample"]["fields"], tensor, rtol=0, atol=0)


def test_pin_staging_materializes_broadcast_masks_without_changing_values(monkeypatch):
    mask = torch.arange(12).reshape(1, 1, 3, 4).remainder(2).bool()
    expanded = mask.expand(1, 26, 3, 4)

    def pin_with_preserved_strides(value):
        # Match pin_memory's layout-preserving copy without requiring CUDA.
        result = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype)
        return result.copy_(value)

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_with_preserved_strides)
    with pytest.raises(RuntimeError, match="single memory location"):
        expanded.pin_memory()
    staged = pin_host_tensor(expanded)
    torch.testing.assert_close(staged, expanded, rtol=0, atol=0)
    staged[:, 0] = ~staged[:, 0]
    torch.testing.assert_close(staged[:, 1:], expanded[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(expanded[:, :1], mask, rtol=0, atol=0)


@pytest.mark.cuda
def test_pin_staging_copies_expanded_mask_to_cuda_exactly():
    mask = torch.arange(12).reshape(1, 1, 3, 4).remainder(2).bool()
    expanded = mask.expand(1, 26, 3, 4)
    staged = pin_host_tensor(expanded)
    assert staged.is_pinned()
    torch.testing.assert_close(
        staged.to("cuda", non_blocking=True).cpu(), expanded, rtol=0, atol=0
    )


class TinyData(SimpleNamespace):
    def to(self, device):
        return self


class Sources(TinySources):
    obs = TinyData(device="cpu")


def objective(
    model,
    data,
    source,
    sample,
    auxiliary,
    seed,
    *,
    cached=None,
    coverage=None,
    weights=None,
):
    target = sample["origin"] / 100 + seed % 100 / 100
    loss = (model.weight - target).square().mean()
    marker = 1000 * (TinySources.names.index(source) + 1)
    if cached is not None:
        assert torch.equal(
            cached, torch.full_like(cached, marker + sample["start_lead"])
        )
    states = torch.full((1, 2, 1, 1, 1), float(marker + sample["next_start_lead"]))
    return loss, {"forecast": loss}, states


def configuration(**changes):
    return recipe(
        mix="om4-obs",
        main_updates=4,
        cooldown_updates=2,
        save_freq=1,
        validation_freq=6,
        update_prefetch=True,
        **changes,
    )


def run(cfg, state, folder, limit):
    folder.mkdir(parents=True, exist_ok=True)
    baseline.train(
        cfg,
        state,
        Sources(),
        folder,
        {"config": cfg.scientific_config()},
        validate,
        lambda _: None,
        stop_after=limit,
    )


def test_prefetch_keeps_inputs_pending_plan_optimizer_and_rng_exact(
    monkeypatch, tmp_path
):
    torch.set_num_threads(1)
    monkeypatch.setattr(baseline, "objective", objective)
    cfg = configuration()
    state = make_state(cfg)
    run(cfg, state, tmp_path / "on", 3)
    assert state.pending_plan["update"] == 3
    anchor = tmp_path / "on/ckpt.pt"
    run(cfg, state, tmp_path / "expected", 6)
    expected = torch.load(tmp_path / "expected/endpoint.pt", weights_only=False)
    off = cfg.model_copy(update={"prefetch_enabled": False})
    assert off.scientific_config() == cfg.scientific_config()
    state.restore(anchor, {"config": cfg.scientific_config()})
    run(off, state, tmp_path / "off", 6)
    actual = torch.load(tmp_path / "off/endpoint.pt", weights_only=False)
    equal_tree(actual, expected)
    assert state.pending_plan is None and state.cooled
    assert state.exposure["obs"]["fresh"] + state.exposure["obs"]["replay"] == 32
    assert state.exposure["om4"]["fresh"] + state.exposure["om4"]["replay"] == 16


def _distributed_worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2
    )
    try:
        baseline.objective = objective
        cfg = configuration()
        state = make_state(cfg)
        folder = root / "distributed"
        run(cfg, state, folder, 2)
        if rank == 0:
            state.save(root / "anchor.pt", {"config": cfg.scientific_config()})
        dist.barrier()
        run(cfg, state, root / "expected", 6)
        dist.barrier()
        state.restore(root / "anchor.pt", {"config": cfg.scientific_config()})
        run(cfg, state, root / "resumed", 6)
        weights = [torch.zeros_like(state.model.weight) for _ in range(2)]
        dist.all_gather(weights, state.model.weight.detach())
        torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
        if rank == 0:
            equal_tree(
                torch.load(root / "expected/endpoint.pt", weights_only=False),
                torch.load(root / "resumed/endpoint.pt", weights_only=False),
            )
    finally:
        dist.destroy_process_group()


def test_prefetched_uncooled_fork_matches_continuous_main(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setattr(baseline, "objective", objective)
    short = configuration()
    long = short.model_copy(update={"main_updates": 8, "milestone_updates": [8]})
    parent = make_state(short)
    run(short, parent, tmp_path / "short", 4)
    child = make_state(long)
    child.restore(
        tmp_path / "short/main.pt", {"config": long.scientific_config()}, fork=True
    )
    run(long, child, tmp_path / "fork", 5)
    direct = make_state(long)
    run(long, direct, tmp_path / "direct", 5)
    equal_tree(
        torch.load(tmp_path / "fork/ckpt.pt", weights_only=False),
        torch.load(tmp_path / "direct/ckpt.pt", weights_only=False),
    )


def test_ddp_preserves_global_batch_replay_capacity_and_exact_resume(
    monkeypatch, tmp_path
):
    torch.set_num_threads(1)
    monkeypatch.setattr(baseline, "objective", objective)
    cfg = configuration()
    state = make_state(cfg)
    run(cfg, state, tmp_path / "single", 6)
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)
    expected = torch.load(tmp_path / "single/endpoint.pt", weights_only=False)
    actual = torch.load(tmp_path / "resumed/endpoint.pt", weights_only=False)
    assert actual["exposure"] == expected["exposure"]
    for name in actual["model"]:
        torch.testing.assert_close(
            actual["model"][name], expected["model"][name], rtol=1e-6, atol=1e-7
        )
    equal_tree(actual["pools"], expected["pools"])
    equal_tree(actual["sampler_rng"], expected["sampler_rng"])
    assert len(actual["rank_rng"]) == 2
    with pytest.raises(ValueError, match="world size differs"):
        state.restore(
            tmp_path / "resumed/endpoint.pt", {"config": cfg.scientific_config()}
        )
