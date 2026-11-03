# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Full-size engineering qualification through the configured observation harness."""

import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from samudra.observations import training as baseline
from samudra.observations.archive import validate_normalization_policy
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.constants import (
    FORECAST_STEPS,
    MAX_REPLAY_MONTHS,
    MODEL_DEPTHS,
    OBSERVED_DEPTHS,
    SALINITY_OFFSET,
    STATE_CHANNELS,
    SURFACE_CHANNELS,
    TEMPERATURE_OFFSET,
)
from samudra.observations.replay import ReplayEntry
from samudra.observations.state import TrainingState, write_json
from samudra.observations.train import (
    build_sources,
    init_tracking,
    require_single_process,
)
from samudra.utils.distributed import set_seed


class QualificationConfig(ObservationTrainConfig):
    qualify_chains: bool = False
    qualify_data: bool = False


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def equal_tree(actual, expected, path="checkpoint"):
    """Compare payloads, not serialization byte layout; tensor equality is exact."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif dataclasses.is_dataclass(actual):
        for field in dataclasses.fields(actual):
            equal_tree(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in actual:
            equal_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal_tree(a, b, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def verify_global_moments(data):
    """Independently check fitted scales using standardized residual moments.

    The fitter accumulates raw first/second moments. Here we independently
    require E[(x-mean)/std]=0 and E[((x-mean)/std)^2]=1 over all training samples,
    all latitudes, and each channel's finite wet support. This intentionally does
    not compare with the historical latitude-restricted statistics artifact.
    """
    metadata = json.loads((data.normalization / "normalization.json").read_text())
    validate_normalization_policy(metadata)
    grid = data.grid
    profile = [
        offset + depth
        for offset in (TEMPERATURE_OFFSET, SALINITY_OFFSET)
        for depth in range(OBSERVED_DEPTHS)
    ]
    masks = {
        "surface": grid["mask"][SURFACE_CHANNELS],
        "atmosphere": grid["mask"][TEMPERATURE_OFFSET][None],
        "interior": grid["mask"][profile],
    }
    latitude = grid["lat"][:, None]
    area = np.cos(np.deg2rad(latitude))
    moments = {name: np.zeros((4, len(data.stats[f"{name}_mean"]))) for name in masks}
    for path in data.paths("train"):
        with np.load(path) as sample:
            for name, accum in moments.items():
                values = np.asarray(sample[name], dtype=np.float64)
                if name == "interior":
                    values = values.reshape(
                        1, 2 * OBSERVED_DEPTHS, *grid["mask"].shape[-2:]
                    )
                valid = np.isfinite(values) & masks[name].astype(bool)
                weight = np.where(valid, area, 0)
                mean = data.stats[f"{name}_mean"][None, :, None, None]
                std = data.stats[f"{name}_std"][None, :, None, None]
                standardized = np.where(valid, (values - mean) / std, 0)
                accum[0] += weight.sum((0, 2, 3))
                accum[1] += (standardized * weight).sum((0, 2, 3))
                accum[2] += (standardized**2 * weight).sum((0, 2, 3))
                accum[3] += (weight * (np.abs(latitude) > 60)).sum((0, 2, 3))
    checks = {}
    for name, (weight, first, second, polar_weight) in moments.items():
        if not (weight > 0).all():
            raise AssertionError(f"No finite-wet normalization support for {name}")
        # Artifacts are float32; a salinity mean near 35 with a small standard
        # deviation can have appreciable standardized rounding error.
        mean_roundoff = (
            np.abs(np.spacing(data.stats[f"{name}_mean"])) / data.stats[f"{name}_std"]
            + 1e-6
        )
        np.testing.assert_array_less(np.abs(first / weight), mean_roundoff)
        np.testing.assert_array_less(
            np.abs(second / weight - 1), 1e-5 + mean_roundoff**2
        )
        checks[name] = {
            "max_abs_standardized_mean": float(np.abs(first / weight).max()),
            "max_abs_standardized_second_moment_minus_one": float(
                np.abs(second / weight - 1).max()
            ),
            "polar_weight_fraction_by_channel": (polar_weight / weight).tolist(),
        }
    return checks


def data_parity(data, models):
    """Daily replay preparation agrees with two independently materialized months."""
    checks = []
    for month in ("1993-05", "2001-01"):
        materialized = data.load(data.root / "train" / f"{month}.npz")
        daily = data.trajectory(month, target_month=month)
        for key in (
            "surface",
            "atmosphere",
            "contexts",
            "validity",
            "interior",
            "raw_surface",
            "month_weights",
        ):
            torch.testing.assert_close(
                materialized[key], daily[key], rtol=0, atol=0, equal_nan=True
            )
        checks.append(
            {
                "month": month,
                "monthly_sha256": baseline.digest(data.root / "train" / f"{month}.npz"),
                "exact": True,
            }
        )
    global_moments = verify_global_moments(data)
    # Independently check the 77-channel map of the newly fitted global scales.
    mean = np.zeros(STATE_CHANNELS, np.float32)
    std = np.ones(STATE_CHANNELS, np.float32)
    for offset, group in ((TEMPERATURE_OFFSET, 0), (SALINITY_OFFSET, 1)):
        mean[offset : offset + OBSERVED_DEPTHS] = data.stats["interior_mean"][
            group * OBSERVED_DEPTHS : (group + 1) * OBSERVED_DEPTHS
        ]
        std[offset : offset + OBSERVED_DEPTHS] = data.stats["interior_std"][
            group * OBSERVED_DEPTHS : (group + 1) * OBSERVED_DEPTHS
        ]
        mean[offset + OBSERVED_DEPTHS : offset + MODEL_DEPTHS] = mean[
            offset + OBSERVED_DEPTHS - 1
        ]
        std[offset + OBSERVED_DEPTHS : offset + MODEL_DEPTHS] = std[
            offset + OBSERVED_DEPTHS - 1
        ]
    mean[SURFACE_CHANNELS], std[SURFACE_CHANNELS] = (
        data.stats["surface_mean"],
        data.stats["surface_std"],
    )
    for source, grid in [
        ("obs", data.grid),
        *((name, model.grid) for name, model in models.items()),
    ]:
        np.testing.assert_array_equal(
            grid["mean"], mean, err_msg=f"{source} state mean"
        )
        np.testing.assert_array_equal(grid["std"], std, err_msg=f"{source} state std")
    return {
        "fresh_months": checks,
        "global_training_moment_checks": global_moments,
        "state_normalization_shared_by_all_sources": True,
        "shared_eight_field_era5": all(m.era5 is not None for m in models.values()),
    }


def chain_origin(sources, source):
    if source == "obs":
        return "2001-01"
    return next(
        index
        for index in range(len(sources.models[source].windows))
        if sources.layout(source, index, MAX_REPLAY_MONTHS - 1) is not None
    )


@torch.no_grad()
def qualify_chains(state, sources, cfg):
    """Advance actual model-generated states to one year and populate real pools."""
    model = state.model
    model.eval()
    records = {}
    coverage = sources.load("obs", "2001-01")["validity"]
    for source in sources.names:
        origin = chain_origin(sources, source)
        cached, chain = None, []
        for month in range(MAX_REPLAY_MONTHS):
            layout = sources.layout(source, origin, month)
            if layout is None:
                break
            sample = sources.load(source, origin, month)
            auxiliary = sources.auxiliary(source, origin, month, sample)
            with baseline.autocast(
                torch.device(sources.obs.device).type == "cuda",
                torch.bfloat16,
            ):
                loss, parts, following_states = baseline.objective(
                    model,
                    sources.obs,
                    source,
                    sample,
                    auxiliary,
                    cfg.experiment.rand_seed + month,
                    weights=cfg.loss,
                    cached=cached,
                    coverage=coverage if source != "obs" else None,
                )
            if not torch.isfinite(following_states).all():
                raise FloatingPointError(f"Nonfinite {source} replay history")
            chain.append(
                {
                    "month": month,
                    "start_lead": sample["start_lead"],
                    "end_lead": sample["end_lead"],
                    "loss": float(loss),
                    "components": {key: float(value) for key, value in parts.items()},
                }
            )
            next_layout = sources.layout(source, origin, month + 1)
            if next_layout is not None:
                # Fill capacity to measure worst-case checkpoint storage.
                for _ in range(cfg.replay.per_month):
                    state.pools[source].add(
                        ReplayEntry(
                            source,
                            origin,
                            month + 1,
                            next_layout["start_lead"],
                            0,
                            following_states,
                        )
                    )
            cached = following_states
            emit({"event": "replay_chain", "source": source, **chain[-1]})
        if len(chain) != MAX_REPLAY_MONTHS or chain[-1]["end_lead"] != FORECAST_STEPS:
            raise AssertionError(
                f"{source} did not complete a native contiguous one-year chain"
            )
        records[source] = {"origin": origin, "months": chain, "maximum_days": 365}
    return records


def run(cfg):
    require_single_process()
    if cfg.main_updates != 4 or cfg.cooldown_updates != 2:
        raise ValueError(
            "Qualification requires exactly four main and two cooldown updates"
        )
    if cfg.gradient_accumulation_steps != 8:
        raise ValueError("Qualify the production effective batch of eight")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(cfg.experiment.rand_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Qualification requires a BF16-capable CUDA GPU")
    output = cfg.experiment.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "QUALIFIED.json").exists():
        emit({"event": "already_qualified", "path": str(output)})
        return
    started = time.perf_counter()
    sources = build_sources(cfg, torch.device("cuda"))
    data = sources.obs
    manifest = baseline.run_manifest(cfg, data, sources)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Qualification output belongs to different code/data/config")
    cfg.save_yaml(output / "config.yaml")
    write_json(manifest, manifest_path)
    parity = data_parity(data, sources.models) if cfg.qualify_data else None
    model = cfg.model.build(data.grid["names"].tolist()).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    state = TrainingState(
        model,
        optimizer,
        sources.names,
        cfg.experiment.rand_seed,
        ema_decay=cfg.ema_decay,
        faster_decay_at_start=cfg.faster_decay_at_start,
        replay_per_month=cfg.replay.per_month,
        replay_max_age=cfg.replay.max_age,
    )
    chain_path = output / "chains.json"
    pool_timing_path = output / "full-pool-checkpoint.json"
    if cfg.qualify_chains and not chain_path.exists():
        chains = qualify_chains(state, sources, cfg)
        before_save = time.perf_counter()
        pool_checkpoint = output / "full-pool.pt"
        state.save(pool_checkpoint, manifest)
        write_json(
            {
                "seconds": time.perf_counter() - before_save,
                "bytes": pool_checkpoint.stat().st_size,
            },
            pool_timing_path,
        )
        write_json(chains, chain_path)
        # Benchmark starts with empty pools, like the scientific campaign.
        state = TrainingState(
            model,
            optimizer,
            sources.names,
            cfg.experiment.rand_seed,
            ema_decay=cfg.ema_decay,
            faster_decay_at_start=cfg.faster_decay_at_start,
            replay_per_month=cfg.replay.per_month,
            replay_max_age=cfg.replay.max_age,
        )
    tracker = init_tracking(cfg, manifest)
    records = []
    metrics = {}

    def record(value):
        records.append(value)
        tracker.log(value)
        with (output / "qualification.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        emit(value)

    def validate(current):
        trajectory = data.trajectory(
            cfg.validation_origin, steps=FORECAST_STEPS, split="validation"
        )
        metrics.update(baseline.annual_metrics(current, data, trajectory))
        if metrics["rollout_val/365d/scored_channels"] != 29:
            raise AssertionError("Annual selection must score exactly 29 channels")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite annual metric")
        return metrics

    def train_until(limit, config=cfg, identity=manifest):
        baseline.train(
            config, state, sources, output, identity, validate, record, stop_after=limit
        )

    # Completed checkpoints survive preemption. Repeat comparison stages from
    # the same uncooled checkpoint so interrupted qualification stays bounded.
    main_path = output / "main.pt"
    if main_path.exists():
        state.restore(main_path, manifest)
    else:
        checkpoint = output / "ckpt.pt"
        if checkpoint.exists():
            state.restore(checkpoint, manifest)
        train_until(4)
    parent_sha = baseline.digest(main_path)
    fork_cfg = cfg.model_copy(update={"main_updates": 8, "milestone_updates": [8]})
    fork_manifest = {**manifest, "config": fork_cfg.scientific_config()}
    state.restore(main_path, fork_manifest, fork=True)
    if state.cooled or state.updates != 4:
        raise AssertionError("Main continuation inherited cooldown")
    state.restore(main_path, manifest)
    train_until(5)
    expected = output / "expected.pt"
    before_save = time.perf_counter()
    state.save(expected, manifest)
    save_seconds = time.perf_counter() - before_save
    expected_payload = torch.load(expected, map_location="cpu", weights_only=False)
    before_restore = time.perf_counter()
    state.restore(main_path, manifest)
    restore_seconds = time.perf_counter() - before_restore
    train_until(5)
    resumed = output / "resumed.pt"
    state.save(resumed, manifest)
    actual_payload = torch.load(resumed, map_location="cpu", weights_only=False)
    equal_tree(actual_payload, expected_payload)
    del actual_payload, expected_payload
    if baseline.digest(main_path) != parent_sha:
        raise AssertionError("Cooldown branch modified main checkpoint")
    emit(
        {
            "event": "exact_resume",
            "optimizer_ema_rng_replay": True,
            "isolated_uncooled_fork": True,
        }
    )
    train_until(6)
    replay = cfg.fresh_examples < cfg.gradient_accumulation_steps
    for source, exposure in state.exposure.items():
        if exposure["fresh"] < 1 or (replay and exposure["replay"] < 1):
            raise AssertionError(f"Missing fresh/replay training for {source}")
        if not replay and (exposure["replay"] or state.pools[source].entries):
            raise AssertionError("No-replay control used replay")
    if not state.cooled or state.ema.num_updates != 2:
        raise AssertionError("Cooldown must reset EMA once and retain optimizer")
    if cfg.qualify_chains:
        # Include populated Adam moments AND full replay capacity in the storage
        # benchmark. This diagnostic checkpoint is never used for training.
        saturated = torch.load(
            output / "full-pool.pt", map_location="cpu", weights_only=False
        )
        for name, pool in state.pools.items():
            pool.load_state_dict(saturated["pools"][name])
        del saturated
        before_save = time.perf_counter()
        saturated_path = output / "full-pool-trained.pt"
        state.save(saturated_path, manifest)
        write_json(
            {
                "seconds": time.perf_counter() - before_save,
                "bytes": saturated_path.stat().st_size,
            },
            pool_timing_path,
        )
        state.restore(output / "endpoint.pt", manifest)
    report = {
        "purpose": "engineering qualification; not a scientific training result",
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "exact_resume": True,
        "isolated_uncooled_fork": True,
        "data_parity": parity,
        "annual_metrics": metrics,
        "exposure": state.exposure,
        "training_records": records,
        "checkpoint_seconds": save_seconds,
        "full_pool_checkpoint_seconds": json.loads(pool_timing_path.read_text())[
            "seconds"
        ]
        if pool_timing_path.exists()
        else 0,
        "restore_seconds": restore_seconds,
        "checkpoint_bytes": expected.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "wandb_url": tracker.run.url if tracker.run else None,
    }
    write_json(report, output / "QUALIFIED.json")
    tracker.finish()
    emit({"event": "qualified", **report})


def main():
    run(QualificationConfig.from_yaml_and_cli())


if __name__ == "__main__":
    main()
