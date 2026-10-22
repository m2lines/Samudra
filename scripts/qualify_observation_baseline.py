#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-data qualification of the production U-global harness, not reference training.

Uses the full model, real observations/OM4/CM4, all three replay pools and the
unchanged loss. Only this process shortens the schedule to four main plus two
cooldown updates and two examples/update (one fresh, one replay). A single full
annual validation runs at the end. QUALIFIED.json records engineering evidence;
its forecast scores are not scientifically trained baseline results.
"""

import argparse
import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path

# Required before CUDA context creation for exact optimizer-resume comparison.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from samudra.observations import baseline
from samudra.observations.archive import validate_normalization_policy
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
from samudra.observations.model_data import ModelMonths
from samudra.observations.replay import ReplayEntry
from samudra.observations.samples import Samples
from samudra.observations.state import TrainingState, write_json


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
    # Native model forcing still has its own three-channel normalization. It is
    # never substituted for the shared observation-derived physical-state scales.
    for name, model in models.items():
        if model.forcing_mean.shape != (3,) or model.forcing_std.shape != (3,):
            raise AssertionError(f"Unexpected {name} native forcing scales")
        np.testing.assert_array_equal(model.forcing_mean, models["om4"].forcing_mean)
        np.testing.assert_array_equal(model.forcing_std, models["om4"].forcing_std)
    return {
        "fresh_months": checks,
        "global_training_moment_checks": global_moments,
        "state_normalization_shared_by_all_sources": True,
        "native_forcing_uses_only_three_om4_scales": True,
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
def qualify_chains(state, sources):
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
                device=sources.obs.device,
            ):
                loss, parts, following_states = baseline.objective(
                    model,
                    sources.obs,
                    source,
                    sample,
                    auxiliary,
                    baseline.SEED + month,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "daily", "normalization", "om4", "cm4", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.output / "QUALIFIED.json").exists() or (
        args.output / "latest.pt"
    ).exists():
        raise ValueError("Qualification output must be new; preserve prior evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    random.seed(baseline.SEED)
    np.random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = Samples(
        args.observations,
        args.device,
        daily=args.daily,
        normalization=args.normalization,
    )
    models = {
        "om4": ModelMonths(args.om4 / "OM4.zarr", "om4", data.grid, args.om4),
        "cm4": ModelMonths(args.cm4, "cm4", data.grid, args.om4),
    }
    sources = baseline.Sources(data, models)
    parity = data_parity(data, models)
    emit({"event": "data_parity", **parity})
    model = baseline.SurfaceInitializedModel(data.grid["names"].tolist()).to(
        args.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=baseline.LEARNING_RATE, weight_decay=0.01
    )
    state = TrainingState(model, optimizer, sources.names, baseline.SEED)
    args.mix = "om4-cm4-obs"
    production_manifest = baseline.run_manifest(args, data, sources)
    chains = qualify_chains(state, sources)
    # Qualification-only scope. The imported production recipe on disk is unchanged.
    baseline.MAIN_UPDATES, baseline.COOLDOWN_UPDATES = 4, 2
    baseline.BATCH_SIZE, baseline.FRESH_EXAMPLES = 2, 1
    manifest = baseline.run_manifest(args, data, sources)
    manifest["qualification_only"] = True
    manifest["qualification_script_sha256"] = baseline.digest(__file__)
    write_json(manifest, args.output / "manifest.json")
    metrics, records = {}, []

    def validate(current):
        trajectory = data.trajectory(
            baseline.VALIDATION_ORIGIN, steps=FORECAST_STEPS, split="validation"
        )
        metrics.update(baseline.annual_metrics(current, data, trajectory))
        if metrics["rollout_val/365d/scored_channels"] != 29:
            raise AssertionError("Annual selection must score exactly 29 channels")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite annual metric")
        return metrics

    def record(value):
        records.append(value)
        with (args.output / "qualification.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        emit(value)

    baseline.train(
        state, sources, args.output, manifest, validate, record, stop_after=4
    )
    before = args.output / "before-cooldown.pt"
    state.save(before, manifest)
    baseline.train(
        state, sources, args.output, manifest, validate, record, stop_after=5
    )
    expected = args.output / "expected-next-update.pt"
    state.save(expected, manifest)
    expected_payload = torch.load(expected, map_location="cpu", weights_only=False)
    state.restore(before, manifest)
    baseline.train(
        state, sources, args.output, manifest, validate, record, stop_after=5
    )
    resumed = args.output / "resumed-next-update.pt"
    state.save(resumed, manifest)
    actual_payload = torch.load(resumed, map_location="cpu", weights_only=False)
    equal_tree(actual_payload, expected_payload)
    del actual_payload, expected_payload
    emit(
        {
            "event": "exact_resume",
            "optimizer_ema_rng_replay": True,
            "cooldown_reset": state.cooled,
        }
    )
    baseline.train(
        state, sources, args.output, manifest, validate, record, stop_after=6
    )
    for source, exposure in state.exposure.items():
        if exposure["fresh"] < 1 or exposure["replay"] < 1:
            raise AssertionError(f"Missing fresh/replay optimizer example for {source}")
    if not state.cooled or state.exposure["obs"]["updates"] != 4:
        raise AssertionError("Observation-only cooldown was not exercised")
    report = {
        "purpose": "engineering qualification, not trained scientific baseline results",
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "deterministic_algorithms": True,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "daily_manifests_sha256": {
            str(path.resolve()): baseline.digest(path)
            for path in sorted(args.daily.glob("*/*/manifest.json"))
        },
        "production_manifest": production_manifest,
        "qualification_manifest": manifest,
        "data_parity": parity,
        "replay_chains": chains,
        "exact_next_update_resume": True,
        "cooldown_reset": state.cooled,
        "checkpoint_payloads_compared": [
            "model",
            "optimizer",
            "ema",
            "sampler_rng",
            "pools",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "exposure",
        ],
        "annual_metrics": metrics,
        "examples_by_source": state.exposure,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_records": records,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoints_sha256": {
            path.name: baseline.digest(path)
            for path in (before, expected, resumed, args.output / "endpoint.pt")
        },
    }
    write_json(report, args.output / "QUALIFIED.json")
    emit(
        {
            "event": "qualified",
            "path": str(args.output / "QUALIFIED.json"),
            "seconds": report["elapsed_seconds"],
        }
    )


if __name__ == "__main__":
    main()
