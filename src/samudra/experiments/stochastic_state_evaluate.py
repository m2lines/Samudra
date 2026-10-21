# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen observation metrics for coherent stochastic-state pilot trajectories."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.diffusion_evaluation import evaluate_point_metrics
from samudra.experiments.diffusion_global_annual import ORIGINS, AnnualMembers
from samudra.experiments.diffusion_global_evaluate import RecordingForecast
from samudra.experiments.observation_annual import evaluate_origins
from samudra.experiments.observation_metrics import spatial_error
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.metrics import spectra


def temporal_statistics(fields, mask, lat):
    """Per-member/channel five-day increments and centered lag-one correlation.

    Inputs [member,time,channel,lat,lon] are physical units. Correlation uses
    within-year anomalies at each wet cell, then pools with cosine area weights;
    it is not a correlation of spatially averaged fields or ensemble means.
    """
    values = np.asarray(fields, dtype=np.float64)
    support = np.asarray(mask, dtype=bool) & np.isfinite(values).all(axis=(0, 1))
    weights = support * np.cos(np.deg2rad(lat))[None, :, None]
    denominator = weights.sum(axis=(-2, -1))
    if np.any(denominator <= 0):
        raise ValueError("No common finite wet support for temporal diagnostics")
    values = np.where(support[None, None], values, 0)
    increments = np.diff(values, axis=1)
    increment_rms = np.sqrt(
        (increments**2 * weights).sum(axis=(1, 3, 4))
        / ((values.shape[1] - 1) * denominator)
    )
    centered = values - values.mean(axis=1, keepdims=True)
    before, after = centered[:, :-1], centered[:, 1:]
    covariance = (before * after * weights).sum(axis=(1, 3, 4))
    scale = np.sqrt(
        (before**2 * weights).sum(axis=(1, 3, 4))
        * (after**2 * weights).sum(axis=(1, 3, 4))
    )
    correlation = np.divide(
        covariance, scale, out=np.zeros_like(scale), where=scale > 0
    )
    return dict(
        five_day_increment_rms=increment_rms.tolist(),
        lag_one_correlation=[
            [
                float(value) if valid else None
                for value, valid in zip(row, ok, strict=True)
            ]
            for row, ok in zip(correlation, scale > 0, strict=True)
        ],
        definition="Within-year cell anomalies, cosine-area pooled; constant fields have null correlation",
    )


def export_trajectory_diagnostics(output, origin, origin_directory):
    """Use existing annual exports, preserving the exact scored observation bins."""
    with np.load(output / f"members-{origin}.npz") as saved:
        ensemble = dict(saved)
    with np.load(output / f"{origin}.npz") as saved:
        reference = saved["reference"][:, :2]
    fields = ensemble["fields"]
    mask, lat, lon = (ensemble[key] for key in ("mask", "lat", "lon"))
    chosen = [5, 72]
    surface = fields[:, chosen, :2]
    truth = reference[chosen]
    support = mask[:2][None] & np.isfinite(truth)
    truth = np.where(support, truth, np.nan)
    mean, draw = surface.mean(axis=0), surface[0]
    description = json.loads((origin_directory / "COMPLETE.json").read_text())
    initial_records = []
    for record in description["records"][17:19]:
        path = origin_directory / record["file"]
        if (
            Path(record["file"]).name != record["file"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError("Initialization observation payload differs")
        with np.load(path) as saved:
            initial_records.append(saved["surface"])
    initial_truth = np.where(mask[:2][None], np.stack(initial_records), np.nan)
    np.savez_compressed(
        output / f"maps-{origin}.npz",
        leads_days=[30, 365],
        mean=np.where(support, mean, np.nan),
        draw1=np.where(support, draw, np.nan),
        reference=truth,
        mean_error=np.where(support, mean - truth, np.nan),
        draw1_error=np.where(support, draw - truth, np.nan),
        initial_mean=ensemble["initial"].mean(axis=0)[-1],
        initial_draw1=ensemble["initial"][0, -1],
        initial_channels=ensemble["channel_names"],
        initial_surface_reference=initial_truth[-1],
        initial_bin_days=np.array([-5, 0]),
        initial_scope="Latest of two decoded history slots; five-day bin ending at forecast origin",
        initial_surface_mean_error=ensemble["initial"].mean(axis=0)[-1, :2]
        - initial_truth[-1],
        initial_surface_draw1_error=ensemble["initial"][0, -1, :2] - initial_truth[-1],
        lat=lat,
        lon=lon,
    )
    result = dict(
        members=temporal_statistics(fields, mask, lat),
        reference_surface=temporal_statistics(reference[None], mask[:2], lat),
        channel_names=ensemble["channel_names"].tolist(),
        spectra={},
        uncertainty="Each member has its own evolving latent state; deterministic decoding",
    )
    for label, prediction in [
        (f"member{i + 1}", value) for i, value in enumerate(fields)
    ] + [("mean", fields.mean(0))]:
        for day, index in ((30, 5), (365, 72)):
            for channel, name in enumerate(("sst", "adt")):
                pred = np.where(
                    mask[channel], prediction[index : index + 1, channel], np.nan
                )
                target = np.where(
                    mask[channel], reference[index : index + 1, channel], np.nan
                )
                for region in spectra.SPATIAL_REGIONS:
                    curve = spatial_error(pred, target, lat, lon, region)
                    if curve is not None:
                        result["spectra"][f"{label}/{name}/{region[0]}/day{day}"] = (
                            curve
                        )
    atomic_json(result, output / f"trajectory-{origin}.json")


def validate_exposure(saved, arm, observation_updates, om4_updates):
    expected = {"observation": observation_updates, "om4": om4_updates}
    if saved["counts"] != expected or saved["step"] != sum(expected.values()):
        raise ValueError("Checkpoint does not match requested task exposure")
    if saved["contract"]["arm"] != arm:
        raise ValueError("Checkpoint arm differs from requested arm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "annual-data", "checkpoint", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("deterministic", "initial", "fresh", "ar1"), required=True
    )
    parser.add_argument("--expected-observation-updates", type=int, required=True)
    parser.add_argument("--expected-om4-updates", type=int, required=True)
    parser.add_argument("--origins", nargs="+", choices=ORIGINS, default=ORIGINS)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4041729)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.members < 2 or len(set(args.origins)) != len(args.origins):
        raise ValueError("Need multiple members and unique origins")
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_exposure(
        saved, args.arm, args.expected_observation_updates, args.expected_om4_updates
    )
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    data.use_observation_normalization()
    contract = saved["contract"]
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.reference),
    ):
        if digest(path) != contract[key]:
            raise ValueError(f"Changed input: {key}")
    origins = [
        args.annual_data / "test" / origin / "COMPLETE.json" for origin in args.origins
    ]
    protocol = dict(
        checkpoint_sha256=digest(args.checkpoint),
        training_contract=contract,
        counts=saved["counts"],
        step=saved["step"],
        evaluator=os.environ["SAMUDRA_CODE_COMMIT"],
        members=args.members,
        seed=args.seed,
        annual_manifests={p.parent.name: digest(p) for p in origins},
        skip_validation=args.skip_validation,
        sampling="Member-specific latent trajectories; deterministic physical decoder",
        domain="All finite wet latitudes; cosine area weighting; zero-normalized missing surface inputs",
        selection="Fixed-budget diagnostic; 2018 screen is part of final cohort, not independent holdout",
    )
    manifest = args.output / "input.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Existing evaluation has a different protocol")
    atomic_json(protocol, manifest)
    from samudra.experiments.stochastic_state_train import model_from_checkpoint

    model = model_from_checkpoint(saved, data).eval()
    if not args.skip_validation:
        directory = args.output / "validation"
        directory.mkdir(exist_ok=True)
        paths = data.paths("validation")
        recording = RecordingForecast(model, data, paths, directory).eval()
        with torch.no_grad():
            metrics, value = evaluate_point_metrics(
                recording,
                data,
                paths,
                json.loads(args.reference.read_text()),
                members=args.members,
                seed=args.seed,
            )
        atomic_json(
            dict(inputs=protocol, metrics=metrics, score=value),
            directory / "COMPLETE.json",
        )
    annual = args.output / "annual"
    annual.mkdir(exist_ok=True)
    annual_recording = AnnualMembers(
        model, data, origins, annual, members=args.members, seed=args.seed
    ).eval()
    evaluate_origins(annual_recording, data, origins, annual, protocol)
    for origin in args.origins:
        export_trajectory_diagnostics(
            annual, origin, args.annual_data / "test" / origin
        )
    atomic_json(
        dict(
            inputs=protocol,
            files={
                str(path.relative_to(args.output)): digest(path)
                for path in sorted(args.output.rglob("*"))
                if path.is_file() and path != args.output / "COMPLETE.json"
            },
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
