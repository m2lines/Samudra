# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation-only completion audit on hidden known observations, not polar truth."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.experiments.missingness import structured_visibility
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def scored_errors(prediction, target, valid, latitude):
    """Physical-unit errors; absent support remains null, never a perfect score."""
    area = np.cos(np.deg2rad(latitude))[None, :, None]
    result = {}
    for field, name in enumerate(("sst", "ssh")):
        for region, rows in (
            ("global", np.ones(len(latitude), bool)),
            ("tropics_midlatitudes", np.abs(latitude) <= 60),
            ("north_polar", latitude > 60),
            ("south_polar", latitude < -60),
        ):
            support = valid[:, field] & rows[None, :, None]
            if not np.isfinite(target[:, field][support]).all():
                raise ValueError("Nonfinite target on completion scoring support")
            error = np.where(support, prediction[:, field] - target[:, field], 0)
            weight = support * area
            denominator = weight.sum()
            result[name + "/" + region] = {
                "count": int(support.sum()),
                "area_weight": float(denominator),
                "rmse": float(np.sqrt((error**2 * weight).sum() / denominator))
                if denominator > 0
                else None,
                "bias": float((error * weight).sum() / denominator)
                if denominator > 0
                else None,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["normalization_mode"] != "observation-only":
        raise ValueError("Completion audit requires shared observation scaling")
    options = manifest["arguments"]
    data = Samples(options["data"], "cuda", options.get("surface_fill", "climatology"))
    data.use_observation_normalization()
    model = ObservationTransfer.from_arguments(data.grid["names"].tolist(), options)
    model = model.cuda().eval()
    paths = data.paths("validation")
    if len(paths) != 9:
        raise ValueError("Incomplete fixed validation cohort")
    signature = {
        "manifest_sha256": digest(run / "manifest.json"),
        "checkpoints": {name: digest(run / name) for name in args.checkpoints},
        "origins": [p.stem for p in paths],
        "mask_seed": 918271,
        "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "scope": "Validation diagnostics; artificial gaps retain genuine labels. Naturally missing cells have no accuracy claim. No future inputs or checkpoint selection.",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "input.json").exists() and json.loads(
        (output / "input.json").read_text()
    ) != signature:
        raise ValueError("Completion audit input differs")
    atomic_json(signature, output / "input.json")
    if (output / "COMPLETE.json").exists():
        return
    latitude = data.grid["lat"]
    scale = data.grid["std"][[38, 76], None, None]
    mean = data.grid["mean"][[38, 76], None, None]
    with torch.inference_mode():
        for checkpoint in args.checkpoints:
            saved = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            counts = saved.get("task_counts")
            del saved
            records = []
            for index, path in enumerate(paths):
                sample = data.load(path)
                original = sample["validity"][:, :19].bool()
                for pattern in ("natural", "blocks", "polar_caps"):
                    visible = original.clone()
                    if pattern == "blocks":
                        visible = structured_visibility(original, 918271 + index)
                    elif pattern == "polar_caps":
                        visible[..., np.abs(latitude) > 60, :] = False
                    surface = sample["surface"][:, :19]
                    if pattern != "natural":
                        surface = torch.where(visible, surface, 0)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        initial = model.initialize(
                            surface,
                            sample["atmosphere"][:, :19],
                            sample["contexts"][:, 18],
                            data.mask,
                            visible,
                        )
                    if not torch.equal(
                        initial[:, :, [38, 76]][visible[:, -2:]],
                        surface[:, -2:][visible[:, -2:]],
                    ):
                        raise ValueError("Initializer changed available observations")
                    physical = data.physical(initial)[0].cpu().numpy()
                    if not np.isfinite(physical).all():
                        raise ValueError("Nonfinite initialized state")
                    target = sample["raw"]["surface"][17:19]
                    hidden = (original[:, -2:] & ~visible[:, -2:])[0].cpu().numpy()
                    months = (
                        pd.DatetimeIndex(sample["raw"]["midpoints"][17:19]).month - 1
                    )
                    climo = data.stats["surface_climatology"][months]
                    prediction = physical[:, [38, 76]]
                    record = {
                        "origin": path.stem,
                        "pattern": pattern,
                        "model": scored_errors(prediction, target, hidden, latitude),
                        "climatology": scored_errors(climo, target, hidden, latitude),
                        "normalized_zero": scored_errors(
                            np.broadcast_to(mean, target.shape),
                            target,
                            hidden,
                            latitude,
                        ),
                        "unweighted_normalized_completion_mse": float(
                            (
                                (
                                    (prediction - np.where(hidden, target, prediction))
                                    / scale
                                )
                                ** 2
                            )[hidden].mean()
                        )
                        if hidden.any()
                        else None,
                        "natural_missing_cells": int(
                            (~original & data.mask[[38, 76]].bool()).sum()
                        ),
                    }
                    records.append(record)
                    if index in (0, len(paths) - 1):
                        np.savez_compressed(
                            output
                            / f"{Path(checkpoint).stem}-{path.stem}-{pattern}.npz",
                            initial=physical[-1, [38, 47, 57, 66, 0, 19, 76]],
                            channel_names=data.grid["names"][
                                [38, 47, 57, 66, 0, 19, 76]
                            ],
                            target=target[-1],
                            visible=visible[0, -1].cpu().numpy(),
                            artificially_hidden=hidden[-1],
                            natural_validity=original[0].cpu().numpy(),
                            lat=latitude,
                            lon=data.grid["lon"],
                        )
            atomic_json(
                {
                    "checkpoint_sha256": signature["checkpoints"][checkpoint],
                    "task_counts": counts,
                    "records": records,
                },
                output / (Path(checkpoint).stem + ".json"),
            )
            print(
                json.dumps(
                    {"event": "completion_audit_checkpoint", "checkpoint": checkpoint}
                ),
                flush=True,
            )
    atomic_json(signature, output / "COMPLETE.json")


if __name__ == "__main__":
    main()
