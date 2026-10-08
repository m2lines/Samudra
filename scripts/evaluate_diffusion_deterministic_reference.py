#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Point-mass CRPS using the matched deterministic branch's own model loader.

Run with PYTHONPATH pointing to the immutable deterministic source snapshot.
The metric helper is loaded explicitly from this experiment's source snapshot.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def serial(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--metric-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_input = json.loads((args.previous / "evaluation-input.json").read_text())
    lineage = previous_input["fixed_budget_lineage"]
    if lineage["task_counts"] != {"om4": 8000, "observation": 8000}:
        raise ValueError("Need matched deterministic 8k/8k endpoint")
    checkpoint = args.run / lineage["checkpoint"]
    if digest(checkpoint) != previous_input["checkpoint_sha256"]:
        raise ValueError("Deterministic checkpoint fingerprint differs")
    manifest = json.loads((args.run / "manifest.json").read_text())
    if digest(args.run / "manifest.json") != lineage["training_manifest_sha256"]:
        raise ValueError("Deterministic training manifest differs")
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    if digest(data.root / "SHA256SUMS") != previous_input["data_manifest_sha256"]:
        raise ValueError("Deterministic observation data differs")
    data.use_observation_normalization()
    # This factory belongs to the independently versioned deterministic branch.
    model_factory = getattr(ObservationTransfer, "from_arguments")
    model = model_factory(data.grid["names"].tolist(), manifest["arguments"]).cuda()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    model.eval()
    spec = importlib.util.spec_from_file_location("point_metrics", args.metric_helper)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the specified metric helper")
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    point = metric_module.point_field_statistics
    with np.load(args.previous / "fixed-budget-predictions.npz") as saved:
        prior_prediction = saved["prediction"]
        prior_reference = saved["reference"][:, :, :2]
        prior_origins = saved["origins"].tolist()
    paths = data.paths("test")
    if len(paths) != 96 or [p.stem for p in paths] != prior_origins:
        raise ValueError("Deterministic held-out cohort differs")
    records, differences = [], []
    for index, path in enumerate(paths):
        sample = data.load(path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model.forecast(
                sample["surface"],
                sample["atmosphere"],
                sample["contexts"],
                data.mask,
                sample["validity"],
            )
        physical = data.physical(prediction)
        monthly = (physical * sample["month_weights"][None, :, None, None, None]).sum(1)
        surface = physical[:, :, [38, 76]]
        np.testing.assert_array_equal(
            sample["raw_surface"][0].cpu().numpy(), prior_reference[index]
        )
        differences.append(
            float(np.max(np.abs(surface[0].cpu().numpy() - prior_prediction[index])))
        )
        records.append(
            serial(
                dict(
                    origin=path.stem,
                    surface=point(
                        surface,
                        sample["raw_surface"],
                        data.mask[[38, 76]],
                        data.area,
                        data.surface_scale,
                    ),
                    interior=point(
                        monthly[:, data.ts_indices],
                        sample["interior"],
                        data.ts_mask,
                        data.area,
                        data.ts_scale,
                    ),
                )
            )
        )
        atomic_json(records, args.output / "point-statistics.json")
        print(
            json.dumps(
                dict(event="point_reference_origin", completed=index + 1, total=96)
            ),
            flush=True,
        )
    atomic_json(
        dict(
            evaluator=os.environ["SAMUDRA_CODE_COMMIT"],
            baseline_model_producer=os.environ["BASELINE_CODE_COMMIT"],
            checkpoint_sha256=digest(checkpoint),
            data_manifest_sha256=digest(data.root / "SHA256SUMS"),
            metric_helper_sha256=digest(args.metric_helper),
            lineage=lineage,
            previous_surface_max_absolute_difference=max(differences),
            previous_surface_per_origin_max_absolute_difference=differences,
            point_statistics_sha256=digest(args.output / "point-statistics.json"),
            scope="Global point-mass CRPS (MAE) and MSE on exactly the diffusion report's standardized observation support",
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
