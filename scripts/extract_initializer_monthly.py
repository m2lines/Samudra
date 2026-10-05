#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export calendar-weighted, independently initialized December T/S means."""

import argparse
import hashlib
import importlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

from samudra.constants import build_om4_layout
from samudra.experiments.observation_checkpoint import load_fixed_checkpoint
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_training import Samples


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--extra-data", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    if os.environ.get("SAMUDRA_CODE_COMMIT") != contract["producer"]:
        raise ValueError("Original producer environment is not pinned")
    runtime = {}
    for name, expected in contract["module_sha256"].items():
        module_path = importlib.import_module(name).__file__
        if module_path is None:
            raise ValueError(f"Runtime module has no source file: {name}")
        path = Path(module_path)
        if digest(path) != expected or not str(path).startswith(
            "/opt/samudra-code/src/"
        ):
            raise ValueError(f"Runtime source mismatch: {name}")
        runtime[name] = {"path": str(path), "sha256": expected}
    if args.output.exists():
        raise ValueError("Preserve previous exports; choose a new output")
    args.output.mkdir(parents=True)
    preparation = json.loads((args.extra_data / "COMPLETE.json").read_text())
    extra = args.extra_data / "2014-12.npz"
    if (
        digest(extra) != preparation["output_sha256"]
        or extra.stat().st_size != preparation["output_bytes"]
    ):
        raise ValueError("Extra December sample differs from preparation receipt")
    if digest(args.data / "grid.npz") != preparation["grid_sha256"]:
        raise ValueError("Alpha and Torch grids differ")
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    data.use_observation_normalization()
    layout = build_om4_layout()
    np.savez_compressed(
        args.output / "grid.npz",
        lat=data.grid["lat"],
        lon=data.grid["lon"],
        depths=np.array(layout.depth_levels)[:14],
        mask=data.grid["mask"][data.ts_indices].reshape(2, 14, 180, 360),
    )
    provenance = {
        "producer": contract["producer"],
        "runtime": runtime,
        "script_sha256": digest(__file__),
        "contract": contract,
        "sources": {
            str(args.contract): digest(args.contract),
            str(args.extra_data / "COMPLETE.json"): digest(
                args.extra_data / "COMPLETE.json"
            ),
            str(args.data / "grid.npz"): digest(args.data / "grid.npz"),
            str(args.data / "statistics.npz"): digest(args.data / "statistics.npz"),
        },
        "stages": {},
        "months": {},
        "definition": "Original reconstruct_month operator, latest state from each independently initialized 95-day history, seven December bins weighted 5/31 six times and 1/31 once. Last bin December31-January5 exclusive includes January1-4 surface/forcing data. Retrospective reconstruction, no processor autoregression or new training. December anomalies use training-only 1993-2012 climatology; global common finite reference/model support.",
    }
    total = np.zeros((2, 14, 180, 360), dtype=np.float64)
    count = np.zeros_like(total)
    training = sorted((args.data / "train").glob("*-12.npz"))
    if len(training) != 20 or [p.stem for p in training] != [
        f"{year}-12" for year in range(1993, 2013)
    ]:
        raise ValueError("December climatology cohort differs")
    for path in training:
        values = np.load(path)["interior"]
        finite = np.isfinite(values)
        total += np.where(finite, values, 0)
        count += finite
        provenance["sources"][str(path)] = digest(path)
    climate = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    np.savez_compressed(
        args.output / "december-climatology.npz", ts=climate, count=count
    )
    for stage, spec in contract["stages"].items():
        run = Path(spec["run"])
        manifest = json.loads((run / "manifest.json").read_text())
        if (
            not manifest["arguments"]["global_observations"]
            or manifest["arguments"]["surface_fill"] != "zero"
        ):
            raise ValueError("Model input coverage/fill differs")
        if not (
            manifest["arguments"].get("from_scratch")
            or manifest["arguments"].get("observation_normalization")
        ):
            raise ValueError("Observation normalization not declared")
        state, lineage = load_fixed_checkpoint(
            run,
            spec["checkpoint"],
            "test",
            spec["om4_updates"],
            spec["observation_updates"],
        )
        if (
            lineage["checkpoint_sha256"] != spec["checkpoint_sha256"]
            or lineage["training_manifest_sha256"] != spec["manifest_sha256"]
        ):
            raise ValueError("Checkpoint differs from published endpoint")
        model = (
            ObservationTransfer.from_arguments(
                data.grid["names"].tolist(), manifest["arguments"]
            )
            .cuda()
            .eval()
        )
        model.load_state_dict(state, strict=True)
        del state
        provenance["stages"][stage] = lineage
        for month in ["2014-12", "2017-12", "2020-12"]:
            path = (
                extra if month == "2014-12" else args.data / "test" / (month + ".npz")
            )
            sample = data.load(path)
            weights = sample["raw"]["month_weights"]
            np.testing.assert_allclose(weights, np.array([5] * 6 + [1]) / 31, rtol=1e-7)
            np.testing.assert_equal(sample["raw"]["surface"].shape, (26, 2, 180, 360))
            individual = []
            processor_calls = []

            def capture(module, inputs, output):
                individual.append(
                    output[:, -1, data.ts_indices].detach().float().cpu().numpy()
                )

            def forbid_processor(module, inputs):
                processor_calls.append(1)
                raise ValueError("Monthly reconstruction unexpectedly calls processor")

            hook = model.initializer.register_forward_hook(capture)
            guard = model.evolution.register_forward_pre_hook(forbid_processor)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = model.reconstruct_month(
                    sample["surface"],
                    sample["atmosphere"],
                    sample["contexts"],
                    data.mask,
                    sample["validity"],
                    sample["month_weights"],
                )
            hook.remove()
            guard.remove()
            if len(individual) != 7 or processor_calls:
                raise ValueError("Incorrect initializer call count")
            independent_mean = sum(
                value * weight
                for value, weight in zip(individual, weights, strict=True)
            )
            actual = normalized[:, data.ts_indices].float().cpu().numpy()
            np.testing.assert_allclose(actual, independent_mean, rtol=1e-5, atol=1e-5)
            physical = (
                data.physical(normalized[:, None])[0, 0, data.ts_indices]
                .cpu()
                .numpy()
                .reshape(2, 14, 180, 360)
            )
            if not np.isfinite(
                physical[data.grid["mask"][data.ts_indices].reshape(2, 14, 180, 360)]
            ).all():
                raise ValueError("Nonfinite monthly state")
            np.savez_compressed(
                args.output / (stage + "-" + month + ".npz"), ts=physical
            )
            reference = args.output / (month + "-reference.npz")
            if not reference.exists():
                np.savez_compressed(reference, ts=sample["raw"]["interior"])
            else:
                np.testing.assert_array_equal(
                    np.load(reference)["ts"], sample["raw"]["interior"]
                )
            provenance["sources"][str(path)] = digest(path)
            provenance["months"].setdefault(month, {})[stage] = {
                "initializer_calls": len(individual),
                "processor_calls": len(processor_calls),
                "independent_sum_max_absolute_difference_normalized": float(
                    np.max(np.abs(actual - independent_mean))
                ),
                "reconstruction_loss": float(data.interior_loss(normalized, sample)),
                "weights": weights.tolist(),
                "midpoints": sample["raw"]["midpoints"].tolist(),
            }
            print("MONTHLY_INITIALIZER_COMPLETE", stage, month, flush=True)
            individual.clear()
            del normalized, sample
        del model
        torch.cuda.empty_cache()
    provenance["files"] = {
        p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in args.output.glob("*.npz")
    }
    (args.output / "COMPLETE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    archive = args.output.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(args.output, arcname="monthly")
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size}
    archive.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("MONTHLY_EXPORT_COMPLETE", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
