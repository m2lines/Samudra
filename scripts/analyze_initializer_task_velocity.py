#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen mixed initializer: crossed data bundles/task adapters and OM4 losses."""

import argparse
import hashlib
import importlib
import json
import math
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_checkpoint import load_fixed_checkpoint
from samudra.experiments.observation_joint import load_om4
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season, balanced_loss, channel_mse
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def breakdown(prediction, truth, weights, names, interior):
    per_channel = channel_mse(prediction, truth, weights).mean((0, 1))
    groups = {}
    for variable in ["thetao", "so", "uo", "vo", "zos"]:
        ids = [
            i
            for i, n in enumerate(names)
            if (n == variable or n.startswith(variable + "_"))
            and not (interior and n in ["thetao_0", "zos"])
        ]
        if ids:
            groups[variable] = float(per_channel[ids].mean())
    total = sum(groups.values()) / len(groups)
    torch.testing.assert_close(
        torch.tensor(total),
        balanced_loss(prediction, truth, weights, names, interior).cpu(),
    )
    return {
        "group_mse": groups,
        "balanced_loss": total,
        "per_channel_mse": per_channel.cpu().tolist(),
    }


def summary(value, target, mask, area):
    valid = mask.bool() & torch.isfinite(target)
    w = valid * area
    err = torch.where(valid, value - target, 0)

    def avg(z):
        return float((torch.where(valid, z, 0) * w).sum() / w.sum())

    vx = avg(value)
    vy = avg(target)
    variance_x = avg((value - vx) ** 2)
    variance_y = avg((target - vy) ** 2)
    return {
        "rmse_m_s": math.sqrt(avg(err**2)),
        "bias_m_s": avg(err),
        "rms_m_s": math.sqrt(avg(value**2)),
        "spatial_correlation": avg((value - vx) * (target - vy))
        / math.sqrt(variance_x * variance_y)
        if variance_x * variance_y > 0
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / "output"
    if out.exists():
        raise ValueError("Preserve earlier outputs")
    out.mkdir()
    contract = json.loads((root / "contract.json").read_text())
    assert os.environ["SAMUDRA_CODE_COMMIT"] == contract["producer"]
    for module, digest in contract["module_sha256"].items():
        filename = importlib.import_module(module).__file__
        assert filename is not None and sha(filename) == digest
    spec = contract["stages"]["obs08000"]
    run = Path(spec["run"])
    manifest = json.loads((run / "manifest.json").read_text())
    state, lineage = load_fixed_checkpoint(run, spec["checkpoint"], "test", 8000, 8000)
    assert lineage["checkpoint_sha256"] == spec["checkpoint_sha256"]
    assert lineage["training_manifest_sha256"] == spec["manifest_sha256"]
    options = SimpleNamespace(**manifest["arguments"])
    options.output = str(root / "om4-loader")
    options.name = "velocity-task-diagnostic"
    om4 = load_om4(options)
    data = Samples(options.data, "cuda", surface_fill="zero", global_observations=True)
    data.use_observation_normalization()
    torch.testing.assert_close(om4.mask.bool(), data.mask.bool())
    model = (
        ObservationTransfer.from_arguments(
            data.grid["names"].tolist(), manifest["arguments"]
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(state, strict=True)
    del state
    source = om4.context_bundle.inference_source
    assert source is not None
    source = native_om4_source(
        source,
        LocalLocation(path=Path(options.om4_data) / "OM4.zarr"),
        create_rust_io_runtime(4),
    )
    dataset = om4.dataset(source)
    source_times = pd.DatetimeIndex([str(t) for t in source.time.values])
    names = om4.names
    assert names[0] == "uo_0" and names[19] == "vo_0"
    n = len(names)
    shape = tuple(data.mask.shape[-2:])
    meta = {
        "lineage": lineage,
        "producer": contract["producer"],
        "script_sha256": sha(__file__),
        "cases": {},
        "sources": {},
        "velocity_scales": om4.std[:38].cpu().tolist(),
        "native_velocity_scales": om4.native_std[:38].cpu().tolist(),
    }
    for path in [
        run / "manifest.json",
        Path(options.data) / "grid.npz",
        Path(options.data) / "statistics.npz",
        Path(options.om4_data) / "OM4.zarr/.zmetadata",
        Path(options.om4_data) / "OM4_means.zarr/.zmetadata",
        Path(options.om4_data) / "OM4_stds.zarr/.zmetadata",
    ]:
        meta["sources"][str(path)] = sha(path)
    np.savez_compressed(
        out / "grid.npz",
        lat=data.grid["lat"],
        lon=data.grid["lon"],
        mask=data.grid["mask"][[0, 19]],
    )
    for origin in ["2015-01-01", "2018-01-01"]:
        sample_path = Path(options.data) / "test" / (origin[:7] + ".npz")
        sample = data.load(sample_path)
        desired = pd.DatetimeIndex(sample["raw"]["midpoints"][:19])
        matches = np.flatnonzero(source_times == desired[-1])
        assert len(matches) == 1
        index = int(matches[0]) - 18
        np.testing.assert_array_equal(
            source_times[index : index + 19].values, desired.values
        )
        loader = om4.native_loader(dataset, [[index]])
        batch = next(iter(loader))
        history, boundary = batch.get_initial_input()
        history = history.reshape(1, 19, n, *shape)
        past = boundary.reshape(1, 19, 3, *shape)
        native_mean = om4.native_mean[None, None, :, None, None]
        native_std = om4.native_std[None, None, :, None, None]

        def convert(value):
            return (
                (value * native_std + native_mean - data.mean) / data.std
            ) * data.mask

        truth = convert(history[:, -2:])
        surface = convert(history)[:, :, model.initializer.surface]
        labels = convert(
            torch.stack([batch.get_label(i) for i in range(len(batch))], 1)
        )
        forcing = torch.stack(
            [batch.get_input(i)[1][:, -3:] for i in range(len(batch))], 1
        )
        t = source.time.values[index + 18]
        phase = 2 * math.pi * (t.dayofyr - 1) / 365.25
        season = data.tensor([math.sin(phase), math.cos(phase)])[
            None, :, None, None
        ].expand(1, 2, *shape)
        context = torch.cat((om4.geo[None], season), 1)
        bundles = {
            "om4": (
                surface,
                past,
                context,
                data.mask,
                data.mask[model.initializer.surface].expand_as(surface),
            ),
            "obs": (
                sample["surface"][:, :19],
                model.adapt(sample["atmosphere"][:, :19]).detach(),
                sample["contexts"][:, 18],
                data.mask,
                sample["validity"][:, :19],
            ),
        }
        outputs = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Match the original observation evaluator's adapter precision.
            bundles["obs"] = (
                bundles["obs"][0],
                model.adapt(sample["atmosphere"][:, :19]),
                *bundles["obs"][2:],
            )
            for source_name, inputs in bundles.items():
                for task in ["om4", "observation"]:
                    outputs[source_name + "_" + task] = model.initializer(
                        *inputs, task=task
                    )
            initial = outputs["om4_om4"]
            states = initial
            predicted = []
            for lead in range(1, 7):
                value = model.evolution(
                    states,
                    forcing,
                    advance_season(context, (lead - 1) * 5),
                    data.mask,
                    lead,
                    "om4",
                )
                predicted.append(value)
                states = torch.stack((states[:, -1], value), 1)
            forecast = torch.stack(predicted, 1)
        record = {
            "history_midpoints": [str(t) for t in desired],
            "surface": {},
            "reconstruction": breakdown(initial, truth, om4.weights, names, True),
            "forecast": breakdown(forecast, labels, om4.weights, names, False),
        }
        physical = {key: data.physical(v)[0, -1] for key, v in outputs.items()}
        target = data.physical(truth)[0, -1]
        for key, value in physical.items():
            record["surface"][key] = {
                names[c]: summary(value[c], target[c], data.mask[c], data.area)
                for c in [0, 19]
            }
        record["surface"]["zero_current"] = {
            names[c]: summary(
                torch.zeros_like(target[c]), target[c], data.mask[c], data.area
            )
            for c in [0, 19]
        }
        saved_path = Path(
            "/scratch/jr7309/runs/2026-09-29-observation-global/global-endpoint-annual"
        ) / (origin + ".npz")
        saved = np.load(saved_path)["initial"][-1]
        actual = physical["obs_observation"].cpu().numpy()
        delta = actual - saved
        record["reproduction_max_absolute_difference"] = float(np.abs(delta).max())
        np.testing.assert_allclose(actual, saved, rtol=1e-4, atol=1e-4)
        meta["sources"][str(saved_path)] = sha(saved_path)
        for phase_key, coefficient in [("forecast", 1.0), ("reconstruction", 0.1)]:
            groups = record[phase_key]["group_mse"]
            record[phase_key]["weighted_contribution"] = {
                key: coefficient * value / len(groups) for key, value in groups.items()
            }
        meta["cases"][origin] = record
        meta["sources"][str(sample_path)] = sha(sample_path)
        arrays = {key: value[[0, 19]].cpu().numpy() for key, value in physical.items()}
        arrays["om4_target"] = target[[0, 19]].cpu().numpy()
        np.savez_compressed(out / (origin + ".npz"), **arrays)
        print("CASE_COMPLETE", origin, json.dumps(record["surface"]), flush=True)
        del batch, loader, history, past, labels, forecast, outputs, physical, bundles
    meta["files"] = {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.glob("*.npz")
    }
    (out / "COMPLETE.json").write_text(json.dumps(meta, indent=2) + "\n")
    archive = root / "output.tar"
    with tarfile.open(archive, "w") as t:
        t.add(out, arcname="output")
    (root / "receipt.json").write_text(
        json.dumps({"sha256": sha(archive), "bytes": archive.stat().st_size}) + "\n"
    )
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
