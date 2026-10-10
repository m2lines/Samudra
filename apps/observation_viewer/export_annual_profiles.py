#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Retain monthly temperatures from audited, frozen annual model rollouts."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def differences(a, b):
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError("Replay comparison shape or support differs")
    delta = (a.astype(np.float64) - b.astype(np.float64))[np.isfinite(a)]
    return dict(
        rms=float(np.sqrt(np.mean(delta**2))), max_abs=float(np.max(np.abs(delta)))
    )


def run(root, stage, index=None):
    cfg = read(root / "paths.json")
    signature = dict(
        config_sha256=sha(root / "paths.json"), script_sha256=sha(__file__)
    )
    if signature["script_sha256"] != cfg["exporter_sha256"]:
        raise ValueError("Exporter changed")
    for name, expected in cfg["source_files"].items():
        if sha(root / "source" / name) != expected:
            raise ValueError("Frozen source differs: " + name)
    if sha(root / "runtime-contract.json") != cfg["runtime_sha256"]:
        raise ValueError("Runtime contract changed")
    runtime = read(root / "runtime-contract.json")
    for name, spec in runtime["modules"].items():
        module = importlib.import_module(name)
        if getattr(module, "__version__", None) != spec["version"]:
            raise ValueError("Runtime version differs: " + name)
        for path, expected in spec["binaries"].items():
            if sha(path) != expected:
                raise ValueError("Runtime binary changed: " + path)
    annual = importlib.import_module("samudra.experiments.observation_annual")
    training = importlib.import_module("samudra.experiments.observation_training")
    models = importlib.import_module("samudra.experiments.observation_model")
    torch = importlib.import_module("torch")
    meta = read(root / "base/catalog.json")
    if sha(root / "base/catalog.json") != cfg["base_catalog_sha256"]:
        raise ValueError("Viewer catalog changed")
    if cfg["origins"] != meta["origins"]:
        raise ValueError("Origin cohort changed")
    for key in ["interior_mask", "heat_mask", "surface_mask"]:
        filename = meta[key]
        if sha(root / "base" / filename) != meta["files"][filename]["sha256"]:
            raise ValueError("Viewer mask changed")
    if stage == "audit":
        if sha(cfg["image"]) != runtime["image_sha256"]:
            raise ValueError("Container image differs")
        if sha(root / "source.tar") != cfg["source_archive_sha256"]:
            raise ValueError("Frozen source archive changed")
        for row in cfg["models"]:
            folder = Path(row["run"])
            manifest = read(folder / "manifest.json")
            lineage = meta["models"][row["name"]]["lineage"]
            expected = lineage["checkpoint_sha256"]
            if {
                sha(folder / "best.pt"),
                read(folder / "best.json")["checkpoint_sha256"],
                read(folder / "TRAIN_COMPLETE.json")["best_checkpoint_sha256"],
                row["checkpoint_sha256"],
            } != {expected}:
                raise ValueError("Selected weights differ: " + row["name"])
            if sha(folder / "manifest.json") != lineage["training_manifest_sha256"]:
                raise ValueError("Training manifest changed")
            annual.verify_sample_root(cfg["observations"], manifest)
        if not (Path(cfg["annual_data"]) / "DATA_READY.json").is_file():
            raise ValueError("Annual data was not verified")
        origins = sorted((Path(cfg["annual_data"]) / "test").glob("*/COMPLETE.json"))
        if [p.parent.name for p in origins] != cfg["origins"]:
            raise ValueError("Annual test cohort differs")
        for path in origins:
            description, raw = annual.read_origin(path.parent)
            if not np.isfinite(raw["atmosphere"]).all():
                raise ValueError("Nonfinite prescribed forcing")
            for item in description["monthly_interiors"]:
                p = path.parent / item["file"]
                if p.name != item["file"] or sha(p) != item["sha256"]:
                    raise ValueError("Monthly reference differs")
            print(
                json.dumps(dict(event="profile_data_audited", origin=path.parent.name)),
                flush=True,
            )
        write_json(root / "audit-output/AUDITED.json", signature)
        return
    if read(root / "audit-output/AUDITED.json") != signature:
        raise ValueError("Profile audit contract differs")
    row = cfg["models"][index]
    name, folder = row["name"], Path(row["run"])
    lineage = meta["models"][name]["lineage"]
    if (
        sha(folder / "best.pt") != lineage["checkpoint_sha256"]
        or sha(folder / "manifest.json") != lineage["training_manifest_sha256"]
    ):
        raise ValueError("Selected model changed after audit")
    output = root / "exports" / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(folder / "manifest.json")
    data = training.Samples(
        cfg["observations"],
        "cuda",
        manifest["arguments"].get("surface_fill", "climatology"),
        global_observations=True,
    )
    if manifest["normalization_mode"] == "observation-only":
        data.use_observation_normalization()
    for field in ["mean", "std"]:
        np.testing.assert_array_equal(data.grid[field], manifest["effective_" + field])
    for field in ["lat", "lon"]:
        np.testing.assert_array_equal(data.grid[field], meta[field])
    indices = [data.grid["names"].tolist().index(f"thetao_{i}") for i in range(14)]
    mask = data.grid["mask"][indices].astype(bool)
    np.testing.assert_array_equal(
        mask, np.load(root / "base" / meta["interior_mask"])[:, 0]
    )
    heat_mask = np.load(root / "base" / meta["heat_mask"])
    surface_mask = np.load(root / "base" / meta["surface_mask"])
    model = (
        models.ObservationTransfer.from_arguments(
            data.grid["names"].tolist(), manifest["arguments"]
        )
        .cuda()
        .eval()
    )
    state = torch.load(folder / "best.pt", map_location="cuda", weights_only=False)[
        "model"
    ]
    model.load_state_dict(state, strict=True)
    del state
    receipt = dict(
        **signature,
        lineage=lineage,
        model=name,
        source_producer=cfg["source_producer"],
        exporter_producer=cfg["exporter_producer"],
        runtime=dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            autocast="bfloat16",
        ),
        files={},
        cases={},
    )
    for origin in cfg["origins"]:
        print(
            json.dumps(dict(event="profile_origin_start", model=name, origin=origin)),
            flush=True,
        )
        directory = Path(cfg["annual_data"]) / "test" / origin
        description, raw = annual.read_origin(directory)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, initial = model.forecast(*annual.inputs(data, raw))
        if prediction.shape[1] != 73 or not torch.isfinite(prediction).all():
            raise ValueError("Incomplete or nonfinite annual forecast")
        temperature, reference, heats, months = [], [], [], []
        for item in description["monthly_interiors"]:
            p = directory / item["file"]
            if p.name != item["file"] or sha(p) != item["sha256"]:
                raise ValueError("Monthly target hash/path differs")
            month = p.name[:7]
            weights = data.tensor(annual.month_weights(raw["start"][19:], month))
            monthly = (prediction * weights[None, :, None, None, None]).sum(1)
            physical = data.physical(monthly[:, None])[0, 0, indices].cpu().numpy()
            temperature.append(np.where(mask, physical, np.nan).astype(np.float32))
            heats.append(np.where(heat_mask, data.ohc(monthly)[0] / 1e9, np.nan))
            with np.load(p, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["depth"], meta["depths"])
                if str(saved["time"]) != month:
                    raise ValueError("Monthly timestamp differs")
                reference.append(
                    np.where(mask, saved["values"][0], np.nan).astype(np.float32)
                )
            months.append(month)
        if months != meta["records"][origin]["heat_months"]:
            raise ValueError("Calendar months differ")
        surface = data.physical(prediction)[0, :, [38, 76]].cpu().numpy()
        held = data.physical(initial)[0, -1, indices].cpu().numpy()
        arrays = dict(
            temperature=np.array(temperature),
            reference=np.array(reference),
            heat=np.array(heats),
            months=np.array(months),
            temperature_persistence=np.where(mask, held, np.nan).astype(np.float32),
            heat_persistence=np.where(
                heat_mask, data.ohc(initial[:, -1])[0] / 1e9, np.nan
            ),
        )
        if not np.isfinite(arrays["temperature"][:, mask]).all():
            raise ValueError("Nonfinite wet temperature")
        old_path = Path(row["annual_root"]) / name / f"{origin}.npz"
        old_complete = read(Path(row["annual_root"]) / name / "COMPLETE.json")
        if old_complete["inputs"]["checkpoint_sha256"] != lineage["checkpoint_sha256"]:
            raise ValueError("Original annual model differs")
        with np.load(old_path, allow_pickle=False) as old:
            case = dict(
                original_sha256=sha(old_path),
                heat_difference_from_report_gj_m2=differences(
                    arrays["heat"],
                    np.where(
                        heat_mask, old["predicted_ohc"].astype(np.float64) / 1e9, np.nan
                    ),
                ),
                surface_difference_from_report={
                    variable: differences(
                        np.where(surface_mask[i], surface[:, i], np.nan),
                        np.where(surface_mask[i], old["surface"][:, i], np.nan),
                    )
                    for i, variable in enumerate(["sst_celsius", "ssh_meters"])
                },
            )
        file = output / f"{origin}.npz"
        np.savez_compressed(file, **arrays)
        with np.load(file, allow_pickle=False) as stored:
            for key, values in arrays.items():
                np.testing.assert_array_equal(stored[key], values)
        receipt["files"][file.name] = dict(sha256=sha(file), bytes=file.stat().st_size)
        receipt["cases"][origin] = case
        print(
            json.dumps(
                dict(
                    event="profile_origin_complete",
                    model=name,
                    origin=origin,
                    comparison=case,
                )
            ),
            flush=True,
        )
        del prediction, initial
    write_json(output / "COMPLETE.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["audit", "evaluate"], required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.stage == "evaluate" and args.index is None:
        parser.error("Evaluation requires a model index")
    run(args.root, args.stage, args.index)
