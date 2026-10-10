#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Add verified frozen annual models to an immutable viewer bundle; no inference."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from export_interior import sha
from model_info import EARLY_REPORT, MODEL_DETAILS, PRESENTATION_REPORT


def extend(base, source, output):
    if output.exists():
        raise ValueError("Use a new bundle path; preserve the current deployment")
    inventory = json.loads((source / "inventory.json").read_text())
    meta = json.loads((base / "catalog.json").read_text())
    selected = inventory["models"]
    if not selected or any(n.endswith("-cooldown") for n in selected):
        raise ValueError("Expected only the selected non-cooldown models")
    if set(selected) - MODEL_DETAILS.keys() or set(selected) & meta["models"].keys():
        raise ValueError("Unknown or duplicate model")

    def path(remote):
        relative = Path(remote).relative_to("/")
        if ".." in relative.parts:
            raise ValueError("Invalid source path")
        return source / relative

    # Full read-back verification, including the base bundle before copying it.
    for remote, info in inventory["files"].items():
        p = path(remote)
        if p.stat().st_size != info["bytes"] or sha(p) != info["sha256"]:
            raise ValueError(f"Source checksum differs: {remote}")
    for name, info in meta["files"].items():
        if Path(name).name != name or sha(base / name) != info["sha256"]:
            raise ValueError("Base checksum differs: " + name)
    surface_mask = np.load(base / meta["surface_mask"])
    interior_mask = np.load(base / meta["interior_mask"])
    heat_mask = np.load(base / meta["heat_mask"])
    for model in selected.values():
        with np.load(path(model["grid"]), allow_pickle=False) as grid:
            for key in ["lat", "lon"]:
                np.testing.assert_array_equal(grid[key], meta[key])
            np.testing.assert_array_equal(grid["depth"][:14], meta["depths"])
            names = grid["names"].tolist()
            ids = np.array(
                [
                    [names.index(f"{v}_{i}") for v in ["thetao", "so", "uo", "vo"]]
                    for i in range(14)
                ]
            )
            np.testing.assert_array_equal(grid["mask"][ids], interior_mask)
            np.testing.assert_array_equal(
                grid["mask"][[names.index("thetao_0"), names.index("zos")]],
                surface_mask,
            )
    shutil.copytree(base, output)

    def write(name, values):
        filename = name + ".npy"
        p = output / filename
        if p.exists():
            raise ValueError("Array already exists: " + filename)
        np.save(p, values, allow_pickle=False)
        np.testing.assert_array_equal(
            np.load(p, mmap_mode="r", allow_pickle=False), values
        )
        meta["files"][filename] = dict(
            sha256=sha(p),
            shape=list(values.shape),
            dtype=str(values.dtype),
            bytes=p.stat().st_size,
        )
        return filename

    for name, model in selected.items():
        best = json.loads(path(model["best"]).read_text())
        manifest = json.loads(path(model["manifest"]).read_text())
        complete = json.loads(
            path(model["root"] + "/" + name + "/COMPLETE.json").read_text()
        )
        checksum = model["selected"]["checkpoint_sha256"]
        if (
            best["checkpoint_sha256"] != checksum
            or complete["inputs"]["checkpoint_sha256"] != checksum
        ):
            raise ValueError("Selected checkpoint differs")
        if sorted(model["origins"]) != sorted(meta["origins"]):
            raise ValueError("Origin cohort differs")
        meta["models"][name] = dict(
            label=name,
            short_label=name,
            report_url=EARLY_REPORT
            if "early" in name or name == "U-global"
            else PRESENTATION_REPORT,
            lineage=dict(
                checkpoint_policy="validation selected; constant learning rate",
                checkpoint_sha256=checksum,
                global_step=best["global_step"],
                task_counts=best["task_counts"],
                training_manifest_sha256=sha(path(model["manifest"])),
                training_producer=manifest["code_commit"],
            ),
        )
        for origin, remote in model["origins"].items():
            old = meta["records"][origin]
            with np.load(path(remote), allow_pickle=False) as data:
                if data["surface"].shape != (73, 2, len(meta["lat"]), len(meta["lon"])):
                    raise ValueError("Incomplete annual forecast")
                if data["months"].tolist() != old["heat_months"]:
                    raise ValueError("Monthly calendar differs")
                np.testing.assert_array_equal(
                    np.where(surface_mask, data["reference"][:, :2], np.nan),
                    np.load(base / old["surface_reference"]),
                )
                np.testing.assert_array_equal(
                    np.where(
                        heat_mask,
                        data["reference_ohc"].astype(np.float64) / 1e9,
                        np.nan,
                    ),
                    np.load(base / old["heat_reference"]),
                )
                arrays = {
                    "surface": np.where(surface_mask, data["surface"], np.nan),
                    "surface_persistence": np.where(
                        surface_mask, data["persistence_surface"], np.nan
                    ),
                    "interior": np.where(
                        interior_mask, data["initial"][-1][ids], np.nan
                    ),
                    "heat": np.where(
                        heat_mask,
                        data["predicted_ohc"].astype(np.float64) / 1e9,
                        np.nan,
                    ),
                    "heat_persistence": np.broadcast_to(
                        np.where(
                            heat_mask,
                            data["persistence_ohc"].astype(np.float64) / 1e9,
                            np.nan,
                        ),
                        (12, 2, len(meta["lat"]), len(meta["lon"])),
                    ),
                }
                for kind, values in arrays.items():
                    mask = (
                        interior_mask
                        if kind == "interior"
                        else heat_mask
                        if kind.startswith("heat")
                        else surface_mask
                    )
                    if not np.isfinite(values[..., mask]).all():
                        raise ValueError("Nonfinite model values on wet cells")
                record = {
                    kind: write(f"annual-{name}-{origin}-{kind}", value)
                    for kind, value in arrays.items()
                }
                record["report_heat"] = record["heat"]
                old["models"][name] = record
                meta["interior_examples"][origin]["models"][name] = record["interior"]
        print("Prepared " + name, flush=True)
    provenance = output / "annual-model-provenance"
    provenance.mkdir()
    shutil.copy2(source / "inventory.json", provenance / "inventory.json")
    for name, model in selected.items():
        for label in ["manifest", "best"]:
            shutil.copy2(path(model[label]), provenance / f"{name}-{label}.json")
        shutil.copy2(
            path(model["root"] + "/" + name + "/COMPLETE.json"),
            provenance / f"{name}-COMPLETE.json",
        )
    meta["annual_extension"] = dict(
        models=list(selected),
        inventory_sha256=sha(source / "inventory.json"),
        base_catalog_sha256=sha(base / "catalog.json"),
        script_sha256=sha(Path(__file__)),
        monthly_profiles_available=False,
    )
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, info in meta["files"].items():
        if sha(output / name) != info["sha256"]:
            raise ValueError("Prepared output checksum differs: " + name)
    print(
        f"Verified {len(selected)} new models; preserved {len(meta['models']) - len(selected)} existing models",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["base", "source", "output"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    extend(args.base, args.source, args.output)
