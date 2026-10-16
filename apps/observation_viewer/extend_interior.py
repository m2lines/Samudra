#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Add verified T/S/U/V initialization exports to an existing viewer bundle."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from export_interior import sha


def extend(base, export, output):
    meta = json.loads((base / "catalog.json").read_text())
    receipt = json.loads((export / "COMPLETE.json").read_text())
    if "interior_examples" in meta:
        raise ValueError("This bundle already has extended interior examples")
    if output.exists():
        raise ValueError("Choose a new output directory; preserve existing bundles")
    for name, info in receipt["files"].items():
        if Path(name).name != name or sha(export / name) != info["sha256"]:
            raise ValueError(f"Export checksum mismatch: {name}")
    assert receipt["lineage"] == meta["models"]["obs08000"]["lineage"]
    with np.load(export / "grid.npz", allow_pickle=False) as grid:
        for axis in ["lat", "lon", "depths"]:
            np.testing.assert_array_equal(grid[axis], meta[axis])
        mask = grid["mask"]
    assert mask.shape == (len(meta["depths"]), 4, len(meta["lat"]), len(meta["lon"]))
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask[:, :2], np.load(base / meta["interior_mask"]))
    shutil.copytree(base, output)

    def write(name, values):
        filename = name + ".npy"
        path = output / filename
        if path.exists():
            raise ValueError(f"Refusing to overwrite {filename}")
        np.save(path, values, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
        meta["files"][filename] = dict(
            shape=list(values.shape),
            dtype=str(values.dtype),
            sha256=sha(path),
            bytes=path.stat().st_size,
        )
        return filename

    def full_context(name):
        values = np.full(mask.shape, np.nan, dtype=np.float32)
        values[:, :2] = np.load(base / name, allow_pickle=False)
        return values

    def field(path, key):
        with np.load(path, allow_pickle=False) as archive:
            values = archive[key]
        assert values.shape == mask.shape and values.dtype == np.float32
        assert np.isfinite(values[mask]).all() and np.isnan(values[~mask]).all()
        return values

    meta["interior_mask"] = write("initialization-mask-tsuv", mask)
    climate = write(
        "initialization-climatology-tsuv", full_context(meta["climatology"])
    )
    meta["climatology"] = climate
    examples = {}
    for origin in meta["origins"]:
        case = receipt["cases"].get(origin)
        midpoint = case["state_midpoint"] if case else None
        references = {
            "observations": write(
                f"initialization-iap-tsuv-{origin}",
                full_context(meta["records"][origin]["interior_reference"]),
            ),
            "climatology": climate,
        }
        if case:
            assert midpoint is not None
            assert midpoint[:7] == meta["records"][origin]["context_month"]
            gold = write(
                f"initialization-om4-gold-{origin}",
                field(export / f"om4-{origin}.npz", "gold"),
            )
            prediction = write(
                f"initialization-om4-obs08000-{origin}",
                field(export / f"om4-{origin}.npz", "prediction"),
            )
            references["om4"] = gold
        models = {}
        for model, old in meta["records"][origin]["models"].items():
            assert (
                receipt["saved_observations"][model]["lineage"]
                == meta["models"][model]["lineage"]
            )
            values = field(export / f"{model}-obs-{origin}.npz", "prediction")
            # Preserve every previously displayed temperature/salinity value.
            np.testing.assert_array_equal(
                values[:, :2], np.load(base / old["interior"], allow_pickle=False)
            )
            models[model] = write(f"initialization-obs-{model}-{origin}", values)
        shared = (
            f"Last initialized five-day state, centered {midpoint}; 19 matching history intervals. "
            if case
            else "Last initialized five-day state before the forecast origin. "
        ) + "Surface temperature at 2.5 m is copied from the input. "
        om4_context = (
            "OM4 is contemporaneous simulation context, not observational truth. "
            if case
            else "No exactly matched OM4 history is available for this origin; velocity comparisons use another checkpoint. "
        )
        examples[origin] = dict(
            origin=origin,
            label=f"{origin} (obs)",
            source="obs",
            models=models,
            references=references,
            state_midpoint=midpoint,
            description=shared
            + "Observation inputs and observation task adapter. IAP preceding-December T/S is monthly context; it contains no U/V. "
            + om4_context
            + "Observation-task velocity slots are not directly supervised currents.",
        )
        if case:
            examples[origin + "-om4"] = dict(
                origin=origin,
                label=f"{origin} (om4)",
                source="om4",
                models={"obs08000": prediction},
                references={"om4": gold},
                state_midpoint=midpoint,
                description=shared
                + "OM4 surface history, native forcing and OM4 task adapter. Compare with the matching full OM4 state (gold) from that same five-day interval. Available checkpoint: Mixed · 8,000 observation updates.",
            )
    meta["interior_examples"] = examples
    receipt_name = "initialization-export-receipt.json"
    shutil.copy2(export / "COMPLETE.json", output / receipt_name)
    shutil.copy2(
        export / "contract.json", output / "initialization-export-contract.json"
    )
    shutil.copy2(
        export / "export_interior.py", output / "initialization-export-source.py"
    )
    provenance = output / "initialization-provenance"
    provenance.mkdir()
    for path in export.iterdir():
        if path.suffix != ".npz":
            shutil.copy2(path, provenance / path.name)
    meta["initialization_export"] = dict(
        receipt=receipt_name,
        receipt_sha256=sha(export / "COMPLETE.json"),
        base_catalog_sha256=sha(base / "catalog.json"),
        producer=receipt["producer"],
        provenance_directory=provenance.name,
        variables=["thetao", "so", "uo", "vo"],
        units=["°C", "psu", "m/s", "m/s"],
    )
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, info in meta["files"].items():
        if sha(output / name) != info["sha256"]:
            raise ValueError(f"Prepared array checksum mismatch: {name}")
        values = np.load(output / name, mmap_mode="r", allow_pickle=False)
        assert (
            list(values.shape) == info["shape"] and str(values.dtype) == info["dtype"]
        )
    print(
        f"Verified {len(meta['files'])} arrays and {len(examples)} initialization examples"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "export", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    extend(args.base, args.export, args.output)
