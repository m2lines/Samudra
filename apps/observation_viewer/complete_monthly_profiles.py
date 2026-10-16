#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Complete all models' monthly profiles without replacing historical arrays."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from export_interior import sha
from extend_monthly_temperature import integrated_heat


def complete(base, export, output):
    meta = json.loads((base / "catalog.json").read_text())
    config = json.loads((export / "paths.json").read_text())
    if config["base_catalog_sha256"] != sha(base / "catalog.json"):
        raise ValueError("Base catalog differs from rollout inputs")
    signature = dict(
        config_sha256=sha(export / "paths.json"),
        script_sha256=sha(export / "export_annual_profiles.py"),
    )
    if json.loads((export / "audit-output/AUDITED.json").read_text()) != signature:
        raise ValueError("Rollout audit contract differs")
    missing = {
        model
        for origin in meta["origins"]
        for model, row in meta["records"][origin]["models"].items()
        if "temperature" not in row
    }
    if {r["name"] for r in config["models"]} != missing or not missing:
        raise ValueError("Export does not match missing model profiles")
    receipts = {}
    for model in sorted(missing):
        folder = export / "exports" / model
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        if (
            any(receipt[k] != v for k, v in signature.items())
            or receipt["model"] != model
        ):
            raise ValueError("Export signature differs")
        if receipt["lineage"] != meta["models"][model]["lineage"]:
            raise ValueError("Checkpoint lineage differs")
        if sorted(receipt["cases"]) != sorted(meta["origins"]) or sorted(
            receipt["files"]
        ) != sorted(o + ".npz" for o in meta["origins"]):
            raise ValueError("Incomplete export cohort")
        for name, info in receipt["files"].items():
            if (
                Path(name).name != name
                or sha(folder / name) != info["sha256"]
                or (folder / name).stat().st_size != info["bytes"]
            ):
                raise ValueError("Export checksum differs")
        receipts[model] = receipt
    for name, info in meta["files"].items():
        if Path(name).name != name or sha(base / name) != info["sha256"]:
            raise ValueError("Base array changed")
    mask = np.load(base / meta["interior_mask"])[:, 0]
    heat_mask = np.load(base / meta["heat_mask"])
    shape = (12, len(meta["depths"]), len(meta["lat"]), len(meta["lon"]))
    shutil.copytree(base, output)

    def write(name, values):
        filename = name + ".npy"
        path = output / filename
        if path.exists():
            raise ValueError("Output array already exists")
        np.save(path, values, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
        meta["files"][filename] = dict(
            shape=list(values.shape),
            dtype=str(values.dtype),
            sha256=sha(path),
            bytes=path.stat().st_size,
        )
        return filename

    for origin, record in meta["records"].items():
        reference = np.load(base / record["temperature_reference"])
        for model in sorted(missing):
            row = record["models"][model]
            with np.load(
                export / "exports" / model / (origin + ".npz"), allow_pickle=False
            ) as saved:
                if saved["months"].tolist() != record["heat_months"]:
                    raise ValueError("Monthly calendar differs")
                temperature = saved["temperature"]
                if (
                    temperature.shape != shape
                    or temperature.dtype != np.float32
                    or not np.isnan(temperature[:, ~mask]).all()
                    or not np.isfinite(temperature[:, mask]).all()
                ):
                    raise ValueError(
                        "Temperature shape, units/mask or finite support differs"
                    )
                np.testing.assert_array_equal(saved["reference"], reference)
                np.testing.assert_allclose(
                    integrated_heat(temperature, heat_mask),
                    saved["heat"],
                    rtol=1e-12,
                    atol=1e-10,
                    equal_nan=True,
                )
                np.testing.assert_allclose(
                    integrated_heat(saved["temperature_persistence"][None], heat_mask)[
                        0
                    ],
                    saved["heat_persistence"],
                    rtol=1e-12,
                    atol=1e-10,
                    equal_nan=True,
                )
                row["temperature"] = write(
                    f"profiles-{model}-{origin}-temperature", temperature
                )
                row["report_heat"] = row.get("report_heat", row["heat"])
                row["heat"] = write(f"profiles-{model}-{origin}-heat", saved["heat"])
                row["temperature_persistence"] = write(
                    f"profiles-{model}-{origin}-held-temperature",
                    np.broadcast_to(saved["temperature_persistence"], shape),
                )
                row["heat_persistence"] = write(
                    f"profiles-{model}-{origin}-held-heat",
                    np.broadcast_to(saved["heat_persistence"], (12, 2, *shape[-2:])),
                )
            meta["monthly_temperature"]["report_comparison"][model + "-" + origin] = (
                receipts[model]["cases"][origin]["heat_difference_from_report_gj_m2"]
            )
        # Existing models also get a monthly persistence comparator from their
        # own saved initialized profile, with OHC integrated from that profile.
        for model, row in record["models"].items():
            if "temperature_persistence" not in row:
                initial = np.load(base / row["interior"])[:, 0]
                held = np.broadcast_to(initial, shape)
                row["temperature_persistence"] = write(
                    f"profiles-{model}-{origin}-held-temperature", held
                )
                row["heat_persistence"] = write(
                    f"profiles-{model}-{origin}-held-heat",
                    integrated_heat(held, heat_mask),
                )
    provenance = output / "complete-profile-provenance"
    provenance.mkdir()
    for name in ["paths.json", "export_annual_profiles.py", "runtime-contract.json"]:
        shutil.copy2(export / name, provenance / name)
    shutil.copy2(export / "audit-output/AUDITED.json", provenance / "AUDITED.json")
    for model in receipts:
        shutil.copy2(
            export / "exports" / model / "COMPLETE.json",
            provenance / (model + "-COMPLETE.json"),
        )
    meta["monthly_temperature"].update(
        complete_models=list(meta["models"]),
        description="Heat maps and temperature profiles use the same calendar-month forecast means. Profiles compare with IAP monthly temperatures at model depths (2.5–1,850 m).",
        completion_receipts=provenance.name,
    )
    for name, info in meta["files"].items():
        if sha(output / name) != info["sha256"]:
            raise ValueError("Prepared output checksum differs")
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Verified monthly profiles for all {len(meta['models'])} models", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "export", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    complete(args.base, args.export, args.output)
