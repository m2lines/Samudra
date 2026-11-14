#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Extend a report bundle with verified monthly T and consistent replay OHC."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from export_interior import sha


def integrated_heat(temperature, mask):
    """Report convention: native layer overlaps, rho=1035, cp=3850, GJ/m²."""
    interfaces = np.array(
        [0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650, 900, 1200, 1600, 2100]
    )
    result = []
    for i, (lower, upper) in enumerate([(0, 700), (700, 2000)]):
        thickness = np.maximum(
            0, np.minimum(interfaces[1:], upper) - np.maximum(interfaces[:-1], lower)
        )
        heat = (
            np.nansum(
                temperature * thickness[None, :, None, None] * 1035 * 3850, axis=1
            )
            / 1e9
        )
        result.append(np.where(mask[i], heat, np.nan))
    return np.stack(result, axis=1)


def extend(base, export, output):
    meta = json.loads((base / "catalog.json").read_text())
    receipt = json.loads((export / "COMPLETE.json").read_text())
    if "monthly_temperature" in meta:
        raise ValueError("Monthly temperature is already present")
    if sha(base / "catalog.json") != receipt["base_catalog_sha256"]:
        raise ValueError("Base catalog differs from replay inputs")
    for name, info in receipt["files"].items():
        if Path(name).name != name or sha(export / name) != info["sha256"]:
            raise ValueError(f"Export checksum mismatch: {name}")
    for name, key in [
        ("contract.json", "contract_sha256"),
        ("export_monthly_temperature.py", "script_sha256"),
    ]:
        if sha(export / name) != receipt[key]:
            raise ValueError(f"Provenance checksum mismatch: {name}")
    mask = np.load(base / meta["interior_mask"])[:, 0]
    heat_mask = np.load(base / meta["heat_mask"])
    shape = (12, len(meta["depths"]), len(meta["lat"]), len(meta["lon"]))
    shutil.copytree(base, output)  # Refuse to overwrite an existing bundle.

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

    for origin, record in meta["records"].items():
        reference = None
        for model, old in record["models"].items():
            filename = f"{model}-{origin}.npz"
            if (
                receipt["cases"][filename]["lineage"]
                != meta["models"][model]["lineage"]
            ):
                raise ValueError("Replay lineage differs from the report")
            with np.load(export / filename, allow_pickle=False) as saved:
                temperature, current, heat = (
                    saved["temperature"],
                    saved["reference"],
                    saved["heat"],
                )
                assert saved["months"].tolist() == record["heat_months"]
            for values in [temperature, current]:
                assert values.shape == shape and values.dtype == np.float32
                assert np.isnan(values[:, ~mask]).all()
            assert np.isfinite(temperature[:, mask]).all()
            # Independently integrate the exported depth profiles to catch axis,
            # units, calendar or masking mistakes before publishing the bundle.
            np.testing.assert_allclose(
                integrated_heat(temperature, heat_mask),
                heat,
                rtol=1e-12,
                atol=1e-10,
                equal_nan=True,
            )
            old["temperature"] = write(
                f"monthly-temperature-{model}-{origin}", temperature
            )
            old["report_heat"] = old["heat"]  # Retain original report arrays.
            old["heat"] = write(f"monthly-replay-heat-{model}-{origin}", heat)
            if reference is None:
                reference = current
                record["temperature_reference"] = write(
                    f"monthly-temperature-iap-{origin}", reference
                )
            else:
                np.testing.assert_array_equal(current, reference)
    provenance = output / "monthly-temperature-provenance"
    provenance.mkdir()
    for name in ["COMPLETE.json", "contract.json", "export_monthly_temperature.py"]:
        shutil.copy2(export / name, provenance / name)
    meta["monthly_temperature"] = dict(
        units="°C",
        source="regenerated frozen annual forecasts",
        depth_count=14,
        description="Heat maps and temperature profiles use the same regenerated annual forecasts from the report's frozen checkpoints and inputs. Runtime differences change some values; original report arrays are retained. Profiles compare with IAP monthly temperatures on the model depth grid (2.5–1,850 m).",
        base_catalog_sha256=receipt["base_catalog_sha256"],
        receipt=str(Path(provenance.name) / "COMPLETE.json"),
        receipt_sha256=sha(export / "COMPLETE.json"),
        report_comparison={
            name.removesuffix(".npz"): case["heat_difference_from_report_gj_m2"]
            for name, case in receipt["cases"].items()
        },
    )
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, info in meta["files"].items():
        if sha(output / name) != info["sha256"]:
            raise ValueError(f"Prepared checksum mismatch: {name}")
    print(
        f"Verified {len(meta['files'])} arrays including monthly temperature and replay OHC"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "export", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    extend(args.base, args.export, args.output)
