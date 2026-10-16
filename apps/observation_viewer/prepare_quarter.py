#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Extract a separate native-quarter viewer catalog from frozen annual exports.

No model imports or inference. Read one saved array at a time; write/read back
one leading-axis slice at a time. Completion is published only after hashes,
full-cohort identity, geometry, references and output arrays have passed.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]
LEADS = [5, 15, 30, 90, 180, 365]
REPORT = "https://github.com/m2lines/Samudra/blob/experiment/patch-global-om4/docs/experiments/observation-d/quarter-baseline-results-2026-10-15.md"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    require(sha(path) == expected, "Checksum differs: " + str(path))


def interior_indices(names):
    return np.array(
        [
            [names.index(f"{v}_{i}") for v in ("thetao", "so", "uo", "vo")]
            for i in range(19)
        ]
    )


class Writer:
    def __init__(self, output):
        self.output = Path(output)
        self.files = {}

    def write(self, name, values, *, mask=None, indices=None, scale=1):
        """Transform and verify individual leading slices, keeping peak RAM bounded."""
        path = self.output / (name + ".npy")
        require(not path.exists(), "Refusing to overwrite " + str(path))
        shape = (
            values.shape
            if indices is None
            else (len(values), *indices.shape, *values.shape[-2:])
        )
        dtype = values.dtype if scale == 1 else np.dtype("float64")
        target = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        for i in range(len(values)):
            row = values[i] if indices is None else values[i][indices]
            if scale != 1:
                row = row.astype(np.float64) / scale
            if mask is not None:
                row = np.where(mask, row, np.nan)
            require(not np.isinf(row).any(), "Infinite saved values: " + name)
            target[i] = row
        target.flush()
        del target
        stored = np.load(path, mmap_mode="r", allow_pickle=False)
        for i in range(len(values)):
            row = values[i] if indices is None else values[i][indices]
            if scale != 1:
                row = row.astype(np.float64) / scale
            if mask is not None:
                row = np.where(mask, row, np.nan)
            np.testing.assert_array_equal(stored[i], row)
        self.files[path.name] = dict(
            shape=list(stored.shape),
            dtype=str(stored.dtype),
            sha256=sha(path),
            bytes=path.stat().st_size,
        )
        return path.name


def verify_evaluation(model, config):
    directory = Path(model["evaluation"])
    receipt = Path(str(directory) + "-RESULT.json")
    verify(receipt, model["result_sha256"])
    result = json.loads(receipt.read_text())
    require(result["passed"] is True, "Annual evaluation did not pass")
    require(
        result["evaluation_producer"] == config["producer"],
        "Evaluation producer differs",
    )
    verify(directory / "COMPLETE.json", result["complete_sha256"])
    complete = json.loads((directory / "COMPLETE.json").read_text())
    inp = complete["inputs"]
    require(sorted(complete["origins"]) == ORIGINS, "Incomplete annual cohort")
    require(
        "diagnostic_origin_subset" not in inp,
        "Diagnostic subset cannot populate viewer",
    )
    require(
        inp["checkpoint_sha256"] == model["checkpoint_sha256"],
        "Selected checkpoint differs",
    )
    require(
        inp["include_persistence"] and inp["global_observations"],
        "Missing native controls",
    )
    require(
        inp["sample_manifest_sha256"] == config["sample_manifest_sha256"],
        "Observation manifest differs",
    )
    verify(Path(model["training_manifest"]), inp["training_manifest_sha256"])
    for name, expected in inp["data_manifests"].items():
        verify(Path(config["annual_data"]) / name, expected)
    for origin in ORIGINS:
        name = origin + ".npz"
        info = result["outputs"][name]
        require(
            (directory / name).stat().st_size == info["bytes"],
            "Annual archive byte count differs",
        )
        verify(directory / name, info["sha256"])
    return result, complete


def prepare(config_path, output):
    config_path, output = Path(config_path), Path(output)
    require(not output.exists(), "Choose a new immutable output bundle")
    config = json.loads(config_path.read_text())
    require(config["origins"] == ORIGINS, "Unexpected origin cohort")
    require(
        [m["key"] for m in config["models"]]
        == ["quarter-uglobal-selected", "quarter-obs-only-selected"],
        "Unexpected selected models",
    )
    grid_path = Path(config["observations"]) / "grid.npz"
    iap_path = Path(config["observations"]) / "iap-grid.npz"
    verify(grid_path, config["grid_sha256"])
    verify(iap_path, config["iap_grid_sha256"])
    with np.load(grid_path, allow_pickle=False) as grid:
        names = grid["names"].tolist()
        ids = interior_indices(names)
        lat, lon = grid["lat"], grid["lon"]
        depths, thickness = grid["depth"], grid["depth_thickness"]
        wet = grid["mask"]
    with np.load(iap_path, allow_pickle=False) as iap:
        heat_lat, heat_lon = iap["lat"], iap["lon"]
    require(
        (len(lat), len(lon)) == tuple(config["native_shape"]), "Native grid differs"
    )
    require(
        (len(heat_lat), len(heat_lon)) == tuple(config["heat_shape"]),
        "Native IAP grid differs",
    )
    require(len(depths) == 19 and len(names) == 77, "Unexpected physical state layout")
    interfaces = np.r_[0, np.cumsum(thickness)]
    np.testing.assert_array_equal((interfaces[:-1] + interfaces[1:]) / 2, depths)
    surface_mask, interior_mask = (
        wet[[names.index("thetao_0"), names.index("zos")]],
        wet[ids],
    )
    verified = [verify_evaluation(m, config) for m in config["models"]]
    require(
        verified[0][1]["inputs"]["data_manifests"]
        == verified[1][1]["inputs"]["data_manifests"],
        "Annual source cohorts differ",
    )
    output.mkdir(parents=True)
    writer = Writer(output)
    meta = dict(
        schema_version=1,
        resolution="native-quarter",
        report_url=REPORT,
        lat=lat.tolist(),
        lon=lon.tolist(),
        depths=depths.tolist(),
        depth_interfaces=interfaces.tolist(),
        heat_lat=heat_lat.tolist(),
        heat_lon=heat_lon.tolist(),
        heat_unit="GJ/m²",
        heat_input_unit="J/m²",
        heat_depth_bounds_m=[[0, 700], [700, 2000]],
        origins=ORIGINS,
        surface_leads_days=list(range(5, 366, 5)),
        forecast_interior_leads_days=LEADS,
        surface_mask=writer.write("surface-mask", surface_mask),
        interior_mask=writer.write("interior-mask", interior_mask),
        models={},
        records={},
        interior_examples={},
        unavailable=dict(
            monthly_temperature="Not saved by these annual exports; no reconstruction or new forecast performed",
            interior_observations="No instantaneous observed full-state T/S/U/V on the model grid",
        ),
    )
    heat_mask = None
    for model, (result, complete) in zip(config["models"], verified, strict=True):
        key, directory = model["key"], Path(model["evaluation"])
        manifest = json.loads(Path(model["training_manifest"]).read_text())
        meta["models"][key] = dict(
            label=model["label"],
            short_label=model["label"],
            report_url=REPORT,
            description=model["description"],
            lineage=dict(
                checkpoint_sha256=model["checkpoint_sha256"],
                training_manifest_sha256=complete["inputs"]["training_manifest_sha256"],
                training_producer=manifest.get("producer", manifest.get("code_commit")),
                global_step=model.get("selected_step"),
                task_counts=model.get("task_counts"),
                parameters=62_967_680,
                evaluation_producer=config["producer"],
                evaluation_inputs=complete["inputs"],
                result_sha256=model["result_sha256"],
                source_evaluation=str(directory),
            ),
        )
        for origin in ORIGINS:
            record = meta["records"].setdefault(
                origin,
                dict(
                    models={},
                    context_month=f"{int(origin[:4]) - 1}-12",
                    heat_months=[f"{origin[:4]}-{month:02d}" for month in range(1, 13)],
                ),
            )
            row = {}
            with np.load(directory / (origin + ".npz"), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["leads_days"], LEADS)
                np.testing.assert_array_equal(saved["months"], record["heat_months"])
                np.testing.assert_array_equal(saved["iap_lat"], heat_lat)
                np.testing.assert_array_equal(saved["iap_lon"], heat_lon)
                for field in ("surface", "persistence_surface"):
                    values = saved[field]
                    require(
                        values.shape == (73, 2, len(lat), len(lon)),
                        "Unexpected annual surface shape",
                    )
                    require(
                        np.isfinite(values[:, surface_mask]).all(),
                        "Nonfinite wet model surface",
                    )
                    out_key = "surface" if field == "surface" else "surface_persistence"
                    row[out_key] = writer.write(
                        f"{key}-{origin}-{out_key}", values, mask=surface_mask
                    )
                    del values
                reference = saved["reference"][:, :2]
                if "surface_reference" in record:
                    prior = np.load(output / record["surface_reference"], mmap_mode="r")
                    for i in range(73):
                        np.testing.assert_array_equal(
                            prior[i], np.where(surface_mask, reference[i], np.nan)
                        )
                else:
                    record["surface_reference"] = writer.write(
                        f"{origin}-surface-reference", reference, mask=surface_mask
                    )
                del reference
                initial = saved["initial"]
                require(
                    initial.shape == (2, 77, len(lat), len(lon)),
                    "Unexpected initializer shape",
                )
                interior = initial[-1][ids]
                require(
                    np.isfinite(interior[interior_mask]).all(),
                    "Nonfinite wet initializer",
                )
                row["interior"] = writer.write(
                    f"{key}-{origin}-interior",
                    np.where(interior_mask, interior, np.nan),
                )
                np.testing.assert_array_equal(
                    saved["persistence_surface"][0], initial[-1, [38, 76]]
                )
                del initial, interior
                state = saved["state_at_leads"]
                require(
                    state.shape == (6, 77, len(lat), len(lon)),
                    "Unexpected saved interior leads",
                )
                require(
                    all(np.isfinite(s[wet]).all() for s in state),
                    "Nonfinite wet saved state",
                )
                surface_saved = np.load(output / row["surface"], mmap_mode="r")
                for si, day in enumerate(LEADS):
                    np.testing.assert_array_equal(
                        np.where(surface_mask, state[si, [38, 76]], np.nan),
                        surface_saved[day // 5 - 1],
                    )
                row["forecast_interior"] = writer.write(
                    f"{key}-{origin}-forecast-interior",
                    state,
                    indices=ids,
                    mask=interior_mask,
                )
                del state
                for field, out_key in (
                    ("predicted_ohc", "heat"),
                    ("reference_ohc", "heat_reference"),
                    ("persistence_ohc", "heat_persistence"),
                ):
                    values = saved[field]
                    if field == "persistence_ohc":
                        values = np.broadcast_to(values, (12, *values.shape))
                    require(
                        values.shape == (12, 2, len(heat_lat), len(heat_lon)),
                        "Unexpected IAP OHC shape",
                    )
                    if field == "predicted_ohc":
                        current = np.isfinite(values).all(0)
                        require(
                            np.array_equal(
                                np.isfinite(values),
                                np.broadcast_to(current, values.shape),
                            ),
                            "OHC prediction support changes with time",
                        )
                        if heat_mask is None:
                            heat_mask = current
                            meta["heat_mask"] = writer.write("heat-mask", heat_mask)
                        else:
                            np.testing.assert_array_equal(current, heat_mask)
                    if out_key == "heat_reference" and out_key in record:
                        prior = np.load(output / record[out_key], mmap_mode="r")
                        for i in range(12):
                            np.testing.assert_array_equal(
                                prior[i], values[i].astype(np.float64) / 1e9
                            )
                    else:
                        name = (
                            f"{origin}-heat-reference"
                            if out_key == "heat_reference"
                            else f"{key}-{origin}-{out_key}"
                        )
                        filename = writer.write(name, values, scale=1e9)
                        (record if out_key == "heat_reference" else row)[out_key] = (
                            filename
                        )
                    del values
            record["models"][key] = row
            example = meta["interior_examples"].setdefault(
                origin,
                dict(
                    origin=origin,
                    label=origin + " (obs)",
                    source="obs",
                    description="Last saved inferred five-day history state; no instantaneous observed full-state reference. T/S/U/V are physical model fields.",
                    models={},
                    references={},
                ),
            )
            example["models"][key] = row["interior"]
            print(f"Verified {key}/{origin}", flush=True)
        provenance = output / ("provenance-" + key)
        provenance.mkdir()
        for source, name in (
            (directory / "COMPLETE.json", "annual-COMPLETE.json"),
            (Path(str(directory) + "-RESULT.json"), "annual-RESULT.json"),
            (Path(model["training_manifest"]), "training-manifest.json"),
        ):
            shutil.copyfile(source, provenance / name)
    shutil.copyfile(config_path, output / "extraction-config.json")
    shutil.copyfile(__file__, output / "prepare_quarter.py")
    meta["files"] = writer.files
    meta["source_grid_sha256"] = config["grid_sha256"]
    meta["source_iap_grid_sha256"] = config["iap_grid_sha256"]
    meta["extraction_config_sha256"] = sha(config_path)
    meta["extraction_script_sha256"] = sha(__file__)
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                passed=True,
                catalog_sha256=sha(output / "catalog.json"),
                files=writer.files,
                provenance_files={
                    str(p.relative_to(output)): dict(
                        sha256=sha(p), bytes=p.stat().st_size
                    )
                    for p in output.rglob("*")
                    if p.is_file() and p.suffix != ".npy"
                },
                scope="Saved selected annual arrays only; no inference or new observations",
            ),
            indent=2,
        )
        + "\n"
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)
