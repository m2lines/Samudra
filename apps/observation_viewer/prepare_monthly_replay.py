#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a complete six-model monthly-diffusion child catalog from audited inference."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from export_interior import sha
from extend_monthly_temperature import integrated_heat

REPORT = "https://github.com/m2lines/Samudra/blob/experiment/observation-stability-long-training/docs/experiments/observation-d/monthly-replay-2026-10-20.md"
NAMES = ("A", "E", "A-extend", "E-extend", "A-replay", "E-replay")


def prepare(export, output, context_catalog):
    context_meta = json.loads((context_catalog / "catalog.json").read_text())
    config = json.loads((export / "paths.json").read_text())
    audit = json.loads((export / "AUDITED.json").read_text())
    qualified = json.loads((export / "QUALIFIED.json").read_text())
    assert audit["config_sha256"] == sha(export / "paths.json")
    assert qualified["audit_sha256"] == sha(export / "AUDITED.json")
    assert qualified["first_month_readouts_exact"]
    assert qualified["producer"] == config["producer"]
    assert tuple(row["name"] for row in config["models"]) == NAMES
    assert config["origins"] == ["2015-01-01", "2018-01-01", "2021-01-01"]
    grid_path = next(p for p in audit["contracts"] if p.endswith("/grid.npz"))
    assert sha(export / "grid.npz") == audit["contracts"][grid_path]
    with np.load(export / "grid.npz") as saved:
        grid = dict(saved)
    for key in ("lat", "lon"):
        np.testing.assert_array_equal(grid[key], context_meta[key])
    np.testing.assert_array_equal(grid["depth"][:14], context_meta["depths"])
    mask = grid["mask"]
    h, w = len(grid["lat"]), len(grid["lon"])
    indices = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(14)])
    interior_mask = mask[indices]
    heat_mask = np.stack([interior_mask[:11, 0].all(0), interior_mask[10:14, 0].all(0)])
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def write(name, value):
        path = output / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
        files[path.name] = dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=sha(path),
            bytes=path.stat().st_size,
        )
        return path.name

    meta = dict(
        schema_version=1,
        report_url=REPORT,
        models={},
        origins=config["origins"],
        lat=grid["lat"].tolist(),
        lon=grid["lon"].tolist(),
        depths=grid["depth"][:14].tolist(),
        surface_mask=write("surface-mask", mask[[38, 76]]),
        interior_mask=write("interior-mask", interior_mask),
        heat_mask=write("heat-mask", heat_mask),
        heat_unit="GJ/m²",
        heat_input_unit="J/m²",
        heat_depth_bounds_m=[[0, 700], [700, 2000]],
        records={},
        interior_examples={},
        monthly_temperature=dict(
            units="°C",
            depth_count=14,
            source="Exact replay of fixed annual diffusion readouts; temperatures retained before heat integration",
        ),
    )
    for row in config["models"]:
        name = row["name"]
        model_id = "monthly-" + name
        directory = export / name
        receipt = json.loads((directory / "COMPLETE.json").read_text())
        assert receipt["producer"] == config["producer"]
        assert receipt["model"] == row and receipt["origins"] == config["origins"]
        assert receipt["exact_annual_overlap"]
        assert receipt["checkpoint_sha256"] == audit["contracts"][row["checkpoint"]]
        main = 4000 if "extend" in name else 2000
        mixed = name.startswith("A")
        replay = "replay" in name
        description = (
            f"**{name}** — monthly latent/diffusion pilot, **48,702,016 parameters (~48.7M)**; global 1° (180×360). "
            "One shared initializer, latent evolution processor and window-conditioned diffusion decoder. Two recurrent latent slots, 128 channels each, on a 45×90 grid; 19 five-day surface-history bins. "
            f"Seed 1729; {main:,} main optimizer updates ({'50% OM4 / 50% observations' if mixed else 'observations only'}) + 1,000 observation-only cooldown updates. "
            f"{'Half of observation examples use detached long-lead replay, including cooldown.' if replay else 'No observation replay.'} "
            f"{'Extended from the original raw 2k checkpoint before its first cooldown.' if 'extend' in name else 'Trained from random initialization.'} "
            f"Displayed weights are the fixed cooled EMA endpoint at total update {main + 1000:,}, not validation-selected. No latent magnitude penalty or bound.\n\n"
            "Annual rollouts initialize once, then evolve 73 five-day steps under prescribed ERA5. Displayed fields average eight diffusion draws with 32 sampler steps. Surface forecasts and monthly heat content exactly reproduce the published annual outputs. Monthly profiles are the actual decoder temperature fields used in those OHC integrals. Initial interiors use an independent eight-draw reconstruction of the initial latent state; they are not fed back into the rollout.\n\n"
            "IAP T/S is preceding-December monthly context, not instantaneous truth. Initialized U/V are decoder outputs without observed-current reconstruction supervision; observation-only runs never receive full-field model supervision. Own persistence holds the decoded initial physical state fixed. Three annual starts and one seed do not establish reproducibility."
        )
        meta["models"][model_id] = dict(
            label=f"Monthly diffusion · {name}",
            short_label=name,
            report_url=REPORT,
            description=description,
            lineage=dict(
                checkpoint=row["checkpoint"],
                checkpoint_sha256=receipt["checkpoint_sha256"],
                checkpoint_policy="fixed cooled EMA endpoint",
                global_step=main + 1000,
                task_counts=dict(
                    om4=main // 2 if mixed else 0,
                    observation=main // 2 + 1000 if mixed else main + 1000,
                ),
                export_producer=config["producer"],
            ),
        )
        assert len(receipt["files"]) == 3
        for origin in config["origins"]:
            filename = origin + ".npz"
            assert sha(directory / filename) == receipt["files"][filename]
            with np.load(directory / filename) as saved:
                c = dict(saved)
            for key, shape in dict(
                surface=(73, 2, h, w),
                ohc=(12, 2, h, w),
                temperature=(12, 14, h, w),
                initial_interior=(14, 4, h, w),
                reference_temperature=(12, 14, h, w),
            ).items():
                assert c[key].shape == shape, (name, key)
            assert c["months"].tolist() == [
                f"{origin[:4]}-{i:02d}" for i in range(1, 13)
            ]
            temperature = np.where(interior_mask[:, 0], c["temperature"], np.nan)
            assert np.isfinite(temperature[:, interior_mask[:, 0]]).all()
            heat = np.where(heat_mask, c["ohc"] / 1e9, np.nan)
            # Independent depth-overlap integration protects against mismatched
            # decoder windows, temperature units, axes and incomplete columns.
            np.testing.assert_allclose(
                integrated_heat(temperature, heat_mask),
                heat,
                rtol=2e-7,
                atol=1e-6,
                equal_nan=True,
            )
            initial = np.where(interior_mask, c["initial_interior"], np.nan)
            held_temperature = np.broadcast_to(initial[:, 0], temperature.shape)
            held_heat = np.where(heat_mask, c["initial_ohc"] / 1e9, np.nan)
            np.testing.assert_allclose(
                integrated_heat(held_temperature, heat_mask),
                np.broadcast_to(held_heat, heat.shape),
                rtol=2e-7,
                atol=1e-6,
                equal_nan=True,
            )
            initial_name = write(f"{name}-{origin}-interior", initial)
            record = meta["records"].setdefault(
                origin,
                dict(
                    context_month=f"{int(origin[:4]) - 1}-12",
                    heat_months=c["months"].tolist(),
                    models={},
                ),
            )
            context_record = context_meta["interior_examples"][origin]
            assert context_record["source"] == "obs"
            context_name = context_record["references"]["observations"]
            assert (
                sha(context_catalog / context_name)
                == context_meta["files"][context_name]["sha256"]
            )
            initial_reference = np.load(context_catalog / context_name)
            assert initial_reference.shape == (14, 4, h, w)
            assert np.isnan(initial_reference[:, 2:]).all()
            references = dict(
                surface_reference=np.where(
                    mask[[38, 76]], c["reference_surface"][:, :2], np.nan
                ),
                heat_reference=np.where(heat_mask, c["reference_ohc"] / 1e9, np.nan),
                temperature_reference=np.where(
                    interior_mask[:, 0], c["reference_temperature"], np.nan
                ),
                interior_reference=np.where(interior_mask, initial_reference, np.nan),
            )
            for key, values in references.items():
                if key in record:
                    np.testing.assert_array_equal(np.load(output / record[key]), values)
                else:
                    record[key] = write(f"{origin}-{key}", values)
            record["models"][model_id] = dict(
                surface=write(
                    f"{name}-{origin}-surface",
                    np.where(mask[[38, 76]], c["surface"], np.nan),
                ),
                surface_persistence=write(
                    f"{name}-{origin}-surface-persistence",
                    np.broadcast_to(
                        np.where(mask[[38, 76]], c["initial_surface"], np.nan),
                        c["surface"].shape,
                    ),
                ),
                interior=initial_name,
                heat=write(f"{name}-{origin}-heat", heat),
                heat_persistence=write(
                    f"{name}-{origin}-held-heat", np.broadcast_to(held_heat, heat.shape)
                ),
                temperature=write(f"{name}-{origin}-temperature", temperature),
                temperature_persistence=write(
                    f"{name}-{origin}-held-temperature", held_temperature
                ),
            )
            example = meta["interior_examples"].setdefault(
                origin,
                dict(
                    origin=origin,
                    label=f"{origin} (obs)",
                    source="obs",
                    models={},
                    references=dict(observations=record["interior_reference"]),
                    description="Eight-draw physical reconstruction of the initial latent state from 19 five-day surface-history bins; not copied physical input fields. Preceding-December IAP T/S is monthly context, not instantaneous truth, and contains no observed U/V. Initial U/V outputs have no observation reconstruction supervision. This reconstruction is not used to evolve the forecast.",
                ),
            )
            example["models"][model_id] = initial_name
            print(
                f"Verified {name} / {origin}: exact annual replay and independent OHC integration",
                flush=True,
            )
    meta["files"] = files
    provenance = output / "provenance"
    provenance.mkdir()
    for name in ["paths.json", "AUDITED.json", "QUALIFIED.json"]:
        shutil.copyfile(export / name, provenance / name)
    for row in config["models"]:
        shutil.copyfile(
            export / row["name"] / "COMPLETE.json",
            provenance / (row["name"] + "-COMPLETE.json"),
        )
    meta["initial_context_provenance"] = dict(
        catalog_sha256=sha(context_catalog / "catalog.json"),
        arrays={
            o: context_meta["files"][
                context_meta["interior_examples"][o]["references"]["observations"]
            ]
            for o in config["origins"]
        },
    )
    meta["export_provenance"] = dict(
        producer=config["producer"],
        exact_published_annual_overlap=True,
        independently_integrated_ohc=True,
        model_count=6,
        origin_count=3,
    )
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    for filename, info in files.items():
        assert sha(output / filename) == info["sha256"]
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-catalog", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.export, args.output, args.context_catalog)
