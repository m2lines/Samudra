# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert verified deterministic endpoint exports to a standalone viewer catalog."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    meta: dict[str, Any] = dict(
        schema_version=1,
        report_url="https://github.com/m2lines/Samudra/pull/958",
        models={},
        origins=["2015-01-01", "2018-01-01", "2021-01-01"],
        records={},
        interior_examples={},
        files={},
        heat_unit="GJ/m²",
        heat_input_unit="J/m²",
        heat_depth_bounds_m=[[0, 700], [700, 2000]],
    )

    def write(name, values):
        name += ".npy"
        p = output / name
        assert not p.exists()
        np.save(p, values, allow_pickle=False)
        np.testing.assert_array_equal(np.load(p, mmap_mode="r"), values)
        meta["files"][name] = dict(
            sha256=sha(p),
            shape=list(values.shape),
            dtype=str(values.dtype),
            bytes=p.stat().st_size,
        )
        return name

    for arm in ["C", "D"]:
        name = f"deterministic-{arm}-061126"
        folder = source / name
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        assert (
            receipt["updates"] == 5000
            and receipt["weights"] == "EMA"
            and receipt["arm"] == arm
        )
        (output / f"{name}-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        recipe = (
            "5,000 observation updates"
            if arm == "C"
            else "2,000 OM4 + 3,000 observation updates"
        )
        meta["models"][name] = dict(
            label=name,
            short_label=name,
            report_url=meta["report_url"],
            description=f"**{name}.** Deterministic full Samudra evolution U-Net and large Samudra initializer; direct eight-channel ERA5 conditioning; source-separated replay. EMA at the exact 5,000-update endpoint: 4,000 main updates plus 1,000 observation-only cooldown updates ({recipe}), effective batch 8. Observation-history initialization; future ERA5 is prescribed, not issuance-time forcing. Annual surface fields and saved interior leads are five-day means; monthly T/S and heat content use exact calendar-overlap aggregation. Initialization IAP is preceding-month context, not instantaneous truth. Velocity and deep fields lack observational supervision. No spatial smoothing.",
            lineage=dict(
                checkpoint_sha256=receipt["checkpoint_sha256"],
                global_step=5000,
                weights="EMA",
                checkpoint_policy="fixed 5k cooldown endpoint, not best-validation selection",
                exposure=receipt["exposure"],
                training_manifest=receipt["training_manifest"],
                exporter_sha256=receipt["exporter_sha256"],
            ),
        )
        for origin in meta["origins"]:
            info = receipt["cases"][origin]
            p = folder / info["file"]
            assert sha(p) == info["sha256"]
            with np.load(p) as z:
                saved = dict(z)
            mask = saved["mask"]
            sm = mask[[38, 76]]
            ids = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(19)])
            im = mask[ids]
            tsmask = im[:, :2]
            edges = saved["depth_interfaces"]
            hm = np.stack(
                [
                    mask[38:57][(edges[:-1] < hi) & (edges[1:] > lo)].all(0)
                    for lo, hi in [(0, 700), (700, 2000)]
                ]
            )
            if "lat" not in meta:
                meta.update(
                    lat=saved["lat"].tolist(),
                    lon=saved["lon"].tolist(),
                    depths=saved["depth"].tolist(),
                    depth_interfaces=edges.tolist(),
                    surface_mask=write("surface-mask", sm),
                    interior_mask=write("interior-mask", im),
                    heat_mask=write("heat-mask", hm),
                )
            else:
                for key in ["lat", "lon"]:
                    np.testing.assert_array_equal(saved[key], meta[key])
                np.testing.assert_array_equal(
                    im, np.load(output / meta["interior_mask"])
                )
            if arm == "C":
                ref = np.full((12, 19, 2, *sm.shape[-2:]), np.nan, np.float32)
                ref[:, :14] = saved["reference_monthly_ts"].transpose(0, 2, 1, 3, 4)
                ref = np.where(tsmask, ref, np.nan)
                record = dict(
                    origin=origin,
                    source="obs",
                    label=origin + " (obs; EMA 5000)",
                    models={},
                    forecast_interior_leads_days=saved["interior_leads"].tolist(),
                    starts=saved["starts"].tolist(),
                    ends=saved["ends"].tolist(),
                    complete=True,
                    completed_bins=73,
                    first_nonfinite=None,
                    heat_months=saved["months"].tolist(),
                    surface_reference=write(
                        origin + "-surface-reference",
                        np.where(sm, saved["reference_surface"], np.nan),
                    ),
                    monthly_interior_reference=write(
                        origin + "-monthly-reference", ref
                    ),
                    temperature_reference=write(
                        origin + "-temperature-reference", ref[:, :, 0]
                    ),
                    heat_reference=write(
                        origin + "-heat-reference",
                        np.where(hm, saved["reference_monthly_ohc"] / 1e9, np.nan),
                    ),
                )
                meta["records"][origin] = record
                e = dict(
                    origin=origin,
                    label=record["label"],
                    source="obs",
                    models={},
                    references={},
                    description=f"Observation-history initialization; five-day state from {saved['initial_start']} to {saved['initial_end']} (exclusive end).",
                )
                if "initial_iap_ts" in saved:
                    context = np.full(im.shape, np.nan, np.float32)
                    assert saved["initial_iap_ts"].shape == (2, 14, *sm.shape[-2:])
                    context[:14, :2] = saved["initial_iap_ts"].transpose(1, 0, 2, 3)
                    e["references"]["observations"] = write(
                        origin + "-initial-context", np.where(im, context, np.nan)
                    )
                    e["description"] += (
                        f" IAP {saved['initial_context_month']} is monthly context, not instantaneous truth; no velocity observations."
                    )
                else:
                    e["description"] += " Preceding-month IAP context unavailable."
                meta["interior_examples"][origin] = e
            else:
                record = meta["records"][origin]
                np.testing.assert_array_equal(
                    np.where(sm, saved["reference_surface"], np.nan),
                    np.load(output / record["surface_reference"]),
                )
                assert saved["months"].tolist() == record["heat_months"]
                np.testing.assert_array_equal(
                    np.where(hm, saved["reference_monthly_ohc"] / 1e9, np.nan),
                    np.load(output / record["heat_reference"]),
                )
                reference = np.full((12, 19, 2, *sm.shape[-2:]), np.nan, np.float32)
                reference[:, :14] = saved["reference_monthly_ts"].transpose(
                    0, 2, 1, 3, 4
                )
                np.testing.assert_array_equal(
                    np.where(tsmask, reference, np.nan),
                    np.load(output / record["monthly_interior_reference"]),
                )
            initial = np.where(im, saved["initial"][ids], np.nan)
            monthly = np.where(
                tsmask, saved["monthly_ts"].transpose(0, 2, 1, 3, 4), np.nan
            )
            surface = np.where(sm, saved["surface"], np.nan)
            assert surface.shape == (73, 2, *sm.shape[-2:]) and monthly.shape == (
                12,
                19,
                2,
                *sm.shape[-2:],
            )
            assert (
                np.isfinite(surface[:, sm]).all()
                and np.isfinite(initial[im]).all()
                and np.isfinite(monthly[:, tsmask]).all()
            )
            heat = np.where(hm, saved["monthly_ohc"] / 1e9, np.nan)
            # Independently check native-layer heat integration against exported T.
            for j, (lo, hi) in enumerate([(0, 700), (700, 2000)]):
                dz = np.maximum(
                    0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)
                )
                used = dz > 0
                recomputed = (
                    (monthly[:, used, 0].astype(np.float64) * dz[used, None, None]).sum(
                        1
                    )
                    * 1035
                    * 3850
                    / 1e9
                )
                np.testing.assert_allclose(
                    recomputed[:, hm[j]], heat[:, j, hm[j]], rtol=1e-12, atol=1e-10
                )
            prefix = name + "-" + origin
            row = dict(
                interior=write(prefix + "-initial", initial),
                surface=write(prefix + "-surface", surface),
                surface_persistence=write(
                    prefix + "-surface-held",
                    np.broadcast_to(
                        np.where(sm, saved["initial"][[38, 76]], np.nan), surface.shape
                    ),
                ),
                forecast_interior=write(
                    prefix + "-forecast-interior",
                    np.where(im, saved["forecast_interior"][:, ids], np.nan),
                ),
                monthly_interior=write(prefix + "-monthly-ts", monthly),
                monthly_interior_persistence=write(
                    prefix + "-monthly-held",
                    np.broadcast_to(initial[:, :2], monthly.shape),
                ),
                temperature=write(prefix + "-temperature", monthly[:, :, 0]),
                heat=write(prefix + "-heat", heat),
            )
            meta["records"][origin]["models"][name] = row
            meta["interior_examples"][origin]["models"][name] = row["interior"]
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, info in meta["files"].items():
        assert sha(output / name) == info["sha256"]
    print("Verified", len(meta["models"]), "models and", len(meta["files"]), "arrays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
