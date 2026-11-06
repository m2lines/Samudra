# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Build an additive catalog for signed direct-ERA5 8k mean and fixed-member outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPORT = "https://github.com/m2lines/Samudra/blob/codex/stochastic-transition-gallery/docs/experiments/stochastic-transitions/signed-dense-results.md"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(source, output, kind_filter=None):
    receipt = json.loads((source / "COMPLETE.json").read_text())
    if (
        not receipt["complete"]
        or receipt["readouts"] != ["mean8", "draw1"]
        or receipt["checkpoint_sha256"]
        != "e28fb15f5e71cbb22f45cd5c974c518b296b74563add6ddc7763a605f2e82200"
    ):
        raise ValueError("Export receipt is incomplete")
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(
        schema_version=1,
        report_url=REPORT,
        models={},
        origins=[],
        records={},
        interior_examples={},
        files={},
        export_receipt=receipt,
        heat_unit="GJ/m²",
        heat_input_unit="J/m²",
        heat_depth_bounds_m=[[0, 700], [700, 2000]],
    )

    def write(name, values):
        path = output / (name + ".npy")
        np.save(path, values, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
        meta["files"][path.name] = dict(
            shape=list(values.shape),
            dtype=str(values.dtype),
            sha256=sha(path),
            bytes=path.stat().st_size,
        )
        return path.name

    def read(directory, record):
        path = directory / Path(record["path"]).name
        if sha(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            return dict(z)

    def masked(x, mask):
        return np.where(mask, np.asarray(x, dtype=np.float32), np.nan)

    for export_key, entry in receipt["exports"].items():
        if kind_filter and not export_key.startswith(kind_filter + "-"):
            continue
        if entry.get("available") is False:
            meta.setdefault("unavailable_exports", {})[export_key] = entry
            continue
        kind, origin_stamp = export_key.split("/", 1)
        year = origin_stamp[:10]
        directory = source / kind / year
        is_om4 = kind.startswith("om4-")
        source_label = "om4" if is_om4 else "obs"
        step = int(entry["step"])
        if (
            step != 8000
            or entry["weights"] != "ema"
            or entry["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Unexpected checkpoint")
        data = {key: read(directory, entry["files"][key]) for key in ["mean8", "draw1"]}
        base = data["mean8"]
        names = base["channel_names"].tolist()
        n_depth = len(base["depth"])
        ids = np.array(
            [
                [names.index(f"{v}_{d}") for v in ["thetao", "so", "uo", "vo"]]
                for d in range(n_depth)
            ]
        )
        surface_ids = [names.index("thetao_0"), names.index("zos")]
        mask = base["mask"].astype(bool)
        interior_mask = mask[ids]
        edges = base["depth_interfaces"]
        heatmask = np.stack(
            [
                interior_mask[(edges[:-1] < end) & (edges[1:] > start), 0].all(0)
                for start, end in [(0, 700), (700, 2000)]
            ]
        )
        if "lat" not in meta:
            meta.update(
                lat=base["lat"].tolist(),
                lon=base["lon"].tolist(),
                depths=base["depth"].tolist(),
                depth_interfaces=edges.tolist(),
                surface_mask=write("surface-mask", mask[surface_ids]),
                interior_mask=write("interior-mask", interior_mask),
                heat_mask=write("heat-mask", heatmask),
            )
        for field, target in [
            ("lat", "lat"),
            ("lon", "lon"),
            ("depth", "depths"),
            ("depth_interfaces", "depth_interfaces"),
        ]:
            np.testing.assert_array_equal(base[field], meta[target])
        np.testing.assert_array_equal(
            interior_mask, np.load(output / meta["interior_mask"])
        )
        np.testing.assert_array_equal(
            mask[surface_ids], np.load(output / meta["surface_mask"])
        )
        starts = base["starts"].astype("datetime64[s]")
        ends = base["ends"].astype("datetime64[s]")
        if len(starts):
            np.testing.assert_array_equal(
                ends - starts, np.full(len(starts), np.timedelta64(5, "D"))
            )
            np.testing.assert_array_equal(
                np.diff(starts),
                np.full(max(0, len(starts) - 1), np.timedelta64(5, "D")),
            )
            if np.any(starts != starts.astype("datetime64[D]")):
                raise ValueError("Non-midnight forecast bins")
            start_date = str(starts[0].astype("datetime64[D]"))
        else:
            start_date = year
        origin = start_date + ("-om4" if is_om4 else "")
        if origin in meta["origins"]:
            raise ValueError("Duplicate origin")
        meta["origins"].append(origin)
        record = dict(
            origin=start_date,
            source=source_label,
            label=f"{start_date} ({source_label}; EMA {step})",
            models={},
            forecast_interior_leads_days=base["interior_leads"].tolist(),
            starts=base["starts"].tolist(),
            ends=base["ends"].tolist(),
            complete=entry["complete"],
            completed_bins=entry["completed_bins"],
            first_nonfinite=entry["first_nonfinite"],
        )
        initial_models = {}
        references = {}
        refs = read(directory, entry["gold"] if is_om4 else entry["references"])

        def ts_pad(ts):
            values = np.asarray(ts, dtype=np.float32)
            if values.shape[-3] > n_depth:
                raise ValueError("Reference has too many depths")
            padded = np.full(
                (*values.shape[:-3], n_depth, *values.shape[-2:]),
                np.nan,
                dtype=np.float32,
            )
            padded[..., : values.shape[-3], :, :] = values
            return np.swapaxes(padded, -4, -3)

        # A partial calendar month is all-NaN in the producer, never a shorter mean.
        keep = np.isfinite(base["monthly_ts"]).any(axis=(1, 2, 3, 4))
        months = base["months"][keep].tolist()
        record["heat_months"] = months
        for saved in data.values():
            for key in [
                "channel_names",
                "lat",
                "lon",
                "depth",
                "depth_interfaces",
                "mask",
                "leads",
                "interior_leads",
                "months",
                "starts",
                "ends",
            ]:
                np.testing.assert_array_equal(saved[key], base[key])
            np.testing.assert_array_equal(
                np.isfinite(saved["monthly_ts"]).any(axis=(1, 2, 3, 4)), keep
            )
        if len(starts):
            reference_surface = refs["surface"] if is_om4 else refs["reference_surface"]
            record["surface_reference"] = write(
                f"{origin}-surface-reference",
                masked(reference_surface[: len(starts)], mask[surface_ids]),
            )
        if is_om4:
            references["om4"] = write(
                f"{origin}-initial-gold", masked(refs["initial"], mask)[ids]
            )
            if len(base["interior_leads"]):
                positions = [
                    refs["interior_leads"].tolist().index(x)
                    for x in base["interior_leads"]
                ]
                record["forecast_interior_reference"] = write(
                    f"{origin}-forecast-gold",
                    masked(refs["forecast_interior"][positions], mask)[:, ids],
                )
        elif "initial_iap_ts" in refs:
            initial_ref = np.full(
                (n_depth, 4, *mask.shape[-2:]), np.nan, dtype=np.float32
            )
            initial_ref[:, :2] = ts_pad(refs["initial_iap_ts"])
            references["observations"] = write(
                f"{origin}-initial-iap-context", masked(initial_ref, interior_mask)
            )
            record["context_month"] = str(refs["initial_context_month"])
        if months:
            ref_ts = refs["monthly_ts"] if is_om4 else refs["reference_monthly_ts"]
            ref_ohc = refs["monthly_ohc"] if is_om4 else refs["reference_monthly_ohc"]
            np.testing.assert_array_equal(refs["months"], base["months"])
            ts = masked(ts_pad(ref_ts[keep]), interior_mask[:, :2])
            record["monthly_interior_reference"] = write(
                f"{origin}-monthly-ts-reference", ts
            )
            record["temperature_reference"] = write(
                f"{origin}-monthly-temperature-reference", ts[:, :, 0]
            )
            record["heat_reference"] = write(
                f"{origin}-monthly-ohc-reference", masked(ref_ohc[keep] / 1e9, heatmask)
            )
        for readout, saved in data.items():
            key = "signed-direct8-dense8k-061126-" + (
                "mean8" if readout == "mean8" else "member1"
            )
            label = {"mean8": "mean of 8", "draw1": "member 1", "draw2": "member 2"}[
                readout
            ]
            meta["models"].setdefault(
                key,
                dict(
                    label=f"Signed direct-ERA5 · 8k EMA · {label}",
                    short_label=f"Signed 8k {label}",
                    report_url=REPORT,
                    description=f"**Signed direct-ERA5 · 8k EMA · {label}.** Stochastic Samudra ConvNeXt initializer with learned shared coarse/fine noise and a deterministic residual Samudra ConvNeXt processor. Each member samples its initial state once, then follows its own deterministic trajectory under shared prescribed ERA5. Mean of eight averages those same eight trajectories; member 1 is fixed through initialization and every later lead.\n\nFinal EMA after 8,000 mixed optimizer updates (4,000 OM4 and 4,000 observation), effective batch 8, with signed-increment CRPS on OM4 initialization velocities. Direct eight-channel ERA5 inputs, 2,548 overlapping OM4 windows; no replay or cooldown. Initializer widths 96/192/384, three ConvNeXt blocks per stage; processor widths 64/128/256, one block per stage; InstanceNorm.\n\nInitialization uses 95 days of SST/SSH surface history with matching ERA5 and predicts two complete ocean states. No interior reference or future ocean surface observations enter inference. Origin labels distinguish observational and OM4 surface inputs. OM4 references are exactly matched simulation fields; IAP initialization references are preceding-month context, not instantaneous truth. Monthly T/S and OHC are calendar-month averages; saved forecast interiors are five-day bins. No clipping, damping, or spatial smoothing is applied. Annual rollouts remain finite but develop localized nonphysical extremes by day 365, including negative salinity and temperatures outside plausible ocean ranges. These raw outputs are preserved without clipping; finite completion does not establish physical stability.",
                    lineage=dict(
                        checkpoint=entry["checkpoint"],
                        checkpoint_sha256=entry["checkpoint_sha256"],
                        global_step=step,
                        weights="ema",
                        readout=readout,
                        source="OBS or OM4 surface history; see origin",
                    ),
                ),
            )
            initial = masked(saved["initial"], mask)
            initial_name = write(f"{origin}-{readout}-initial", initial[ids])
            initial_models[key] = initial_name
            row = dict(interior=initial_name)
            if len(starts):
                surface = masked(saved["surface"], mask[surface_ids])
                row.update(
                    surface=write(f"{origin}-{readout}-surface", surface),
                    surface_persistence=write(
                        f"{origin}-{readout}-held-surface",
                        np.broadcast_to(initial[surface_ids], surface.shape),
                    ),
                )
            if len(saved["interior_leads"]):
                row["forecast_interior"] = write(
                    f"{origin}-{readout}-forecast-interior",
                    masked(saved["forecast_interior"], mask)[:, ids],
                )
            if months:
                ts = masked(ts_pad(saved["monthly_ts"][keep]), interior_mask[:, :2])
                row.update(
                    monthly_interior=write(f"{origin}-{readout}-monthly-ts", ts),
                    monthly_interior_persistence=write(
                        f"{origin}-{readout}-held-monthly-ts",
                        np.broadcast_to(initial[ids][:, :2], ts.shape),
                    ),
                    temperature=write(
                        f"{origin}-{readout}-monthly-temperature", ts[:, :, 0]
                    ),
                    heat=write(
                        f"{origin}-{readout}-monthly-ohc",
                        masked(saved["monthly_ohc"][keep] / 1e9, heatmask),
                    ),
                )
            record["models"][key] = row
        if not entry["complete"]:
            warning = f"Rollout stopped after {entry['completed_bins']} finite five-day bins; later states are unavailable. Only completed calendar months are shown."
            record["surface_readout_description"] = record[
                "forecast_interior_readout_description"
            ] = record["monthly_interior_readout_description"] = warning
        meta["records"][origin] = record
        meta["interior_examples"][origin] = dict(
            origin=start_date,
            label=record["label"],
            source=source_label,
            models=initial_models,
            references=references,
            description=(
                "OM4 surface-history-only initialization; reference is its exact full-state gold for the last history bin. Interior gold is withheld from model input."
                if is_om4
                else (
                    f"Observation-surface-history initialization. IAP {record['context_month']} is monthly context, not instantaneous interior truth; it contains no velocity reference."
                    if "observations" in references
                    else "Observation-surface-history initialization. Preceding-December IAP context is unavailable for this origin; no instantaneous interior observation reference is supplied."
                )
            )
            + f" The predicted state belongs to the same stochastic member used in its rollout. Initial bin: {str(base['initial_start'])} to {str(base['initial_end'])}, exclusive end.",
        )
    if not meta["origins"]:
        raise ValueError("No matching exported origins")
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["obs", "om4"])
    args = p.parse_args()
    prepare(args.source, args.output, args.kind)
