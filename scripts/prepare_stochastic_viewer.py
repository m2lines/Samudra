# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert verified stochastic rollouts to a standalone immutable viewer catalog."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPORT = "https://github.com/m2lines/Samudra/blob/codex/stochastic-transition-gallery/docs/experiments/stochastic-transitions/results.md"
NOISE_REPORT = "https://github.com/m2lines/Samudra/blob/codex/stochastic-transition-gallery/docs/experiments/stochastic-transitions/noise-ablation-results.md"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source, output, report_url=None, *, profile="C-final"):
    receipt = json.loads((source / "COMPLETE.json").read_text())
    if profile not in ("C-final", "learned-noise-2k"):
        raise ValueError("Unknown viewer checkpoint profile")
    learned = profile == "learned-noise-2k"
    if not (
        receipt["complete"]
        and receipt["arm"] == ("B" if learned else "C")
        and receipt["weights"] == "ema"
        and receipt["step"] == (2000 if learned else 5000)
    ):
        raise ValueError("Export does not match the requested checkpoint profile")
    if learned:
        contract = receipt["contract"]
        if (
            receipt["variant"] != "B-noise-processing"
            or receipt["counts"] != dict(om4=1000, observation=1000)
            or contract["noise_processing_blocks"] != 3
            or contract.get("noise_mode", "iid") != "iid"
        ):
            raise ValueError("Learned-noise export lineage differs")
        if sorted(receipt["origins"]) != ["2013-11", "2014-01", "2014-07"]:
            raise ValueError("Learned-noise observation origin cohort differs")
    report_url = report_url or (NOISE_REPORT if learned else REPORT)
    prefix = "stochastic-B-noise-processing-2k-" if learned else "stochastic-C-final-"
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(
        schema_version=1,
        report_url=report_url,
        models={},
        origins=[],
        records={},
        interior_examples={},
        files={},
        forecast_interior_leads_days=[5, 10, 15, 20, 25, 30],
        export_receipt=receipt,
    )

    def write(name, value):
        path = output / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
        meta["files"][path.name] = dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=sha(path),
            bytes=path.stat().st_size,
        )
        return path.name

    for readout, label in [
        ("mean8", "mean of 8"),
        ("draw1", "member 1"),
        ("draw2", "member 2"),
    ]:
        key = prefix + readout
        meta["models"][key] = dict(
            label=f"Stochastic ConvNeXt C · {label}",
            short_label=f"C · {label}",
            report_url=report_url,
            description=f"**Stochastic ConvNeXt C · {label}.** One-pass Samudra ConvNeXt initializer and residual physical-state processor with fresh stochastic FiLM features. 19 five-day surface-history bins initialize two physical states; eight ERA5 fields pass through a learned 8→32→3 adapter. Both initializer and processor are stochastic. Members retain separate autoregressive physical-state trajectories; these are not independent decoder draws from one deterministic trajectory.\n\nTrained from scratch for 4,000 mixed optimizer updates (2,000 OM4 + 2,000 observations), followed by 1,000 observation-only cooldown updates; effective batch 8. Final EMA at step 5,000, no replay. Displayed five-day bins span days 5–30. The mean is across eight trajectories; members 1 and 2 are two retained trajectories from that same ensemble.\n\nInitialization is freely predicted, without surface anchoring. Contemporaneous published-v3 OM4 is simulation context, not observational truth, and differs from the staged training-store version. No instantaneous observed interior reference is available. No spatial smoothing is applied.",
            lineage=dict(
                checkpoint=receipt["checkpoint"],
                checkpoint_sha256=receipt["checkpoint_sha256"],
                global_step=5000,
                task_counts=dict(om4=2000, observation=3000),
                weights="ema",
                readout=readout,
            ),
        )
        if learned:
            meta["models"][key].update(
                label=f"Learned-noise B · EMA 2k · {label}",
                short_label=f"B learned-noise · {label}",
                description=f"**Learned-noise B · EMA 2k · {label}.** Samudra ConvNeXt stochastic initializer with three learned ConvNeXt blocks processing spatial noise before feature modulation; deterministic residual physical-state processor. Uncertainty is sampled at initialization. Each sampled initial state follows its own deterministic autoregressive trajectory under shared ERA5; no fresh processor noise is drawn.\n\n"
                "Trained from scratch for 2,000 mixed optimizer updates: 1,000 OM4 and 1,000 observation updates, effective batch eight. No cooldown or replay. This is the EMA checkpoint, with four stochastic members during training and eight for these saved evaluations.\n\n"
                "The initializer receives 19 five-day SST/SSH history bins and eight ERA5 fields through a learned 8→32→3 adapter. It freely predicts two 77-channel physical states; surface outputs are not anchored to the inputs. Mean of eight averages the same eight trajectories from which members 1 and 2 are retained. The five-day forecasts cover days 5–30. No spatial smoothing is applied.\n\n"
                "Observation-initialized interiors have no instantaneous observational truth. Any contemporaneous OM4 comparison is simulation context. OM4-initialized examples, when available, use OM4 surface history and compare against exactly matched interior simulation gold; that interior gold is not an input.",
                lineage=dict(
                    checkpoint=receipt["checkpoint"],
                    checkpoint_sha256=receipt["checkpoint_sha256"],
                    global_step=2000,
                    task_counts=dict(om4=1000, observation=1000),
                    weights="ema",
                    readout=readout,
                ),
            )
    for month, provenance in receipt["origins"].items():
        loaded = {}
        for readout in ["mean8", "draw1", "draw2"]:
            path = source / f"{month}-{readout}.npz"
            assert sha(path) == provenance["files"][readout]["sha256"]
            with np.load(path, allow_pickle=False) as saved:
                loaded[readout] = dict(saved)
        base = loaded["mean8"]
        starts = base["starts"].astype("datetime64[s]")
        ends = base["ends"].astype("datetime64[s]")
        np.testing.assert_array_equal(
            np.diff(starts), np.full(6, np.timedelta64(5, "D"))
        )
        np.testing.assert_array_equal(ends - starts, np.full(7, np.timedelta64(5, "D")))
        if np.any(starts != starts.astype("datetime64[D]")):
            raise ValueError(
                "Viewer date labels require midnight-aligned five-day bins"
            )
        origin = str(starts[1].astype("datetime64[D]"))
        if origin in meta["origins"]:
            raise ValueError("Duplicate forecast start")
        meta["origins"].append(origin)
        names = base["channel_names"].tolist()
        if learned and names != [
            f"{v}_{d}" for v in ["uo", "vo", "thetao", "so"] for d in range(19)
        ] + ["zos"]:
            raise ValueError("Learned-noise viewer requires the full 77-channel state")
        ids = np.array(
            [
                [names.index(f"{var}_{i}") for var in ["thetao", "so", "uo", "vo"]]
                for i in range(len(base["depth"]))
            ]
        )
        surface_ids = [names.index("thetao_0"), names.index("zos")]
        if "lat" not in meta:
            meta.update(
                lat=base["lat"].tolist(),
                lon=base["lon"].tolist(),
                depths=base["depth"].tolist(),
                depth_interfaces=base["depth_interfaces"].tolist(),
                surface_mask=write(
                    "surface-mask", base["mask"][surface_ids].astype(bool)
                ),
                interior_mask=write("interior-mask", base["mask"][ids].astype(bool)),
            )
        np.testing.assert_array_equal(base["leads"], np.arange(0, 31, 5))
        models = {}
        initial_models = {}
        for readout, saved in loaded.items():
            for coord in [
                "lat",
                "lon",
                "depth",
                "depth_interfaces",
                "channel_names",
                "leads",
                "mask",
                "surface_reference",
                "starts",
                "ends",
                "midpoints",
                "units",
            ]:
                np.testing.assert_array_equal(saved[coord], base[coord])
            np.testing.assert_array_equal(saved["lat"], meta["lat"])
            np.testing.assert_array_equal(saved["lon"], meta["lon"])
            if learned:
                if not np.isfinite(
                    saved["fields"][:, saved["mask"].astype(bool)]
                ).all():
                    raise ValueError("Nonfinite prediction on model wet support")
            values = np.where(
                saved["mask"].astype(bool), saved["fields"], np.nan
            ).astype(np.float32)
            assert values.shape == (7, len(names), len(meta["lat"]), len(meta["lon"]))
            key = prefix + readout
            initial = write(f"{month}-{readout}-initial", values[0][ids])
            models[key] = dict(
                surface=write(f"{month}-{readout}-surface", values[1:][:, surface_ids]),
                surface_persistence=write(
                    f"{month}-{readout}-held-surface",
                    np.broadcast_to(
                        values[0, surface_ids], values[1:][:, surface_ids].shape
                    ),
                ),
                interior=initial,
                forecast_interior=write(
                    f"{month}-{readout}-forecast-interior", values[1:][:, ids]
                ),
            )
            initial_models[key] = initial
        references = {}
        if provenance.get("om4") and provenance["om4"].get("sha256"):
            om4_path = source / f"{month}-om4-comparison.npz"
            assert sha(om4_path) == provenance["om4"]["sha256"]
            with np.load(om4_path, allow_pickle=False) as om4:
                for coord in ["lat", "lon", "channel_names", "midpoints"]:
                    np.testing.assert_allclose(
                        om4[coord], base[coord], rtol=0, atol=1e-6
                    ) if coord in ["lat", "lon"] else np.testing.assert_array_equal(
                        om4[coord], base[coord]
                    )
                reference = np.where(
                    om4["mask"].astype(bool), om4["fields"][0], np.nan
                )[ids].astype(np.float32)
            references["om4"] = write(f"{month}-om4-initial-context", reference)
        meta["interior_examples"][origin] = dict(
            origin=origin,
            label=f"{origin} (obs; {month} validation cohort)",
            source="obs",
            models=initial_models,
            references=references,
            description=(
                "Exact contemporaneous OM4 context is available. "
                if references
                else "No exact contemporaneous OM4 reference is available for this origin. "
            )
            + "Initialization from observation surface history and shared ERA5. All displayed fields are predictions, not copied inputs. OM4 is contemporaneous simulation context for the final five-day history bin, not observational ground truth. No instantaneous observed interior reference is available. Each member supplies the state used to begin its own subsequent stochastic rollout.",
        )
        meta["records"][origin] = dict(
            origin=origin,
            source="obs",
            label=f"{origin} ({month} validation cohort)",
            validation_cohort=month,
            models=models,
            surface_reference=write(
                f"{month}-surface-reference",
                np.where(
                    base["mask"][surface_ids], base["surface_reference"][1:], np.nan
                ).astype(np.float32),
            ),
            surface_readout_description="Six saved five-day means through day 30. Mean of eight trajectories, or the selected coherent member trajectory.",
            forecast_interior_readout_description="Six saved five-day states at days 5, 10, 15, 20, 25 and 30; own initialized persistence and the other retained trajectories are available for comparison.",
            time_starts=base["starts"].tolist(),
            time_ends=base["ends"].tolist(),
            seed=provenance["seed"],
        )
        if learned:
            meta["interior_examples"][origin]["description"] = (
                "Initialization from observation surface history and ERA5. "
                + (
                    "OM4 is contemporaneous simulation context."
                    if references
                    else "No instantaneous interior reference is available."
                )
            )
            meta["records"][origin]["surface_readout_description"] = (
                "Mean of eight trajectories, or one retained member; six five-day means through day 30."
            )
            meta["records"][origin]["forecast_interior_readout_description"] = (
                "Mean of eight trajectories, or one retained member; saved five-day states through day 30."
            )
    if learned and receipt.get("om4_initializations"):
        add_om4_initializations(source, receipt, meta, prefix, write, base)
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def add_om4_initializations(source, receipt, meta, prefix, write, reference_grid):
    """Expose reconstruction gold without inventing unsaved OM4 forecasts."""
    directory = source / "om4-initialization"
    entries = receipt["om4_initializations"]
    if sorted(entries) != ["2014-01", "2014-04", "2014-05"]:
        raise ValueError("OM4 initialization cohort differs")
    complete = json.loads((directory / "COMPLETE.json").read_text())
    if not complete["complete"] or complete["origins"] != entries:
        raise ValueError("OM4 initialization receipt differs")
    for month, entry in entries.items():
        if (
            entry["weights"] != "ema"
            or entry["checkpoint_sha256"] != receipt["checkpoint_sha256"]
            or entry["members"] != 8
        ):
            raise ValueError("OM4 initializer checkpoint/ensemble differs")
        loaded = {}
        for readout in ["mean8", "draw1", "draw2", "gold"]:
            path = directory / f"{month}-{readout}.npz"
            if sha(path) != entry["files"][readout]["sha256"]:
                raise ValueError(f"OM4 initialization checksum differs: {path}")
            with np.load(path, allow_pickle=False) as data:
                loaded[readout] = dict(data)
        base = loaded["gold"]
        for key in ["mask", "channel_names", "units"]:
            np.testing.assert_array_equal(base[key], reference_grid[key])
        for key, meta_key in [
            ("lat", "lat"),
            ("lon", "lon"),
            ("depth", "depths"),
            ("depth_interfaces", "depth_interfaces"),
        ]:
            np.testing.assert_array_equal(base[key], meta[meta_key])
        names = base["channel_names"].tolist()
        ids = np.array(
            [
                [names.index(f"{var}_{i}") for var in ["thetao", "so", "uo", "vo"]]
                for i in range(len(base["depth"]))
            ]
        )
        ends = base["ends"].astype("datetime64[s]")
        starts = base["starts"].astype("datetime64[s]")
        np.testing.assert_array_equal(ends - starts, np.array([np.timedelta64(5, "D")]))
        if np.any(ends != ends.astype("datetime64[D]")):
            raise ValueError("OM4 initialization interval must end at midnight")
        date = str(ends[0].astype("datetime64[D]"))
        origin = date + "-om4"
        if origin in meta["origins"]:
            raise ValueError("Duplicate OM4 initialization origin")
        models = {}
        reference = None
        for readout, saved in loaded.items():
            for key in [
                "lat",
                "lon",
                "depth",
                "depth_interfaces",
                "channel_names",
                "mask",
                "starts",
                "ends",
                "midpoints",
                "leads",
                "units",
            ]:
                np.testing.assert_array_equal(saved[key], base[key])
            if saved["fields"].shape != (
                1,
                len(names),
                len(meta["lat"]),
                len(meta["lon"]),
            ):
                raise ValueError("Expected one full OM4 initialization state")
            if not np.isfinite(saved["fields"][:, saved["mask"].astype(bool)]).all():
                raise ValueError("Nonfinite OM4 initialization on wet support")
            values = np.where(saved["mask"].astype(bool), saved["fields"][0], np.nan)[
                ids
            ].astype(np.float32)
            filename = write(f"{month}-om4-{readout}-initial", values)
            if readout == "gold":
                reference = filename
            else:
                models[prefix + readout] = filename
        meta["origins"].append(origin)
        meta["records"][origin] = dict(
            origin=date,
            source="om4",
            label=f"{date} (OM4 initialization; {month})",
            models={key: dict(interior=value) for key, value in models.items()},
        )
        meta["interior_examples"][origin] = dict(
            origin=date,
            source="om4",
            label=f"{date} (OM4 initialization; {month})",
            models=models,
            references=dict(om4=reference),
            description="Initialization from OM4 surface history and ERA5. The comparison is the exactly matched OM4 interior gold; it is not supplied to the initializer.",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-url")
    parser.add_argument(
        "--profile", choices=["C-final", "learned-noise-2k"], default="C-final"
    )
    args = parser.parse_args()
    prepare(args.source, args.output, args.report_url, profile=args.profile)
