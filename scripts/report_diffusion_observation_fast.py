#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify the complete three-arm observation pilot before rendering comparisons.

Reuses the intervention report's exact annual metric and scored spectral bins.
Partial cohorts are rejected before any output is written. No model execution.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import report_diffusion_direct as direct  # type: ignore[import-not-found]
import report_diffusion_interventions as common  # type: ignore[import-not-found]

from samudra.experiments.diffusion_ab_figures import save_png

ARMS = ("control", "fast", "hybrid")
NAMES = ("parent", *ARMS)
ORIGINS = tuple(common.ORIGINS)
FIELDS = (
    ("sst", "°C", (-2, 32)),
    ("ssh", "m", (-1, 2)),
    ("t550", "°C", (0, 15)),
    ("s550", "salinity", (33, 36)),
)


def verify(args):
    roots = {"parent": args.parent, **{a: args.root / a for a in ARMS}}
    missing = [
        str(root / section / "COMPLETE.json")
        for root in roots.values()
        for section in ("annual", "native", "validation")
        if not (root / section / "COMPLETE.json").is_file()
    ]
    if missing:
        raise ValueError(
            "Incomplete comparison cohort; no report generated: " + ", ".join(missing)
        )
    protocols = []
    evidence = {}
    for name, root in roots.items():
        annual = common.read(root / "annual/input.json")
        complete = common.read(root / "annual/COMPLETE.json")
        native = common.read(root / "native/COMPLETE.json")
        validation = common.read(root / "validation/COMPLETE.json")
        counts = {
            "om4": 8128 if name == "parent" else 8192,
            "observation": 8128 if name == "parent" else 8192,
        }
        if complete["inputs"] != annual or set(complete["origins"]) != set(ORIGINS):
            raise ValueError(f"Incomplete or mismatched annual receipt: {name}")
        if annual["counts"] != counts or annual["step"] != sum(counts.values()):
            raise ValueError(f"Wrong checkpoint counts: {name}")
        if annual["members"] != 8 or annual["sampling_steps"] != 32:
            raise ValueError(f"Wrong sampler settings: {name}")
        for receipt in (native, validation):
            if (
                receipt["checkpoint_sha256"] != annual["checkpoint_sha256"]
                or receipt["counts"] != counts
            ):
                raise ValueError(f"Cross-evaluator checkpoint mismatch: {name}")
            if receipt["sampling_steps"] != 32:
                raise ValueError(f"Cross-evaluator sampler mismatch: {name}")
        protocols.append(
            {
                "annual": {
                    k: annual[k] for k in ("annual_manifests", "seed", "domain")
                },
                "native": {k: native[k] for k in ("seed", "members", "leads_days")},
                "validation_origins": validation["metrics"]["origins"],
            }
        )
        evidence[name] = dict(
            checkpoint_sha256=annual["checkpoint_sha256"], counts=counts
        )
        if name != "parent":
            intervention = annual["intervention"]
            if (
                intervention["campaign"] != "obs-fast"
                or intervention["arm"] != name
                or intervention["parent_sha256"]
                != evidence["parent"]["checkpoint_sha256"]
            ):
                raise ValueError(f"Wrong continuation lineage: {name}")
        members = common.read(root / "annual/MEMBERS_COMPLETE.json")
        if members["inputs"] != annual:
            raise ValueError(f"Mismatched member receipt: {name}")
        # Verify plotted payloads, including the parent in supplemental spectra.
        for section, receipt, files in (
            ("annual", members, [f"{args.origin}.npz", f"members-{args.origin}.npz"]),
            ("native", native, [f"fields-{args.origin[:4]}.npz"]),
            ("validation", validation, list(validation["files"])),
        ):
            for filename in files:
                if (
                    direct.digest(root / section / filename)
                    != receipt["files"][filename]
                ):
                    raise ValueError(
                        f"Export checksum mismatch: {root / section / filename}"
                    )
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError(
            "Different data manifests, origins, masks/domain or random seeds"
        )
    return roots, evidence


def calibration(roots, output):
    rows = []
    for arm, root in roots.items():
        statistics = common.read(root / "validation/calibration.json")
        for group, section, channels, indices in (
            ("SST", "surface", 2, [0]),
            ("SSH", "surface", 2, [1]),
            ("T", "interior", 28, list(range(14))),
            ("S", "interior", 28, list(range(14, 28))),
        ):
            sums = {
                key: sum(
                    np.asarray(s[section][key]).reshape(-1, channels)[:, indices].sum()
                    for s in statistics
                )
                for key in (
                    "weight",
                    "mean_squared_error",
                    "ensemble_variance",
                    "fair_crps",
                    "coverage_90",
                )
            }
            rows.append(
                dict(
                    arm=arm,
                    group=group,
                    spread_rmse_ratio=float(
                        np.sqrt(sums["ensemble_variance"] / sums["mean_squared_error"])
                    ),
                    coverage90=float(sums["coverage_90"] / sums["weight"]),
                    fair_crps=float(sums["fair_crps"] / sums["weight"]),
                )
            )
    common.table(output / "calibration-with-parent.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    groups = ("SST", "SSH", "T", "S")
    for ax, key, ideal, title in zip(
        axes,
        ("spread_rmse_ratio", "coverage90"),
        (1, 0.9),
        ("Spread / ensemble-mean RMSE", "Nominal 90% interval coverage"),
        strict=True,
    ):
        for arm, color, marker in zip(
            NAMES, direct.COLORS, direct.MARKERS, strict=True
        ):
            values = [
                next(r[key] for r in rows if r["arm"] == arm and r["group"] == group)
                for group in groups
            ]
            ax.plot(groups, values, label=arm, color=color, marker=marker)
        ax.axhline(ideal, color="0.3", linewidth=1)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Nine validation months; pointwise calibration from eight conditional readouts\n"
        "Interior values are calendar averages; not trajectory uncertainty"
    )
    save_png(fig, output / "calibration.png", dpi=130)
    plt.close(fig)


def timing(args, output):
    execution = common.read(args.execution_results)
    by_arm = {record["arm"]: record for record in execution["arms"]}
    if set(by_arm) != set(ARMS) or not all(
        record["success"] for record in by_arm.values()
    ):
        raise ValueError("Production execution receipt lacks three successful arms")
    rows, update_rows = [], []
    for arm in ARMS:
        done = common.read(args.runs / arm / "COMPLETE.json")
        if (
            not done["complete"]
            or done["updates"] != 128
            or done["counts"] != {"om4": 8192, "observation": 8192}
        ):
            raise ValueError(f"Incomplete training receipt: {arm}")
        # COMPLETE hashes last.pt (optimizer/RNG/replay included); evaluation
        # reads the separate step-0128.pt snapshot. Verify that actual file's
        # independently recorded digest rather than equating the two archives.
        snapshot = common.read(args.runs / arm / "snapshot-sha256.json")
        if snapshot["file"] != "step-0128.pt" or (
            snapshot["sha256"]
            != common.read(args.root / arm / "annual/input.json")["checkpoint_sha256"]
        ):
            raise ValueError(f"Training/evaluation checkpoint mismatch: {arm}")
        if by_arm[arm]["timings"]["training"] + 1 < done["seconds"]:
            raise ValueError(
                f"Execution timing predates the completed training run: {arm}"
            )
        events = [
            json.loads(line)
            for line in (args.runs / arm / "events.jsonl").read_text().splitlines()
        ]
        events = [event for event in events if event.get("event") == "update"]
        if [event["update"] for event in events] != list(range(1, 129)):
            raise ValueError(f"Missing or duplicated training events: {arm}")
        for event in events:
            mode = event.get("observation_mode", "OM4")
            update_rows.append(
                dict(
                    arm=arm,
                    update=event["update"],
                    task=event["task"],
                    mode=mode,
                    seconds=event["seconds"],
                )
            )
        expected_fast = {"control": 0, "fast": 64, "hybrid": 48}[arm]
        if sum(e.get("observation_mode") == "fast" for e in events) != expected_fast:
            raise ValueError(f"Wrong observation objective schedule: {arm}")
        for label, seconds in by_arm[arm]["timings"].items():
            rows.append(
                dict(
                    arm=arm,
                    stage=label,
                    seconds=seconds,
                    note="Wall time; waiting is not additional active GPU compute"
                    if label == "wait_for_cache"
                    else "Single-GPU subprocess wall time including startup",
                )
            )
        rows.append(
            dict(
                arm=arm,
                stage="training_loop",
                seconds=done["seconds"],
                note="Trainer elapsed time; excludes process startup and cache preparation",
            )
        )
    rows.append(
        dict(
            arm="shared",
            stage="cache_preparation",
            seconds=execution["preparation"]["seconds"],
            note="One preparation GPU; charged once, not once per fast/hybrid arm",
        )
    )
    rows.append(
        dict(
            arm="allocation",
            stage="runner_wall",
            seconds=execution["wall_seconds"],
            note="Four GPUs allocated throughout; Slurm accounting remains authoritative",
        )
    )
    common.table(output / "timing.csv", rows)
    common.table(output / "update-timing.csv", update_rows)
    (output / "execution-provenance.json").write_text(
        json.dumps(execution, indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, task in zip(axes, ("om4", "observation"), strict=True):
        for arm, color, marker in zip(
            ARMS, direct.COLORS[1:], direct.MARKERS[1:], strict=True
        ):
            values = [r for r in update_rows if r["arm"] == arm and r["task"] == task]
            ax.plot(
                [r["update"] for r in values],
                [r["seconds"] for r in values],
                color=color,
                marker=marker,
                markevery=8,
                markersize=4,
                label=arm,
            )
        ax.set(
            title=task.upper(),
            xlabel="Added mixed update",
            ylabel="Seconds/update",
            yscale="log",
        )
    axes[0].legend()
    fig.suptitle(
        "Measured training cost; full sampled CRPS versus one-call surrogate\n"
        "Cache preparation and allocation costs are separate in timing.csv"
    )
    save_png(fig, output / "update-timing.png", dpi=130)
    plt.close(fig)


def state_maps(roots, output, origin, native):
    data: list[dict[str, Any]] = []
    domain = "native" if native else "annual"
    for name in NAMES:
        root = roots[name] / domain
        filename = f"fields-{origin[:4]}.npz" if native else f"members-{origin}.npz"
        with np.load(root / filename) as saved:
            indices = [saved["leads_days"].tolist().index(d) for d in (30, 365)]
            members = saved["members" if native else "fields"][:, indices].copy()
            record = {k: saved[k] for k in ("mask", "lat", "lon")}
            if native:
                record["reference"] = saved["reference"][indices]
        if not native:
            with np.load(root / f"{origin}.npz") as saved:
                record["reference"] = saved["reference"][[5, 72], :2]
        if not np.isfinite(members).all() or members.shape[:3] != (8, 2, 4):
            raise ValueError("Wrong member shape or nonfinite physical output")
        record.update(mean=members.mean(0), draw=members[0])
        if data:
            for key in ("mask", "lat", "lon", "reference"):
                np.testing.assert_array_equal(record[key], data[0][key])
        data.append(record)
    clipping = []
    for lead_index, lead in enumerate((30, 365)):
        for channel, (label, unit, limits) in enumerate(FIELDS):
            has_reference = native or channel < 2
            mask = data[0]["mask"][channel].astype(bool)
            height, width = mask.shape
            tw, th = 2 * width + 90, 5 * (height + 28) + 190
            fig = plt.figure(figsize=(tw / 100, th / 100), dpi=100)
            target = (
                data[0]["reference"][lead_index, channel]
                if has_reference
                else np.full(mask.shape, np.nan)
            )
            panels = [
                ("OM4 reference" if native else "Observed reference", target, target)
            ] + [
                (
                    name,
                    item["mean"][lead_index, channel],
                    item["draw"][lead_index, channel],
                )
                for name, item in zip(NAMES, data, strict=True)
            ]
            for row, (name, mean, draw) in enumerate(panels):
                for column, field in enumerate((mean, draw)):
                    valid = mask & np.isfinite(field)
                    ax = fig.add_axes(
                        (
                            (30 + column * (width + 25)) / tw,
                            (th - 105 - row * (height + 28) - height) / th,
                            width / tw,
                            height / th,
                        )
                    )
                    im = ax.imshow(
                        np.where(valid, field, np.nan),
                        origin="lower",
                        interpolation="nearest",
                        cmap="viridis",
                        vmin=limits[0],
                        vmax=limits[1],
                    )
                    mode = "mean of 8" if column == 0 else "draw 1"
                    suffix = ""
                    if valid.any():
                        clipped = float(
                            (
                                (field[valid] < limits[0]) | (field[valid] > limits[1])
                            ).mean()
                        )
                        clipping.append(
                            dict(
                                initialization=domain,
                                arm=name,
                                day=lead,
                                field=label,
                                mode=mode,
                                minimum=float(field[valid].min()),
                                maximum=float(field[valid].max()),
                                clipped_fraction=clipped,
                                vmin=limits[0],
                                vmax=limits[1],
                            )
                        )
                        suffix = f" · {mode} · {clipped:.0%} clipped" if row else ""
                    elif row == 0 and not has_reference:
                        name = "No instantaneous interior observation"
                        ax.text(
                            0.5,
                            0.5,
                            "Interior observations supervise monthly averages",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
                    ax.set_title(name + suffix, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
            bar = fig.add_axes((0.15, 48 / th, 0.7, 10 / th))
            fig.colorbar(
                im,
                cax=bar,
                orientation="horizontal",
                extend="both",
                label=f"{label.upper()} ({unit}); fixed range {limits}",
            )
            initialization = (
                "OM4 surface history + flux forcing"
                if native
                else "Observation surface history + ERA5 forcing"
            )
            fig.suptitle(
                f"{initialization} · {origin[:4]} · day {lead} · {label.upper()}\n"
                "Neither initializer observes gold interiors; independent conditional draws across time\n"
                "Native 1:1 grid pixels; identical fixed scales; out-of-range values clip",
                fontsize=9,
                y=1 - 15 / th,
            )
            save_png(
                fig,
                output / f"{'native' if native else 'obs'}-day{lead}-{label}.png",
                dpi=100,
            )
            plt.close(fig)
    common.table(output / f"{domain}-map-clipping.csv", clipping)


def annual_chart(output):
    """Use a zero-based linear axis for this modest-range comparison."""
    rows = common.read(output / "results.json")["annual"]
    names = ("parent", "deterministic", *ARMS)
    colors = ("#777777", "#333333", *direct.COLORS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, day in zip(axes, (30, 365), strict=True):
        values = [
            next(
                r["score"]
                for r in rows
                if r["arm"] == arm and r["origin"] == "pooled" and r["day"] == day
            )
            for arm in names
        ]
        bars = ax.barh(names, values, color=colors)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set(
            title=f"Day {day}",
            xlabel="Climatology-normalized blended RMSE (lower is better)",
            xlim=(0, max(values) * 1.18),
        )
    fig.suptitle(
        "Diffusion: eight-member means · three January origins · fixed final checkpoints"
    )
    save_png(fig, output / "annual-rmse.png", dpi=130)
    fig.savefig(output / "annual-rmse.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Campaign evaluation directory"
    )
    parser.add_argument(
        "--parent", type=Path, required=True, help="Replay-bound evaluation directory"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--presentation", type=Path, required=True)
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Fetched production runs with events and COMPLETE receipts",
    )
    parser.add_argument(
        "--execution-results",
        type=Path,
        required=True,
        help="Production logs/JOB/RESULTS.json, not qualification",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", choices=ORIGINS, default=ORIGINS[0])
    args = parser.parse_args()
    roots, evidence = verify(args)
    # Check timing receipt before creating any scientific output as well.
    execution = common.read(args.execution_results)
    if not all(r["success"] for r in execution["arms"]) or {
        r["arm"] for r in execution["arms"]
    } != set(ARMS):
        raise ValueError("Incomplete final production execution receipt")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="diffusion-obs-report-") as staging_name:
        staging = Path(staging_name)
        for name, path in roots.items():
            (staging / name).symlink_to(path.resolve(), target_is_directory=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("report_diffusion_interventions.py")),
                "--root",
                str(staging),
                "--baseline",
                str(args.baseline),
                "--presentation",
                str(args.presentation),
                "--output",
                str(args.output),
                "--arms",
                *ARMS,
                "--updates",
                "128",
                "--parent-updates-per-task",
                "8128",
                "--parent-annual",
                str(args.parent / "annual"),
            ],
            check=True,
        )
        # Include the immediate parent in member/mean spectra on the same validation support.
        common.ARMS[:] = NAMES
        common.scored_spectra(staging, args.output)
    direct.NAMES = NAMES
    annual_chart(args.output)
    direct.latent_curves(roots, args.output)
    calibration(roots, args.output)
    timing(args, args.output)
    for native in (False, True):
        state_maps(roots, args.output, args.origin, native)
    (args.output / "pilot-provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
