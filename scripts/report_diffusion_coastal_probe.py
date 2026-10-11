# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify and report the paired 2015 OM4 coastal-decoder counterfactual probe."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("control", "replay-bound")
FIELDS = ("sst", "ssh")
UNITS = ("°C", "m")
KINDS = {
    "sample": ("Original", "#333333", "o"),
    "sample_readout_rms_bound": ("Bound before decoding", "#e69f00", "s"),
    "sample_recurrent_rms_bound": ("Bound after each step", "#0072b2", "^"),
    "sample_landfill": ("Fill land at decoder stem", "#009e73", "D"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def csv_file(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(root):
    complete = json.loads((root / "COMPLETE.json").read_text())
    for filename, expected in complete["files"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"Hash mismatch: {filename}")
    records = json.loads((root / "metrics.json").read_text())
    expected_counts = {
        "denoise": 48,
        "sample": 16,
        "sample_readout_rms_bound": 8,
        "sample_recurrent_rms_bound": 8,
        "sample_landfill": 4,
    }
    if len(records) != 84 or Counter(row["kind"] for row in records) != expected_counts:
        raise ValueError("Expected all 84 probe records")
    if len({row["key"] for row in records}) != 84:
        raise ValueError("Duplicate probe records")
    if complete["leads_days"] != [30, 365] or complete["sampling_steps"] != 32:
        raise ValueError("Unexpected evaluation protocol")
    if len(complete["seeds"]) != 2 or len(set(complete["seeds"])) != 2:
        raise ValueError("Expected two paired seeds")
    if (
        complete["checkpoints"]["control"]["contract"]
        != complete["checkpoints"]["replay-bound"]["contract"]
    ):
        raise ValueError("Checkpoint data contracts differ")
    for checkpoint in complete["checkpoints"].values():
        if (
            checkpoint["counts"] != {"om4": 8128, "observation": 8128}
            or checkpoint["step"] != 16256
        ):
            raise ValueError("Unexpected checkpoint training budget")
    maps = np.load(root / "maps.npz", allow_pickle=False)
    mask = maps["mask"].astype(bool)
    weights = mask * np.maximum(np.cos(np.deg2rad(maps["lat"])), 0)[None, :, None]
    weights = weights / weights.sum((-2, -1), keepdims=True)
    for row in records:
        field = maps[row["key"]]
        reference = maps[f"reference_day{row['lead_days']}"]
        if field.shape != (2, 180, 360) or not np.isfinite(field).all():
            raise ValueError(f"Bad field: {row['key']}")
        rmse = np.sqrt(((field - reference) ** 2 * weights).sum((-2, -1)))
        np.testing.assert_allclose(
            rmse, [row["metrics"]["all"][field]["rmse"] for field in FIELDS], rtol=1e-6
        )
    return complete, records, maps


def compact_rows(records):
    result = []
    for record in records:
        for source in ("metrics", "input_metrics"):
            if source not in record:
                continue
            for region, metrics in record[source].items():
                for field, values in metrics.items():
                    if values is None:
                        continue
                    result.append(
                        dict(
                            key=record["key"],
                            kind=record["kind"],
                            decoder=record["decoder"],
                            conditioning=record["conditioning"],
                            lead_days=record["lead_days"],
                            seed=record.get("seed", ""),
                            sigma=record.get("sigma", ""),
                            source=source,
                            region=region,
                            field=field,
                            **values,
                        )
                    )
    return result


def pooled_rows(records):
    result = []
    for arm in ARMS:
        for kind in KINDS:
            for lead in (30, 365):
                selected = [
                    r
                    for r in records
                    if r["decoder"] == arm
                    and r["conditioning"] == arm
                    and r["kind"] == kind
                    and r["lead_days"] == lead
                ]
                if not selected:
                    if kind == "sample_landfill" and lead == 30:
                        continue
                    raise ValueError("Missing sampled comparison")
                if len(selected) != 2:
                    raise ValueError("Expected exactly two individual members")
                for region in ("all", "0-2", "2-5", "5-10", "over-10"):
                    for field in FIELDS:
                        scores = [r["metrics"][region][field]["rmse"] for r in selected]
                        result.append(
                            dict(
                                decoder=arm,
                                kind=kind,
                                lead_days=lead,
                                region=region,
                                field=field,
                                member_count=2,
                                pooled_member_rmse=float(
                                    np.sqrt(np.mean(np.square(scores)))
                                ),
                                min_member_rmse=min(scores),
                                max_member_rmse=max(scores),
                            )
                        )
    return result


def sample_figure(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    positions = [(arm, lead) for arm in ARMS for lead in (30, 365)]
    for ci, (field, unit) in enumerate(zip(FIELDS, UNITS, strict=True)):
        for ri, region in enumerate(("all", "0-2")):
            ax = axes[ri, ci]
            for offset, (kind, (label, color, marker)) in enumerate(KINDS.items()):
                selected = [
                    r
                    for r in rows
                    if r["field"] == field
                    and r["region"] == region
                    and r["kind"] == kind
                ]
                x = [
                    positions.index((r["decoder"], r["lead_days"]))
                    + (offset - 1.5) * 0.12
                    for r in selected
                ]
                y = [r["pooled_member_rmse"] for r in selected]
                ax.scatter(
                    x, y, marker=marker, color=color, s=48, label=label, zorder=3
                )
                ax.vlines(
                    x,
                    [r["min_member_rmse"] for r in selected],
                    [r["max_member_rmse"] for r in selected],
                    colors=color,
                )
            ax.set_xticks(
                range(4),
                [
                    "Control\n30 d",
                    "Control\n365 d",
                    "Replay + bound\n30 d",
                    "Replay + bound\n365 d",
                ],
            )
            ax.set_yscale("log")
            ax.set_ylabel(f"{field.upper()} RMSE ({unit}, log scale)")
            ax.set_title(
                "All wet cells" if region == "all" else "Within 2 grid cells of land"
            )
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle(
        "Paired inference counterfactuals • 2015 OM4 initialization\nPoints pool two individual-member MSEs before square root; bars span members"
    )
    save(fig, output, "sampled-rmse")


def denoise_figure(records, output):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    styles = {
        "correct": ("Matching-lead latent", "#0072b2", "o"),
        "zero": ("Zero latent", "#d55e00", "s"),
        "origin": ("Origin latent", "#009e73", "^"),
    }
    for ri, arm in enumerate(ARMS):
        for li, lead in enumerate((30, 365)):
            for ci, (field, unit) in enumerate(zip(FIELDS, UNITS, strict=True)):
                ax = axes[ri, li * 2 + ci]
                for condition, (label, color, marker) in styles.items():
                    selected = sorted(
                        [
                            r
                            for r in records
                            if r["kind"] == "denoise"
                            and r["decoder"] == arm
                            and r["lead_days"] == lead
                            and r["conditioning"] == condition
                        ],
                        key=lambda r: r["sigma"],
                    )
                    if len(selected) != 4:
                        raise ValueError("Incomplete denoising curve")
                    x = [r["sigma"] for r in selected]
                    y = [r["metrics"]["all"][field]["rmse"] for r in selected]
                    ax.loglog(x, y, marker=marker, color=color, label=label)
                    if condition == "correct":
                        ax.loglog(
                            x,
                            [
                                r["input_metrics"]["all"][field]["rmse"]
                                for r in selected
                            ],
                            color="black",
                            marker="x",
                            linestyle="--",
                            label="Noisy target input",
                        )
                ax.set_title(f"{arm} • day {lead} • {field.upper()}")
                ax.set_ylabel(f"RMSE ({unit})")
                ax.set_xlabel("Noise σ (normalized units)")
                ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle(
        "Single-call denoising of the true OM4 target plus identical noise\nGlobal area-weighted RMSE • one noise draw per lead; these are not forecast scores"
    )
    save(fig, output, "known-target-denoising")


def maps_figure(maps, seed, output):
    panels = [
        ("Reference", "reference_day365"),
        ("Control", f"sample_decoder-control_latent-control_day365_seed{seed}"),
        (
            "Control, bound after every step",
            f"sample_recurrent_rms_bound_control_day365_seed{seed}",
        ),
        (
            "Replay + bound",
            f"sample_decoder-replay-bound_latent-replay-bound_day365_seed{seed}",
        ),
        (
            "Replay + bound, bound after every step",
            f"sample_recurrent_rms_bound_replay-bound_day365_seed{seed}",
        ),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), constrained_layout=True)
    limits = [(-2, 32), (-2, 2)]
    mask = maps["mask"].astype(bool)
    for ri, (label, key) in enumerate(panels):
        for ci, (field, unit) in enumerate(zip(FIELDS, UNITS, strict=True)):
            ax = axes[ri, ci]
            values = maps[key][ci]
            low, high = limits[ci]
            cmap = plt.get_cmap("coolwarm" if ci else "turbo").copy()
            cmap.set_bad("#d0d0d0")
            image = ax.imshow(
                np.ma.masked_where(~mask[ci], values),
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=low,
                vmax=high,
                extent=[0, 360, float(maps["lat"].min()), float(maps["lat"].max())],
                aspect="auto",
            )
            outside = float(
                (((values < low) | (values > high)) & mask[ci]).sum()
                / mask[ci].sum()
                * 100
            )
            ax.set_title(
                f"{label}\n{field.upper()}: {outside:.1f}% wet cells outside color limits",
                fontsize=10,
            )
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks([-60, 0, 60])
            if ri == 4:
                fig.colorbar(
                    image,
                    ax=axes[:, ci],
                    shrink=0.5,
                    label=f"{field.upper()} ({unit})",
                    extend="both",
                    location="bottom",
                )
    fig.suptitle(
        f"Day 365 • single member, same seed {seed}\nFixed scales deliberately expose out-of-range states; gray = dry cells"
    )
    save(fig, output, "day365-members")


def latent_figure(complete, output):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ai, arm in enumerate(ARMS):
        baseline = complete["latent_rms"][arm]
        recurrent = complete["latent_bound"][arm]["recurrent"]
        if len(baseline) != 74 or len(recurrent) != 73:
            raise ValueError("Expected 73 five-day processor updates")
        for step, value in enumerate(baseline):
            diagnostics = (
                recurrent[step - 1]
                if step
                else dict(pre_rms=value, post_rms=value, fraction_clamped=0.0)
            )
            if step and diagnostics["lead_days"] != step * 5:
                raise ValueError("Nonmatching latent lead")
            rows.append(
                dict(
                    decoder=arm,
                    lead_days=step * 5,
                    original_rms=value,
                    recurrent_proposal_rms=diagnostics["pre_rms"],
                    recurrent_bounded_rms=diagnostics["post_rms"],
                    fraction_slot_channels_clamped=diagnostics["fraction_clamped"],
                )
            )
        selected = [r for r in rows if r["decoder"] == arm]
        ax = axes[ai]
        for key, label, color, marker in (
            ("original_rms", "Original rollout", "#333333", "o"),
            ("recurrent_bounded_rms", "Bound after every step", "#0072b2", "^"),
        ):
            ax.plot(
                [r["lead_days"] for r in selected],
                [r[key] for r in selected],
                color=color,
                marker=marker,
                markevery=12,
                markersize=5,
                label=label,
            )
        ax.set_title(f"{arm} • independent y limits")
        ax.set_xlabel("Lead (days)")
        ax.set_ylabel("Whole-state latent RMS (log scale)")
        ax.set_yscale("log")
        ax.set_xticks([0, 90, 180, 270, 365])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Persistent latent growth • same 2015 OM4 initialization\nInference bound caps each slot/channel RMS; plotted RMS aggregates the whole state"
    )
    save(fig, output, "latent-rms")
    csv_file(output / "latent-rms.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/coastal-decoder-probe")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/diffusion-interior-beta/coastal-decoder-assets"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    complete, records, maps = verify(args.root)
    csv_file(args.output / "metrics.csv", compact_rows(records))
    pooled = pooled_rows(records)
    csv_file(args.output / "sampled-rmse.csv", pooled)
    sample_figure(pooled, args.output)
    denoise_figure(records, args.output)
    maps_figure(maps, complete["seeds"][0], args.output)
    latent_figure(complete, args.output)
    provenance = {
        k: complete[k]
        for k in (
            "origin",
            "producer",
            "script_sha256",
            "leads_days",
            "seeds",
            "sampling_steps",
            "noise_correlation",
            "denoising_sigmas",
            "denoising_noise_seeds",
            "coast_distance",
            "target_pair",
            "land_fill",
            "latent_bound_definition",
            "caveats",
            "elapsed_seconds",
            "files",
        )
    }
    manifest = json.loads((args.root / "manifest.json").read_text())
    provenance.update(
        execution={
            key: manifest[key]
            for key in (
                "job_id",
                "host",
                "state",
                "GPUs",
                "allocated_elapsed_seconds",
                "allocated_gpu_hours",
                "code_id",
                "archive_sha256",
                "base_commit",
            )
        },
        report_script_sha256=digest(Path(__file__)),
        completion_sha256=digest(args.root / "COMPLETE.json"),
        records_verified=84,
        checkpoints={
            arm: {k: c[k] for k in ("sha256", "counts", "step")}
            for arm, c in complete["checkpoints"].items()
        },
        pooling="Square root of mean individual-member area-weighted MSE; no ensemble-mean score in these charts",
        map_seed=complete["seeds"][0],
        coordinate_note="Native latitude rows displayed with nearest interpolation; latitude ticks are illustrative, not a cartographic reprojection",
    )
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for path in args.output.iterdir():
        if path.suffix != ".license":
            path.with_name(path.name + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    print(json.dumps(dict(verified_records=len(records), output=str(args.output))))


if __name__ == "__main__":
    main()
