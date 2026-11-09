# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched final-update model-size comparison from validation receipts."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZES = (4, 8, 16, 32)
COLORS = ("#009e73", "#cc79a7", "#d55e00", "#0072b2")
FIELDS = ("thetao", "so", "uo", "vo", "zos")
NAMES = ("Temperature", "Salinity", "Zonal velocity", "Meridional velocity", "SSH")


def load(root):
    runs = {}
    reference = None
    for depth in SIZES:
        folder = root / f"l{depth:02d}-25k"
        provenance = json.loads((folder / "provenance-restart-0.json").read_text())
        config = json.loads(json.dumps(provenance["config"]))
        assert config["model"].pop("layers") == depth
        compare = dict(
            config=config,
            source=provenance["source"],
            commit=provenance["source_commit"],
        )
        if reference is None:
            reference = compare
        else:
            assert compare == reference, "Unmatched recipe or source"
        complete = json.loads((folder / "COMPLETE.json").read_text())
        assert complete["step"] == 25000 and complete["ema_updates"] == 25000
        assert complete["effective_batch"] == 256 and complete["replicas_agree"]
        assert json.loads((folder / "replica-check.json").read_text())["identical"]
        runs[depth] = dict(provenance=provenance, validation={})
        for kind in ("raw", "ema"):
            results = [
                json.loads(p.read_text())
                for p in sorted(folder.glob(f"validation-{kind}-*.json"))
            ]
            assert [r["step"] for r in results] == [0, 1] + list(
                range(1000, 25001, 1000)
            )
            runs[depth]["validation"][kind] = results
    baseline = runs[16]["validation"]["raw"][-1]
    for run in runs.values():
        for results in run["validation"].values():
            for result in results:
                assert result["origins"] == baseline["origins"]
                for row, expected in zip(
                    result["per_channel"], baseline["per_channel"], strict=True
                ):
                    assert (row["channel"], row["count"]) == (
                        expected["channel"],
                        expected["count"],
                    )
                    np.testing.assert_allclose(
                        row["persistence_full_grid_nmse"],
                        expected["persistence_full_grid_nmse"],
                        rtol=1e-12,
                        atol=1e-15,
                    )
    t_only = {
        kind: json.loads(
            (root / "temperature-25k" / f"validation-{kind}-0025000.json").read_text()
        )
        for kind in ("raw", "ema")
    }
    t_provenance = json.loads(
        (root / "temperature-25k" / "provenance-restart-0.json").read_text()
    )
    t_config = json.loads(json.dumps(t_provenance["config"]))
    assert t_config.pop("arm") == "temperature"
    assert t_config["model"].pop("layers") == 16
    t_config["arm"] = "all_fields"
    assert (
        dict(
            config=t_config,
            source=t_provenance["source"],
            commit=t_provenance["source_commit"],
        )
        == reference
    )
    assert (
        t_provenance["initial_parameter_sha256"]
        == runs[16]["provenance"]["initial_parameter_sha256"]
    )
    for result in t_only.values():
        assert result["step"] == 25000 and result["origins"] == baseline["origins"]
        for row, expected in zip(
            result["per_channel"], baseline["per_channel"][38:57], strict=True
        ):
            assert (row["channel"], row["count"]) == (
                expected["channel"],
                expected["count"],
            )
            np.testing.assert_allclose(
                row["persistence_full_grid_nmse"],
                expected["persistence_full_grid_nmse"],
                rtol=1e-12,
                atol=1e-15,
            )
    return runs, t_only


def style(ax):
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.spines[["right", "top"]].set_visible(False)


def render(runs, t_only, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, minimum in zip(axes, (0, 1000), strict=True):
        for depth, color in zip(SIZES, COLORS, strict=True):
            size = runs[depth]["provenance"]["parameters"] / 1e6
            for kind, line in (("raw", "--"), ("ema", "-")):
                results = [
                    r for r in runs[depth]["validation"][kind] if r["step"] >= minimum
                ]
                ax.plot(
                    [r["step"] for r in results],
                    [r["temperature_full_grid_nmse"] for r in results],
                    color=color,
                    linestyle=line,
                    alpha=0.45 if kind == "raw" else 1,
                    label=f"{size:.0f}M ({depth} blocks), {kind}",
                )
        if minimum == 0:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([0, 1, 1000, 25000], ["0", "1", "1k", "25k"])
        else:
            ax.set_xlim(1000, 25000)
            ax.set_xticks([1000, 10000, 20000, 25000], ["1k", "10k", "20k", "25k"])
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Temperature mean-depth normalized MSE")
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, fontsize=8, frameon=False
    )
    fig.suptitle(
        "All-fields model scaling · fixed validation cases · legacy full-grid MSE\nSolid: EMA; dashed: raw · lower is better"
    )
    fig.savefig(output / "validation-temperature.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    sizes = [runs[d]["provenance"]["parameters"] / 1e6 for d in SIZES]
    for kind, line, marker, color in (
        ("raw", "--", "o", "#777777"),
        ("ema", "-", "o", "#0072b2"),
    ):
        values = [
            runs[d]["validation"][kind][-1]["temperature_full_grid_nmse"] for d in SIZES
        ]
        ax.plot(
            sizes,
            values,
            linestyle=line,
            marker=marker,
            color=color,
            label=f"All-fields: {kind}",
        )
    ax.axhline(
        t_only["ema"]["temperature_full_grid_nmse"],
        color="#009e73",
        linestyle=":",
        label="208M T-only EMA reference",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(
        sizes, [f"{v:.0f}M\n{d} blocks" for v, d in zip(sizes, SIZES, strict=True)]
    )
    ax.set_xlabel("Parameters (width 1,024 held fixed)")
    ax.set_ylabel("Temperature mean-depth normalized MSE")
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "Final update 25,000 · same crop exposure · one seed\nLegacy full-grid MSE; lower is better"
    )
    fig.savefig(output / "temperature-vs-parameters.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(16, 10), layout="constrained")
    for z, ax in enumerate(axes.flat[:19]):
        channel = f"thetao_{z}"
        for depth, color in zip(SIZES, COLORS, strict=True):
            for kind, line in (("raw", "--"), ("ema", "-")):
                results = [
                    r for r in runs[depth]["validation"][kind] if r["step"] >= 1000
                ]
                rows = [
                    next(c for c in r["per_channel"] if c["channel"] == channel)
                    for r in results
                ]
                ax.plot(
                    [r["step"] for r in results],
                    [r["wet_cell_rmse"] for r in rows],
                    color=color,
                    linestyle=line,
                    alpha=0.4 if kind == "raw" else 1,
                    linewidth=1.2,
                    label=f"{depth} blocks: {kind}",
                )
        row = rows[-1]
        scale = row["wet_cell_rmse"] ** 2 / row["full_grid_nmse"]
        ax.axhline(
            np.sqrt(row["persistence_full_grid_nmse"] * scale),
            color=".45",
            linestyle=":",
            linewidth=0.8,
            label="Persistence",
        )
        ax.set_title(f"{row['depth_m']:g} m", fontsize=10)
        ax.set_ylabel("Wet-cell RMSE (°C)", fontsize=8)
        ax.set_xlim(1000, 25000)
        ax.set_xticks([1000, 10000, 25000], ["1k", "10k", "25k"])
        ax.tick_params(labelsize=8)
        style(ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="center",
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Temperature validation by depth · same fixed cases\nSolid: EMA; dashed: raw; dotted: persistence · lower is better"
    )
    fig.supxlabel("Optimizer updates")
    fig.savefig(output / "validation-temperature-depths.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, field, title in zip(axes.flat, FIELDS, NAMES, strict=False):
        for depth, color in zip(SIZES, COLORS, strict=True):
            for kind, line in (("raw", "--"), ("ema", "-")):
                results = [
                    r for r in runs[depth]["validation"][kind] if r["step"] >= 1000
                ]
                ax.plot(
                    [r["step"] for r in results],
                    [r["per_field_nmse"][field] for r in results],
                    color=color,
                    linestyle=line,
                    alpha=0.4 if kind == "raw" else 1,
                    label=f"{depth} blocks: {kind}",
                )
        ax.set_title(title)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Mean-channel normalized MSE")
        style(ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="center",
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "All-fields scaling: supervised field tradeoffs · legacy full-grid MSE\nSolid: EMA; dashed: raw · lower is better"
    )
    fig.savefig(output / "validation-fields.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, t_only = load(args.source)
    rows = []
    final = []
    for depth, run in runs.items():
        for kind, results in run["validation"].items():
            for result in results:
                for channel in result["per_channel"]:
                    rows.append(
                        dict(
                            blocks=depth,
                            parameters=run["provenance"]["parameters"],
                            weights=kind,
                            step=result["step"],
                            **channel,
                        )
                    )
            last = results[-1]
            final.append(
                dict(
                    blocks=depth,
                    parameters=run["provenance"]["parameters"],
                    weights=kind,
                    temperature_nmse=last["temperature_full_grid_nmse"],
                    full_grid_nmse=last["full_grid_nmse"],
                    **last["per_field_nmse"],
                )
            )
    with gzip.open(args.output / "validation-per-channel.csv.gz", "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "final-metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)
    summary = dict(
        final=final,
        t_only_reference=t_only,
        provenance={str(d): r["provenance"] for d, r in runs.items()},
        validation_populations_match=True,
    )
    with gzip.open(args.output / "summary.json.gz", "wt") as f:
        json.dump(summary, f, indent=2)
    render(runs, t_only, args.output)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
