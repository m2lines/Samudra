"""CPU-only maps and selected-versus-final EMA comparison from saved evaluations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = (
    "full",
    "detach",
    "latent_full",
    "latent_only",
    "fine_latent",
    "fine_latent_detach",
)
REGIONS = {
    "global": (-180, 180, -90, 90),
    "south-pacific-gyre": (-140, -110, -40, -20),
    "north-pacific-gyre": (-160, -130, 20, 40),
}


def hp(x, wet):
    interior = wet.copy()
    neighbors = []
    for shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= np.roll(wet, shift, (-2, -1))
        neighbors.append(np.roll(x, shift, (-2, -1)))
    interior[[0, -1], :] = False
    return np.where(interior, x - sum(neighbors) / 4, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arms = [
        a
        for a in ARMS
        if all(
            (args.root / a / s / "EVAL_COMPLETE").exists()
            for s in ("selected", "epoch70")
        )
    ]
    if not arms:
        raise RuntimeError("No complete selected/epoch70 evaluation pairs")
    records = {
        (a, s): json.loads((args.root / a / s / "heldout.json").read_text())
        for a in arms
        for s in ("selected", "epoch70")
    }
    meta = json.loads(
        (args.root / arms[0] / "selected" / "map-metadata.json").read_text()
    )
    lat = np.array(meta["latitude"])
    lon = (np.array(meta["longitude"]) + 180) % 360 - 180
    order = np.argsort(lon)
    lon = lon[order]
    maps = {(a, s): np.load(args.root / a / s / "maps-call-01.npz") for a, s in records}
    reference = maps[arms[0], "selected"]
    for m in maps.values():
        for name in ("truth", "mask", "times", "initial_times"):
            np.testing.assert_array_equal(m[name], reference[name])
    rows = []
    for (a, s), result in records.items():
        for call in (1, 3, 9, 36, 72):
            subset = [
                r
                for r in result["records"]
                if r["mode"] == "persistent" and r["call"] == call
            ]
            assert len(subset) == 6
            rows.append(
                dict(
                    arm=a,
                    selection=s,
                    epoch=result["selected_epoch"],
                    last_lead_day=call * 10,
                    mse=np.mean([r["mse"] for r in subset]),
                )
            )
    with (args.output / "checkpoint-summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Spatial maps and epoch-70 EMA comparison",
        "",
        "Included completed arms: " + ", ".join(arms) + ".",
        "",
        "Six held-out origins; block MSE averages both output timestamps and all 77 channels. Epoch-70 values use the `ema` state in `latest.pt`, not its raw `model` weights. Selected values use `best-ema.pt`.",
        "",
        "| Arm | Selected epoch | Selected days 5/10 | Epoch70 days 5/10 | Selected days715/720 | Epoch70 days715/720 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def value(a, s, c):
        return next(
            r["mse"]
            for r in rows
            if r["arm"] == a and r["selection"] == s and r["last_lead_day"] == c * 10
        )

    for a in arms:
        lines.append(
            f"| {a} | {records[a, 'selected']['selected_epoch']} | {value(a, 'selected', 1):.6g} | {value(a, 'epoch70', 1):.6g} | {value(a, 'selected', 72):.6g} | {value(a, 'epoch70', 72):.6g} |"
        )
    lines += [
        "",
        "## Maps",
        "",
        f"Maps show the first output timestamp, {reference['times'][0]}, for held-out origin 0 (inputs {reference['initial_times'][0]} and {reference['initial_times'][1]}). These single-origin maps illustrate spatial structure; the score table above uses all six origins.",
        "",
        "The cutouts are fixed interior-ocean boxes chosen geographically before inspecting treatment outputs: South Pacific 140–110°W, 40–20°S; North Pacific 160–130°W, 20–40°N. They are intended to sample quiet gyre interiors, not asserted to be the globally quietest regions. CSV diagnostics report actual truth roughness and errors within each box.",
        "",
        "Each image compares selected and epoch-70 EMA predictions, signed errors, and high-pass errors. Physical-value scales are shared across truth and every arm/checkpoint; signed-error and high-pass-error scales are separately shared. Limits use pooled 1st/99th percentiles (values outside them saturate). Gray denotes land, below-bottom cells, or excluded high-pass stencils. High-pass = center minus four-neighbor mean; longitude is periodic and stencils touching land/latitude edges are excluded. No smoothing or spatial regridding is applied.",
        "",
    ]
    regional = []
    for j, field in enumerate(meta["channels"]):
        wet = reference["mask"][j].astype(bool)
        truth = reference["truth"][0, j] * meta["std"][j] + meta["mean"][j]
        physical = {
            key: m["prediction"][0, j] * meta["std"][j] + meta["mean"][j]
            for key, m in maps.items()
        }
        errors = {key: z - truth for key, z in physical.items()}
        highs = {key: hp(z, wet) for key, z in errors.items()}
        unit = (
            "°C"
            if field.startswith("thetao")
            else "psu"
            if field.startswith("so")
            else "m"
            if field == "zos"
            else "m/s"
        )
        for region, (xmin, xmax, ymin, ymax) in REGIONS.items():
            box = (lat[:, None] >= ymin) & (lat[:, None] <= ymax)
            raw_lon = (np.array(meta["longitude"]) + 180) % 360 - 180
            box = box & (raw_lon[None, :] >= xmin) & (raw_lon[None, :] <= xmax)
            valid = wet & box
            lo, hi = np.percentile(
                np.concatenate([truth[valid]] + [z[valid] for z in physical.values()]),
                [1, 99],
            )
            elim = max(
                float(
                    np.percentile(
                        np.concatenate([abs(z[valid]) for z in errors.values()]), 99
                    )
                ),
                1e-12,
            )
            hlim = max(
                float(
                    np.nanpercentile(
                        np.concatenate([abs(z[valid]) for z in highs.values()]), 99
                    )
                ),
                1e-12,
            )
            fig, axs = plt.subplots(
                6,
                len(arms) + 1,
                figsize=(3.3 * (len(arms) + 1), 13),
                layout="constrained",
                squeeze=False,
            )
            for row in range(6):
                selection = "selected" if row < 3 else "epoch70"
                kind = row % 3
                for col, arm in enumerate((None, *arms)):
                    ax = axs[row, col]
                    if arm is None and kind:
                        ax.axis("off")
                        continue
                    z = (
                        truth
                        if arm is None
                        else (
                            physical if kind == 0 else errors if kind == 1 else highs
                        )[arm, selection]
                    )
                    z = np.where(wet, z, np.nan)[:, order]
                    im = ax.pcolormesh(
                        lon,
                        lat,
                        z,
                        shading="nearest",
                        cmap="viridis" if kind == 0 else "RdBu_r",
                        vmin=lo if kind == 0 else -elim if kind == 1 else -hlim,
                        vmax=hi if kind == 0 else elim if kind == 1 else hlim,
                        rasterized=True,
                    )
                    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
                    ax.set_facecolor("#bbb")
                    ax.tick_params(labelsize=6)
                    if row == 0:
                        ax.set_title("Truth" if arm is None else arm, fontsize=9)
                    if col == 1:
                        ax.set_ylabel(
                            selection
                            + "\n"
                            + ["prediction", "error", "high-pass error"][kind],
                            fontsize=8,
                        )
                    if col == len(arms):
                        fig.colorbar(im, ax=ax, label=unit, shrink=0.8, extend="both")
            fig.suptitle(
                f"{field} · {region} · {reference['times'][0]} · first predicted timestamp",
                fontsize=12,
            )
            filename = f"{field}-{region}.png"
            fig.savefig(args.output / filename, dpi=130)
            plt.close(fig)
            lines.append(f"- [{field} — {region}]({filename})")
            truth_hp = hp(truth, wet)
            for key in errors:
                regional.append(
                    dict(
                        arm=key[0],
                        selection=key[1],
                        field=field,
                        region=region,
                        wet_cells=int(valid.sum()),
                        truth_std=float(np.std(truth[valid])),
                        truth_highpass_rms=float(
                            np.sqrt(np.nanmean(truth_hp[valid] ** 2))
                        ),
                        rmse=float(np.sqrt(np.mean(errors[key][valid] ** 2))),
                        highpass_error_rms=float(
                            np.sqrt(np.nanmean(highs[key][valid] ** 2))
                        ),
                        unit=unit,
                    )
                )
    with (args.output / "regional-metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=regional[0], lineterminator="\n")
        w.writeheader()
        w.writerows(regional)
    lines += [
        "",
        "![Deep-temperature South Pacific comparison](thetao_13-south-pacific-gyre.png)",
        "",
        "These are paired-seed exploratory results. Final-epoch weights are a separately reported comparison, not a replacement for the predeclared validation-selected endpoint.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Wrote", args.output / "REPORT.md")


if __name__ == "__main__":
    main()
