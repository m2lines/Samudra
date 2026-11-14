# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Join full-state predictions to cached original-regression surface velocities."""

import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "map_checks", HERE.parent / "soft-priors-25k/render_maps.py"
)
assert SPEC is not None and SPEC.loader is not None
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)
CACHE = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/skyline-soft-priors-25k-artifacts/maps"
)
LABELS = {
    "gold": "OM4",
    "persistence": "Persistence",
    "classifier": "Large classifier",
    "regression": "Original large regression",
    "samudra": "Samudra",
}


def load_patches(arrays, cache):
    patches = []
    with np.load(arrays) as data:
        # Load only the surface channels needed from each full-state array.
        fields = {
            name: data[name][[0, 19]]
            for name in ("gold", "persistence", "classifier", "samudra")
        }
        wet = data["wet"]
        masks = wet[[0, 19]] if wet.shape[0] == 77 else wet[[0, 0]]
        for region in ("gulf-stream", "agulhas"):
            path = cache / f"original-all-fields-regression-{region}-z00.npz"
            old = checks.load_npz(path)
            checks.same(int(old["origin"]), 4273, "forecast input index")
            checks.same(str(old["target_date"]), "2016-07-21", "target date")
            checks.same(int(old["crop"]), 33, "original regression crop")
            rows, cols = old["query_rows"], old["query_cols"]
            checks.same(data["lat"][rows[:, 0]], old["lat"], "latitude")
            checks.same(data["lon"][cols[0]], old["lon"], "longitude")
            checks.same(data["area"][rows, cols], old["area"], "cell area")
            for i, field in enumerate(("uo", "vo")):
                checks.validate_map(old, field, path)
                checks.same(masks[i][rows, cols], old["wet"], field + " wet mask")
                values = {name: value[i][rows, cols] for name, value in fields.items()}
                for name in ("gold", "persistence"):
                    checks.same(
                        values[name][old["wet"]],
                        old[f"{field}_{name}"][old["wet"]],
                        field + " " + name,
                        atol=2e-5,
                    )
                values["regression"] = old[f"{field}_prediction"]
                if any(
                    not np.isfinite(value[old["wet"]]).all()
                    for value in values.values()
                ):
                    raise ValueError("Missing wet velocity predictions")
                patches.append(
                    dict(
                        region=region,
                        field=field,
                        wet=old["wet"],
                        area=old["area"],
                        data=values,
                    )
                )
    return patches


def scores(values, gold, current, wet, area):
    weight = area[wet].astype(float)
    weight /= weight.sum()
    # Leading dimension holds one scalar component or both velocity components.
    forecast = values[:, wet].astype(float)
    truth = gold[:, wet].astype(float)
    reference = current[:, wet].astype(float)
    increment = forecast - reference
    true_increment = truth - reference
    predicted_power = np.sum(weight * np.sum(increment**2, axis=0))
    true_power = np.sum(weight * np.sum(true_increment**2, axis=0))
    error_power = np.sum(weight * np.sum((forecast - truth) ** 2, axis=0))
    denominator = np.sqrt(predicted_power * true_power)
    return dict(
        area_rmse=float(np.sqrt(error_power)),
        persistence_area_rmse=float(np.sqrt(true_power)),
        increment_rms=float(np.sqrt(predicted_power)),
        true_increment_rms=float(np.sqrt(true_power)),
        increment_alignment=float(
            np.sum(weight * np.sum(increment * true_increment, axis=0)) / denominator
        )
        if denominator
        else None,
        rmse_improvement_fraction=float(1 - np.sqrt(error_power / true_power))
        if true_power
        else None,
    )


def metrics(patches):
    rows = []
    for patch in patches:
        for model, value in patch["data"].items():
            if model == "gold":
                continue
            rows.append(
                dict(
                    region=patch["region"],
                    field=patch["field"],
                    model=model,
                    units="m/s",
                    target_date="2016-07-21",
                    wet_count=int(patch["wet"].sum()),
                    **scores(
                        value[None],
                        patch["data"]["gold"][None],
                        patch["data"]["persistence"][None],
                        patch["wet"],
                        patch["area"],
                    ),
                )
            )
    for region in ("gulf-stream", "agulhas"):
        u, v = [p for p in patches if p["region"] == region]
        wet = u["wet"] & v["wet"]

        def stack(model):
            return np.stack([u["data"][model], v["data"][model]])

        for model in ("persistence", "classifier", "regression", "samudra"):
            rows.append(
                dict(
                    region=region,
                    field="uv_vector",
                    model=model,
                    units="m/s",
                    target_date="2016-07-21",
                    wet_count=int(wet.sum()),
                    **scores(
                        stack(model),
                        stack("gold"),
                        stack("persistence"),
                        wet,
                        u["area"],
                    ),
                )
            )
    return rows


def draw(patches, rows, output, errors):
    models = list(LABELS)[1:] if errors else list(LABELS)
    width, height = 220 + 230 * len(models), 4 * 245 + 125
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    for i, patch in enumerate(patches):
        values, wet = patch["data"], patch["wet"]
        plotted = [values[m] - values["gold"] if errors else values[m] for m in models]
        low = min(float(a[wet].min()) for a in plotted)
        high = max(float(a[wet].max()) for a in plotted)
        if errors:
            high = max(abs(low), abs(high), 1e-12)
            low = -high
        elif low == high:
            high = low + 1e-12
        top = height - 110 - i * 245
        fig.text(
            0.005,
            (top - 90) / height,
            f"{patch['region']}\n{patch['field']} · 2.5 m",
            fontsize=9,
        )
        for j, (model, value) in enumerate(zip(models, plotted, strict=True)):
            ax = fig.add_axes(
                (
                    (130 + j * 230) / width,
                    (top - 192) / height,
                    192 / width,
                    192 / height,
                )
            )
            image = ax.imshow(
                np.ma.masked_where(~wet, value),
                origin="lower",
                interpolation="none",
                resample=False,
                vmin=low,
                vmax=high,
                cmap="RdBu_r" if errors else "viridis",
            )
            ax.set_axis_off()
            if i == 0:
                ax.set_title(LABELS[model], fontsize=9)
            if model in ("classifier", "regression"):
                ax.add_patch(
                    Rectangle(
                        (15, 15), 33, 33, fill=False, edgecolor="black", ls="--", lw=0.7
                    )
                )
            if model != "gold":
                score = next(
                    r
                    for r in rows
                    if (r["region"], r["field"], r["model"])
                    == (patch["region"], patch["field"], model)
                )
                ax.text(
                    0.5,
                    -0.08,
                    f"RMSE {score['area_rmse']:.4g} m/s",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                )
        cax = fig.add_axes(
            ((width - 90) / width, (top - 192) / height, 8 / width, 192 / height)
        )
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("m/s", fontsize=8)
    fig.suptitle(
        "Surface velocity · target 2016-07-21 · +5 days · "
        + ("Prediction − OM4" if errors else "Physical fields")
        + "\n64×64 native pixels · Skyline context boxes 33×33 · common row scales",
        fontsize=11,
        y=1 - 15 / height,
    )
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--regression-cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patches = load_patches(args.arrays, args.regression_cache)
    rows = metrics(patches)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "regional-velocity-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for errors in (False, True):
        draw(
            patches,
            rows,
            args.output / f"regional-velocity-{'errors' if errors else 'fields'}.png",
            errors,
        )
    receipt = dict(
        arrays=str(args.arrays.resolve()),
        regression=json.loads(
            (
                args.regression_cache / "original-all-fields-regression-receipt.json"
            ).read_text()
        ),
        component_rows=16,
        vector_rows=8,
        units="m/s",
        increment_alignment="Area-weighted dot product / product of area-weighted L2 norms; uv_vector uses joint component mask; zero increment yields undefined alignment",
        target_atol_physical=2e-5,
    )
    with gzip.open(args.output / "regional-velocity-receipt.json.gz", "wt") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
