# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, strict native-cell map comparisons and short-wave diagnostics.

No inference or interpolation is performed. Every wet prediction is required.
Short-wave power is a scale diagnostic, not proof of noise or physical realism.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CACHE = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/skyline-soft-priors-25k-artifacts/original-classifier/evaluation"
)
REGRESSION = HERE.parent / "pre-spike-noise/report/native-fields.npz"
REGIONS = ("north-pacific", "gulf-stream", "agulhas")
DEPTHS = {0: 2.5, 7: 250, 13: 1850}
UNITS = {"thetao": "degC", "uo": "m/s", "vo": "m/s", "so": "PSU"}
LABELS = {
    "gold": "OM4 model target",
    "new-temperature": "New T only",
    "new-all-fields": "New all fields",
    "original-temperature-regression": "Original T regression",
    "original-all-fields-regression": "Original all regression",
    "original-temperature-classifier": "Original T classifier",
    "original-all-fields-classifier": "Original all classifier",
    "persistence": "Persistence",
}
TREG = (
    "gold",
    "new-temperature",
    "new-all-fields",
    "original-temperature-regression",
    "original-all-fields-regression",
)
TCE = (
    "gold",
    "new-temperature",
    "new-all-fields",
    "original-temperature-classifier",
    "original-all-fields-classifier",
)
AUX = (
    "gold",
    "new-all-fields",
    "original-all-fields-regression",
    "original-all-fields-classifier",
    "persistence",
)

# Reuse the tested native CPU spectral helpers, without running that script's CLI.
_spec = importlib.util.spec_from_file_location(
    "pre_spike_spectral_helpers", HERE.parent / "pre-spike-noise/render.py"
)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
wet_patch = _helpers.wet_patch
short_wave_power = _helpers.short_wave_power


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def same(a, b, name, atol=0):
    if np.shape(a) != np.shape(b):
        raise ValueError(f"{name}: shape mismatch {np.shape(a)} != {np.shape(b)}")
    if atol:
        if not np.allclose(a, b, rtol=0, atol=atol, equal_nan=False):
            raise ValueError(f"{name}: values differ (absolute tolerance {atol})")
    elif not np.array_equal(a, b):
        raise ValueError(f"{name}: values differ")


def validate_map(a, field, path):
    wet = a["wet"]
    if wet.shape != (64, 64) or wet.dtype != np.bool_ or not wet.any():
        raise ValueError(f"{path}: expected nonempty boolean 64×64 mask")
    if a["lat"].shape != (64,) or a["lon"].shape != (64,):
        raise ValueError(f"{path}: coordinate axes must have 64 values")
    for key in ("lat", "lon"):
        if not np.isfinite(a[key]).all() or not np.all(np.diff(a[key]) > 0):
            raise ValueError(f"{path}: nonfinite or unordered {key}")
    rows, cols = a["query_rows"], a["query_cols"]
    if rows.shape != (64, 64) or cols.shape != (64, 64):
        raise ValueError(f"{path}: invalid query grid")
    same(rows, np.broadcast_to(rows[:, :1], (64, 64)), "row mesh")
    same(cols, np.broadcast_to(cols[:1, :], (64, 64)), "col mesh")
    if not (np.all(np.diff(rows[:, 0]) == 1) and np.all(np.diff(cols[0]) == 1)):
        raise ValueError(f"{path}: noncontiguous native cells")
    if str(a[f"{field}_units"]) != UNITS[field]:
        raise ValueError(f"{path}: incorrect units for {field}")
    for key in ("area", f"{field}_prediction", f"{field}_gold", f"{field}_persistence"):
        if a[key].shape != wet.shape or not np.isfinite(a[key][wet]).all():
            raise ValueError(f"{path}: invalid or missing wet values: {key}")
    if not np.all(a["area"][wet] > 0):
        raise ValueError(f"{path}: nonpositive cell area")


def matched(a, b, field, name):
    for key in (
        "wet",
        "lat",
        "lon",
        "query_rows",
        "query_cols",
        "origin",
        "depth",
        "input_date",
        "target_date",
        "split",
        "area",
    ):
        same(a[key], b[key], f"{name}/{key}")
    for key in (f"{field}_gold", f"{field}_persistence"):
        same(a[key][a["wet"]], b[key][a["wet"]], f"{name}/{key}", atol=2e-5)


def axis_indices(source, target, name):
    indices = np.searchsorted(source, target)
    if np.any(indices >= len(source)):
        raise ValueError(f"Cached {name} does not cover requested grid")
    same(source[indices], target, f"Cached {name}", atol=1e-10)
    return indices


def regression_prediction(archive, a, region, model):
    prefix = f"{model}-r{region}-z{int(a['depth']):02d}/"
    b = {k.removeprefix(prefix): v for k, v in archive.items() if k.startswith(prefix)}
    if not b:
        raise ValueError(f"Missing regression archive group {prefix}")
    for key in ("origin", "depth"):
        same(a[key], b[key], prefix + key)
    yi = axis_indices(b["lat"], a["lat"], "latitude")
    xi = axis_indices(b["lon"], a["lon"], "longitude")
    # Standard tasks must be the centered native 64-square of the old region.
    same(
        yi,
        np.arange((len(b["lat"]) - 64) // 2, (len(b["lat"]) - 64) // 2 + 64),
        "centered cached rows",
    )
    same(
        xi,
        np.arange((len(b["lon"]) - 64) // 2, (len(b["lon"]) - 64) // 2 + 64),
        "centered cached cols",
    )
    ix = np.ix_(yi, xi)
    same(a["wet"], b["wet"][ix], prefix + "wet")
    same(a["area"], b["area"][ix], prefix + "area", atol=1e-10)
    same(
        a["thetao_gold"][a["wet"]], b["gold"][ix][a["wet"]], prefix + "gold", atol=2e-5
    )
    result = b["prediction"][ix]
    if not np.isfinite(result[a["wet"]]).all():
        raise ValueError(f"Missing wet predictions in {prefix}")
    return result


def classifier_prediction(cache, a):
    """Join point-cache records exactly; duplicate regional/lattice records agree."""
    origin, depth = int(a["origin"]), int(a["depth"])
    result = np.full((64, 64), np.nan, dtype=np.float32)
    seen = np.zeros((64, 64), dtype=bool)
    r0, c0 = int(a["query_rows"][0, 0]), int(a["query_cols"][0, 0])
    files = sorted(cache.glob(f"points-{origin}-{depth:02d}-*.npz"))
    if not files:
        raise ValueError(f"Missing classifier cache for origin {origin}, depth {depth}")
    for path in files:
        b = load_npz(path)
        y, x = b["row"] - r0, b["col"] - c0
        use = (y >= 0) & (y < 64) & (x >= 0) & (x < 64)
        for i in np.flatnonzero(use):
            yy, xx = int(y[i]), int(x[i])
            if not a["wet"][yy, xx]:
                raise ValueError(f"{path}: cached wet point is dry in map")
            for key, expected in (("origin", origin), ("depth", depth)):
                same(b[key][i], expected, f"{path}/{key}")
            for key, expected, atol in (
                ("lat", a["lat"][yy], 1e-10),
                ("lon", a["lon"][xx], 1e-10),
                ("area", a["area"][yy, xx], 1e-10),
                ("gold_c", a["thetao_gold"][yy, xx], 2e-5),
                ("persistence_c", a["thetao_persistence"][yy, xx], 2e-5),
            ):
                same(b[key][i], expected, f"{path}/{key}", atol=atol)
            value = b["classifier_mode_c"][i]
            if not np.isfinite(value):
                raise ValueError(f"{path}: missing wet classifier prediction")
            if seen[yy, xx]:
                same(
                    result[yy, xx], value, "Duplicate classifier predictions", atol=2e-5
                )
            result[yy, xx], seen[yy, xx] = value, True
    same(seen, a["wet"], "Classifier cache wet coverage")
    return result


def build_patch(root, task, field, names):
    maps = {}
    for name in names:
        path = root / f"{name}-{task}.npz"
        a = load_npz(path)
        validate_map(a, field, path)
        expected_crop = 17 if name.startswith("new-") else 33
        same(a["crop"], expected_crop, f"{path}/crop")
        maps[name] = a
    ref = maps[names[0]]
    for name, a in maps.items():
        matched(ref, a, field, name)
    data = {"gold": ref[f"{field}_gold"], "persistence": ref[f"{field}_persistence"]}
    data.update({name: a[f"{field}_prediction"] for name, a in maps.items()})
    return dict(task=task, field=field, ref=ref, data=data, tile=wet_patch(ref["wet"]))


def load_patches(root, cache, regression):
    archive = load_npz(regression)
    temp, velocity, salinity = [], [], []
    for region, name in enumerate(REGIONS):
        for depth in DEPTHS:
            task = f"{name}-z{depth:02d}"
            p = build_patch(
                root,
                task,
                "thetao",
                ("new-temperature", "new-all-fields", "original-all-fields-classifier"),
            )
            same(p["ref"]["depth"], depth, task + " depth")
            same(p["ref"]["origin"], 4273, task + " origin")
            for label, model in (
                ("original-temperature-regression", "temperature25k"),
                ("original-all-fields-regression", "fields25k"),
            ):
                p["data"][label] = regression_prediction(
                    archive, p["ref"], region, model
                )
            p["data"]["original-temperature-classifier"] = classifier_prediction(
                cache, p["ref"]
            )
            temp.append(p)
            if name != "north-pacific" and depth == 0:
                for field in ("uo", "vo"):
                    q = build_patch(root, task, field, AUX[1:-1])
                    matched(p["ref"], q["ref"], "thetao", task)
                    velocity.append(q)
    for task, origin in (("black-sea-june", 4121), ("black-sea-september", 4142)):
        p = build_patch(root, task, "so", AUX[1:-1])
        same(p["ref"]["origin"], origin, task + " origin")
        same(p["ref"]["depth"], 13, task + " depth")
        same(p["ref"]["query_rows"][:, 0], np.arange(502, 566), task + " rows")
        same(p["ref"]["query_cols"][0], np.arange(106, 170), task + " cols")
        salinity.append(p)
    return temp, velocity, salinity


def analyze(patches):
    metrics = []
    for p in patches:
        a, data = p["ref"], p["data"]
        wet = a["wet"]
        weight = a["area"][wet] / a["area"][wet].sum()
        tile = p["tile"]
        for model, values in data.items():
            if not np.isfinite(values[wet]).all():
                raise ValueError(f"{p['task']}/{model}: missing wet prediction")
            error = values[wet].astype(float) - data["gold"][wet]
            row = dict(
                task=p["task"],
                field=p["field"],
                model=model,
                units=UNITS[p["field"]],
                origin=int(a["origin"]),
                target_date=str(a["target_date"]),
                depth_m=DEPTHS[int(a["depth"])],
                wet_count=int(wet.sum()),
                area_rmse=float(np.sqrt(np.sum(weight * error**2))),
                area_bias=float(np.sum(weight * error)),
                spectrum_side=None,
                tile_row=None,
                tile_col=None,
                shortwave_prediction_power=None,
                shortwave_error_power=None,
                shortwave_om4_power=None,
                shortwave_prediction_ratio_om4=None,
                shortwave_error_ratio_om4=None,
                spectrum_status="no fully wet 32x32 tile",
            )
            if tile is not None:
                y, x, side = tile
                cut = np.s_[y : y + side, x : x + side]
                gold, prediction = data["gold"][cut], values[cut]
                if not (np.isfinite(gold).all() and np.isfinite(prediction).all()):
                    raise ValueError("Nonfinite spectrum input")
                pg, pp, pe = (
                    short_wave_power(gold),
                    short_wave_power(prediction),
                    short_wave_power(prediction - gold),
                )
                row.update(
                    spectrum_side=side,
                    tile_row=y,
                    tile_col=x,
                    shortwave_prediction_power=pp,
                    shortwave_error_power=pe,
                    shortwave_om4_power=pg,
                    shortwave_prediction_ratio_om4=pp / pg if pg > 0 else None,
                    shortwave_error_ratio_om4=pe / pg if pg > 0 else None,
                    spectrum_status="ok"
                    if pg > 0
                    else "zero OM4 band power; ratios undefined",
                )
            metrics.append(row)
    return metrics


def draw(patches, names, metrics, output, stem, errors=False):
    if errors:
        names = names[1:]
    # Each native cell occupies exactly 3×3 rendered pixels, without interpolation.
    panel, stride, left, row_height = 192, 220, 175, 245
    width, height = left + len(names) * stride + 90, 85 + len(patches) * row_height
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    index = {(r["task"], r["field"], r["model"]): r for r in metrics}
    for i, p in enumerate(patches):
        a, data, wet = p["ref"], p["data"], p["ref"]["wet"]
        top = height - 65 - i * row_height
        # Use all compared models for identical ranges across the two T sets.
        if errors:
            limit = max(
                float(np.max(np.abs(v[wet] - data["gold"][wet])))
                for n, v in data.items()
                if n != "gold"
            )
            vmin, vmax, cmap = -max(limit, 1e-12), max(limit, 1e-12), "RdBu_r"
        else:
            vmin = min(float(np.min(v[wet])) for v in data.values())
            vmax = max(float(np.max(v[wet])) for v in data.values())
            if vmin == vmax:
                vmin, vmax = vmin - 1e-6, vmax + 1e-6
            cmap = "viridis"
        unit = "°C" if p["field"] == "thetao" else UNITS[p["field"]]
        fig.text(
            8 / width,
            (top - panel / 2) / height,
            f"{p['task']}\n{p['field']} · {DEPTHS[int(a['depth'])]:g} m\n{str(a['target_date'])[:10]}\n64×64 native cells",
            fontsize=9,
            va="center",
        )
        for j, name in enumerate(names):
            ax = fig.add_axes(
                (
                    (left + j * stride) / width,
                    (top - panel) / height,
                    panel / width,
                    panel / height,
                )
            )
            value = data[name] - data["gold"] if errors else data[name]
            im = ax.imshow(
                np.ma.masked_where(~wet, value),
                origin="lower",
                interpolation="none",
                resample=False,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_axis_off()
            if name not in {"gold", "persistence"}:
                crop = 17 if name.startswith("new-") else 33
                ax.add_patch(
                    Rectangle(
                        (31.5 - crop / 2, 31.5 - crop / 2),
                        crop,
                        crop,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.2,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (31.5 - crop / 2, 31.5 - crop / 2),
                        crop,
                        crop,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.5,
                        linestyle="--",
                    )
                )
                ax.text(
                    0.02,
                    0.02,
                    f"{crop}×{crop}",
                    transform=ax.transAxes,
                    color="black",
                    fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                )
            if i == 0:
                ax.set_title(LABELS[name], fontsize=9, pad=13)
            if name != "gold":
                row = index[p["task"], p["field"], name]
                ax.text(
                    0.5,
                    -0.055,
                    f"Area RMSE {row['area_rmse']:.4g} {unit}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
        cax = fig.add_axes(
            ((width - 65) / width, (top - panel) / height, 10 / width, panel / height)
        )
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8)
        cax.set_title(unit, fontsize=8)
    fig.text(
        0.5,
        0.992,
        ("Prediction − OM4" if errors else "Physical fields; OM4 is model context")
        + " · boxes: input context, to scale",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.savefig(output / f"{stem}-{'errors' if errors else 'fields'}.png", dpi=100)
    plt.close(fig)


def save_sources(patches, output):
    payload = {}
    for p in patches:
        prefix = f"{p['task']}/{p['field']}/"
        for key in (
            "wet",
            "lat",
            "lon",
            "area",
            "query_rows",
            "query_cols",
            "depth",
            "origin",
            "target_date",
        ):
            payload[prefix + key] = p["ref"][key]
        payload[prefix + "units"] = np.asarray(UNITS[p["field"]])
        for name, values in p["data"].items():
            payload[prefix + name] = np.where(p["ref"]["wet"], values, np.nan).astype(
                np.float32
            )
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    dest = output / "source-fields.npz"
    if stream.tell() > 5_000_000:
        raise ValueError(
            f"Compact archive exceeds 5 MB ({stream.tell()} bytes); retain source artifacts instead"
        )
    dest.write_bytes(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classifier-cache", type=Path, default=CACHE)
    parser.add_argument("--regression-archive", type=Path, default=REGRESSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = load_patches(args.root, args.classifier_cache, args.regression_archive)
    patches = [p for group in groups for p in group]
    metrics = analyze(patches)
    args.output.mkdir(parents=True, exist_ok=True)
    save_sources(patches, args.output)
    with (args.output / "map-metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for group, names, stem in (
        (groups[0], TREG, "temperature-regression"),
        (groups[0], TCE, "temperature-classifier"),
        (groups[1], AUX, "velocity"),
        (groups[2], AUX, "salinity"),
    ):
        for errors in (False, True):
            draw(group, names, metrics, args.output, stem, errors)
    manifest = dict(
        root=str(args.root.resolve()),
        classifier_cache=str(args.classifier_cache.resolve()),
        regression_archive=str(args.regression_archive.resolve()),
        regression_archive_sha256=hashlib.sha256(
            args.regression_archive.read_bytes()
        ).hexdigest(),
        regression_models=["temperature25k", "fields25k"],
        classifier_readout="classifier_mode_c",
        gold_atol_physical=2e-5,
        spectrum_method="Same fully wet 64/32 tile per row; planar detrend; periodic Hann; radial 0.25..0.5 cycles/native cell (2–4 cell wavelengths); corners excluded; power in squared field units. Short-wave power is not proof of noise.",
        maps="Native 64×64 cells; no interpolation; same field/error limits across each row, including both temperature comparison sets; context boxes are 17 or 33 cells.",
    )
    (args.output / "map-render-receipt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
