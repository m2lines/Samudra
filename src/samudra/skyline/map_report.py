# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU summaries and maps for the pinned skyline confidence assessment."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samudra.metrics.spectra import SPATIAL_REGIONS

from .data import DEPTHS


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(
        values[
            order[
                min(
                    np.searchsorted(cumulative, quantile * cumulative[-1]),
                    len(order) - 1,
                )
            ]
        ]
    )


def weighted_ranks(values, weights):
    _, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    return (np.cumsum(mass) - 0.5 * mass)[inverse] / mass.sum()


def rank_metrics(score, error, weights, threshold):
    """Tie-aware weighted AUROC, average precision and ranked error diagnostics."""
    if np.isnan(score).any() or not np.isfinite(error).all():
        raise ValueError("Invalid detection score/error")
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    label = error > threshold
    _, inverse = np.unique(score, return_inverse=True)
    pos = np.bincount(inverse, weights=weights * label)
    neg = np.bincount(inverse, weights=weights * ~label)
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    prevalence = float(pos.sum())
    auc = (
        float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (pos.sum() * neg.sum()))
        if pos.sum() and neg.sum()
        else None
    )
    ap = float(np.sum((pos[::-1] / pos.sum()) * tp / (tp + fp))) if pos.sum() else None
    rx, ry = weighted_ranks(score, weights), weighted_ranks(error, weights)
    rx -= np.sum(weights * rx)
    ry -= np.sum(weights * ry)
    denominator = np.sqrt(np.sum(weights * rx**2) * np.sum(weights * ry**2))
    correlation = (
        float(np.sum(weights * rx * ry) / denominator) if denominator else None
    )
    # Include tied scores together; disclose the actual selected mass.
    cutoff = weighted_quantile(score, weights, 0.9)
    selected = score >= cutoff
    sse = weights * error**2
    return dict(
        auroc=auc,
        average_precision=ap,
        large_error_prevalence=prevalence,
        spearman=correlation,
        top_decile_selected_fraction=float(weights[selected].sum()),
        top_decile_sse_fraction=float(sse[selected].sum() / sse.sum())
        if sse.sum()
        else None,
        error_threshold_c=threshold,
    )


def summarize(data, weights):
    weights = weights / weights.sum()
    result: dict[str, Any] = dict(
        count=len(weights), effective_weight_count=float(1 / np.sum(weights**2))
    )
    for name in ("regression", "samudra", "persistence", "classifier_mode"):
        error = data[name + "_c"] - data["gold_c"]
        result[name + "_rmse_c"] = float(np.sqrt(np.sum(weights * error**2)))
        result[name + "_bias_c"] = float(np.sum(weights * error))
    for name in ("gold", "samudra"):
        outside = data[name + "_outside"].astype(bool)
        result[name + "_outside_fraction"] = float(np.sum(weights * outside))
        for kind in ("bin", "density"):
            values = data[f"{name}_{kind}_nll"]
            inside = ~outside
            mean = (
                float(np.average(values[inside], weights=weights[inside]))
                if inside.any()
                else None
            )
            result[f"{name}_in_support_{kind}_nll"] = mean
            result[f"{name}_{kind}_nll"] = mean if not outside.any() else None
            if kind == "bin":
                result[name + "_perplexity"] = (
                    float(np.exp(mean))
                    if mean is not None and not outside.any() and mean < 700
                    else None
                )
    for level in (0.5, 0.8, 0.9, 0.95):
        tail = (1 - level) / 2
        covered = (
            (data["gold_cdf"] >= tail)
            & (data["gold_cdf"] <= 1 - tail)
            & ~data["gold_outside"]
        )
        result[f"gold_coverage_{int(level * 100)}"] = float(np.sum(weights * covered))
    result["gold_pit_counts"] = np.histogram(
        data["gold_cdf"], bins=np.linspace(0, 1, 21), weights=weights
    )[0].tolist()
    return result


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


def equal_date_weights(data, dense_origins):
    w = np.where(np.isin(data["origin"], dense_origins), data["area"], 1.0).astype(
        float
    )
    for origin in np.unique(data["origin"]):
        chosen = data["origin"] == origin
        w[chosen] /= w[chosen].sum()
    return w / w.sum()


def render(data, name, depth, output, *, dpi=140):
    if not len(data["row"]):
        return
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    temperature = np.concatenate(
        [data[n + "_c"] for n in ("gold", "regression", "samudra", "classifier_mode")]
    )
    lo, hi = np.quantile(temperature, [0.01, 0.99])
    errors = [data[n + "_c"] - data["gold_c"] for n in ("regression", "samudra")]
    bound = max(1e-4, float(np.quantile(np.abs(np.concatenate(errors)), 0.99)))
    surprises = [data[n + "_density_nll"] for n in ("gold", "samudra")]
    finite = np.concatenate(surprises)
    finite = finite[np.isfinite(finite)]
    a, b = np.quantile(finite, [0.01, 0.99]) if len(finite) else (0, 1)
    panels = [
        (data[n + "_c"], title, "viridis", lo, hi, "°C")
        for n, title in [
            ("gold", "OM4 target (model reference)"),
            ("regression", "Skyline regression"),
            ("samudra", "Samudra"),
            ("classifier_mode", "Classifier mode"),
        ]
    ]
    panels += [
        (values, title, "RdBu_r", -bound, bound, "°C")
        for values, title in zip(
            errors, ["Regression − OM4", "Samudra − OM4"], strict=True
        )
    ]
    panels += [
        (values, title, "magma", a, b, "−log density per °C")
        for values, title in zip(
            surprises, ["OM4 surprise", "Samudra surprise"], strict=True
        )
    ]
    for ax, (values, title, cmap, vmin, vmax, unit) in zip(
        axes.flat, panels, strict=True
    ):
        p = ax.scatter(
            data["lon"],
            data["lat"],
            c=np.clip(values, vmin, vmax),
            s=2,
            marker="s",
            linewidths=0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set(title=title, xlabel="Longitude °E", ylabel="Latitude °N")
        fig.colorbar(p, ax=ax, label=unit, shrink=0.75)
    fig.suptitle(
        f"{name} · {DEPTHS[depth]:g} m · +5 days from true histories\nColor scales clip at pooled 1st/99th percentiles; metrics use all values"
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def report(root):
    receipt = json.loads((root / "receipt-rank-0.json").read_text())
    for rank in range(receipt["world_size"]):
        if not (root / f"COMPLETE-rank-{rank}.json").exists():
            raise ValueError(f"Missing completed map shard {rank}")
        if json.loads((root / f"receipt-rank-{rank}.json").read_text()) != receipt:
            raise ValueError("Map rank provenance differs")
    chunks = sorted(root.glob("points-*.npz"))
    lists: dict[str, list] = {}
    for file in chunks:
        with np.load(file) as chunk:
            for k, v in chunk.items():
                lists.setdefault(k, []).append(v)
    data = {k: np.concatenate(v) for k, v in lists.items()}
    dates = receipt["dates"]
    dense_origins = [d["origin"] for d in dates if d["dense"]]
    calibration_origins = [d["origin"] for d in dates if d["role"] == "calibration"]
    assessment_origins = [d["origin"] for d in dates if d["role"] == "assessment"]
    dense = np.isin(data["origin"], dense_origins)
    global_points = ~dense | data["global_lattice"]
    summaries, detection, thresholds, by_date = [], [], [], []
    out = root / "report"
    out.mkdir(exist_ok=True)
    for z in range(19):
        cal = subset(
            data, (data["depth"] == z) & np.isin(data["origin"], calibration_origins)
        )
        cw = equal_date_weights(cal, dense_origins)
        threshold = weighted_quantile(np.abs(cal["samudra_c"] - cal["gold_c"]), cw, 0.9)
        gold_threshold = weighted_quantile(cal["gold_density_nll"], cw, 0.95)
        thresholds.append(
            dict(
                depth=z,
                large_error_c=threshold,
                gold_density_nll_95=gold_threshold
                if np.isfinite(gold_threshold)
                else None,
            )
        )
        masks = {"Global": global_points}
        for name, lon, lat in SPATIAL_REGIONS:
            masks[name] = (
                dense
                & (data["lon"] >= lon.start)
                & (data["lon"] < lon.stop)
                & (data["lat"] >= lat.start)
                & (data["lat"] < lat.stop)
            )
        for region, mask in masks.items():
            selected = subset(
                data,
                mask
                & (data["depth"] == z)
                & np.isin(data["origin"], assessment_origins),
            )
            if not len(selected["row"]):
                continue
            w = equal_date_weights(selected, dense_origins)
            common = dict(depth=z, depth_m=float(DEPTHS[z]), region=region)
            row = dict(
                **common,
                **summarize(selected, w),
                gold_false_alarm_fraction=float(
                    np.sum(w * (selected["gold_density_nll"] > gold_threshold))
                ),
            )
            summaries.append(row)
            error = np.abs(selected["samudra_c"] - selected["gold_c"])
            scores = dict(
                samudra_density_surprise=selected["samudra_density_nll"],
                samudra_bin_surprise=selected["samudra_bin_nll"],
                predictive_std=selected["predictive_std_c"],
                samudra_regression_disagreement=np.abs(
                    selected["samudra_c"] - selected["regression_c"]
                ),
                samudra_persistence_disagreement=np.abs(
                    selected["samudra_c"] - selected["persistence_c"]
                ),
            )
            for name, score in scores.items():
                detection.append(
                    dict(
                        **common, score=name, **rank_metrics(score, error, w, threshold)
                    )
                )
            for origin in np.unique(selected["origin"]):
                single = subset(selected, selected["origin"] == origin)
                sw = equal_date_weights(single, dense_origins)
                by_date.append(
                    dict(**common, origin=int(origin), **summarize(single, sw))
                )
            if z in (0, 7, 13):
                map_data = subset(selected, np.isin(selected["origin"], dense_origins))
                render(
                    map_data,
                    region,
                    z,
                    out / f"map-{region.lower().replace(' ', '-')}-depth-{z:02d}.png",
                )
    result = dict(
        receipt=receipt,
        thresholds=thresholds,
        assessment=summaries,
        detection=detection,
        by_date=by_date,
        notes=[
            "Global dates receive equal weight; dense-date lattice points use native cell area; other dates are sampled by native cell area.",
            "Regional metrics use only the dense map date; detection thresholds come from global calibration dates at each depth.",
            "Null unconditional likelihood/perplexity means at least one target is outside finite histogram support.",
            "Density surprise can be negative; bin perplexity is exp(mean bin NLL).",
            "Central interval coverage uses the histogram CDF with uniform density within each bin.",
            "OM4 is a model reference, not observational truth. Spatial/date independence is not assumed; no independent-point confidence intervals.",
        ],
    )
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for name, rows in [
        ("per-depth", summaries),
        ("detection", detection),
        ("per-date", by_date),
    ]:
        flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
        with (out / (name + ".csv")).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    (root / "COMPLETE.json").write_text(
        json.dumps(dict(points=len(data["row"]), report=str(out))) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)


if __name__ == "__main__":
    main()
