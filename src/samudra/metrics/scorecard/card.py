# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""From the series tables to the scorecard: one tidy frame of metric rows.

Each row is one metric for one *scope* (a single channel, or a variable over a
depth layer) and one *period*:

- ``window``: the whole scoring window, as equal-weight calendar years, with a
  calendar-year block-bootstrap interval;
- ``year``: one calendar year of the window, the blocks the interval and the
  champion comparison resample;
- ``lead``: a bin of forecast lead from the rollout's start. A single rollout
  has one sample per lead, so these rows carry no interval.

Layer rows combine levels by thickness, so they are volume-weighted: a 1000 m
level counts 200 times a 5 m one. Layers follow the Samudra 2 paper: upper
(< 700 m), intermediate (700-2000 m), deep (> 2000 m), plus the full depth.
Level-free variables (``zos``) form a single ``surface`` layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from samudra.metrics import kernels, skill
from samudra.metrics.scorecard.series import REFERENCES, Series, channel_parts

LAYERS: tuple[tuple[str, float, float], ...] = (
    ("upper", 0.0, 700.0),
    ("intermediate", 700.0, 2000.0),
    ("deep", 2000.0, np.inf),
)
FULL_DEPTH = "full_depth"
SURFACE = "surface"

# Leads in 5-day steps from the rollout's start: (label, first, last), inclusive.
LEAD_BINS: tuple[tuple[str, int, int], ...] = (
    ("step 1 (5 d)", 1, 1),
    ("steps 2-6 (to 1 mo)", 2, 6),
    ("steps 7-18 (to 3 mo)", 7, 18),
    ("steps 19-73 (to 1 yr)", 19, 73),
)

UNITS = {"thetao": "degC", "so": "psu", "uo": "m s-1", "vo": "m s-1", "zos": "m"}

# How a metric's value is judged: what "better" means.
HIGHER = "higher"
LOWER = "lower"
CLOSER_TO_0 = "closer_to_0"
CLOSER_TO_1 = "closer_to_1"
# Judged against fixed physical bounds by the gate, never ranked against a model.
BOUNDED = "bounded"
BETTER = {
    "rmse": LOWER,
    "skill_vs_persistence": HIGHER,
    "skill_vs_climatology": HIGHER,
    "bias": CLOSER_TO_0,
    "spatial_std_ratio": CLOSER_TO_1,
    "anomaly_variance_ratio": CLOSER_TO_1,
    "max": BOUNDED,
    "min": BOUNDED,
    "nonfinite_area_fraction": LOWER,
    "unphysical_area_fraction": LOWER,
}

COLUMNS = [
    "metric",
    "scope",
    "variable",
    "layer",
    "channel",
    "depth_m",
    "period_kind",
    "period",
    "value",
    "ci_low",
    "ci_high",
    "truth_value",
    "units",
    "better",
    "n_years",
]


def badness(metric: str, value):
    """Map a metric's value to a number where lower is always better."""
    v = np.asarray(value, dtype=float)
    rule = BETTER[metric]
    if rule == LOWER:
        return v
    if rule == HIGHER:
        return -v
    if rule == CLOSER_TO_0:
        return np.abs(v)
    if rule == CLOSER_TO_1:
        return np.abs(np.log(v))
    raise ValueError(f"{metric} is checked against bounds, not ranked")


@dataclass(frozen=True)
class Scope:
    """A set of channels scored together, each with its thickness weight."""

    scope: str
    variable: str
    layer: str
    channel: str
    depth_m: float
    weights: dict[str, float]


def scopes(channels: list[str], depth: dict[str, float], dz: dict[str, float]):
    """Every channel on its own, plus each variable over each layer and full depth."""
    out = [
        Scope("channel", channel_parts(c)[0], "", c, depth[c], {c: 1.0})
        for c in channels
    ]
    by_var: dict[str, list[str]] = {}
    for c in channels:
        by_var.setdefault(channel_parts(c)[0], []).append(c)
    for var, chans in by_var.items():
        if len(chans) == 1:
            out.append(Scope("layer", var, SURFACE, "", np.nan, {chans[0]: 1.0}))
            continue
        for name, top, bottom in LAYERS:
            members = {c: dz[c] for c in chans if top <= depth[c] < bottom}
            if members:
                out.append(Scope("layer", var, name, "", np.nan, members))
        out.append(
            Scope("layer", var, FULL_DEPTH, "", np.nan, {c: dz[c] for c in chans})
        )
    return out


SUM_COLUMNS = [
    "weight",
    "error",
    *(f"{skill.SE_PREFIX}{r}" for r in ("model", *REFERENCES)),
]


def _combine(sums: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Thickness-weighted per-timestamp sums for one scope, indexed by time."""
    part = sums[sums["channel"].isin(list(weights))]
    w = part["channel"].map(weights).to_numpy()
    scaled = part[SUM_COLUMNS].mul(w, axis=0)
    scaled["std_ratio"] = (part["std_model"] / part["std_truth"]).to_numpy() * w
    scaled["dz"] = w
    scaled["time"] = part["time"].to_numpy()
    out = scaled.groupby("time").sum(min_count=1)
    # The std ratio is a mean over channels, not an additive sum.
    out["std_ratio"] = out["std_ratio"] / out["dz"]
    return out.drop(columns="dz")


def _year_means(per_time: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Per calendar year: mean-square errors, mean bias and mean std ratio.

    Each year becomes one equal-weight block, whatever its number of steps.
    """
    blocks = skill.year_blocks(per_time.drop(columns="std_ratio"), years)
    index = kernels.coerce_datetime_values(np.asarray(per_time.index))
    ratio = per_time["std_ratio"].groupby(index.year).mean().loc[years]
    out = pd.DataFrame(index=blocks.index)
    for col in blocks.columns:
        if col.startswith(skill.SE_PREFIX):
            out[col] = blocks[col] / blocks["weight"]
    out["bias"] = blocks["error"] / blocks["weight"]
    out["std_ratio"] = ratio
    return out


# Each metric as a function of (mean over the selected years) of the year table.
STATISTICS: dict[str, Callable[[pd.Series], float]] = {
    "rmse": lambda m: float(np.sqrt(m["se_model"])),
    "skill_vs_persistence": lambda m: float(
        skill.skill_score(m["se_model"], m["se_persistence"])
    ),
    "skill_vs_climatology": lambda m: float(
        skill.skill_score(m["se_model"], m["se_climatology"])
    ),
    "bias": lambda m: float(m["bias"]),
    "spatial_std_ratio": lambda m: float(m["std_ratio"]),
}


def _bootstrap(
    table: pd.DataFrame, stat: Callable[[pd.Series], float], samples: int
) -> tuple[float, float]:
    """Calendar-year block bootstrap of a statistic of the equal-year means."""
    n = len(table)
    if n < kernels.MIN_BOOTSTRAP_BLOCKS or samples <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    values = table.to_numpy()
    draws = rng.integers(0, n, size=(samples, n))
    stats = np.array(
        [stat(pd.Series(values[d].mean(axis=0), index=table.columns)) for d in draws]
    )
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _row(scope: Scope, metric: str, **fields) -> dict:
    return {
        "metric": metric,
        "scope": scope.scope,
        "variable": scope.variable,
        "layer": scope.layer,
        "channel": scope.channel,
        "depth_m": scope.depth_m,
        "units": UNITS.get(scope.variable, "")
        if metric in ("rmse", "bias", "max", "min")
        else "1",
        "better": BETTER[metric],
        "ci_low": np.nan,
        "ci_high": np.nan,
        "truth_value": np.nan,
        "n_years": 0,
        **fields,
    }


def build(
    series: Series,
    *,
    depth: dict[str, float],
    dz: dict[str, float],
    years: list[int],
    bootstrap_samples: int,
) -> pd.DataFrame:
    """The scorecard frame for one rollout. See the module docstring for the rows."""
    channels = list(dict.fromkeys(series.sums["channel"]))
    sums = series.sums
    first = kernels.coerce_datetime_values(sums["time"].to_numpy()).min()
    rows: list[dict] = []
    for scope in scopes(channels, depth, dz):
        per_time = _combine(sums, scope.weights)
        table = _year_means(per_time, years)
        window = table.mean()
        for metric, stat in STATISTICS.items():
            lo, hi = _bootstrap(table, stat, bootstrap_samples)
            rows.append(
                _row(
                    scope,
                    metric,
                    period_kind="window",
                    period=f"{years[0]}-{years[-1]}",
                    value=stat(window),
                    ci_low=lo,
                    ci_high=hi,
                    n_years=len(table),
                )
            )
            for year, values in table.iterrows():
                rows.append(
                    _row(
                        scope,
                        metric,
                        period_kind="year",
                        period=str(year),
                        value=stat(values),
                        n_years=1,
                    )
                )
            rows.extend(_lead_rows(scope, metric, stat, per_time, first))
        rows.append(_anomaly_row(scope, series.anomaly_variance, years))
    rows.extend(_health_rows(series.health, years))
    frame = pd.DataFrame(rows)
    return frame[COLUMNS]


def _lead_rows(scope, metric, stat, per_time, first) -> list[dict]:
    times = kernels.coerce_datetime_values(np.asarray(per_time.index))
    step = np.arange(1, len(times) + 1)
    if not (times == times.sort_values()).all() or times[0] != first:
        raise ValueError(
            "Rollout timestamps must be sorted and start at its first step"
        )
    rows = []
    for label, lo, hi in LEAD_BINS:
        sel = per_time[(step >= lo) & (step <= hi)]
        if sel.empty:
            continue
        m = sel.drop(columns="std_ratio").sum()
        means = pd.Series(
            {
                **{
                    k: m[k] / m["weight"]
                    for k in m.index
                    if k.startswith(skill.SE_PREFIX)
                },
                "bias": m["error"] / m["weight"],
                "std_ratio": sel["std_ratio"].mean(),
            }
        )
        rows.append(
            _row(scope, metric, period_kind="lead", period=label, value=stat(means))
        )
    return rows


def _anomaly_row(scope: Scope, anomaly: pd.DataFrame, years: list[int]) -> dict:
    part = anomaly.set_index("channel").loc[list(scope.weights)]
    w = pd.Series(scope.weights)
    model = float((part["model"] * w).sum() / w.sum())
    truth = float((part["truth"] * w).sum() / w.sum())
    return _row(
        scope,
        "anomaly_variance_ratio",
        period_kind="window",
        period=f"{years[0]}-{years[-1]}",
        value=model / truth,
        n_years=len(years),
    )


def _health_rows(health: pd.DataFrame, years: list[int]) -> list[dict]:
    """Extremes and non-finite area per variable, over the window, model beside truth."""
    index = kernels.coerce_datetime_values(health["time"].to_numpy())
    in_window = np.isin(index.year, years)
    rows = []
    for quantity, group in health[in_window].groupby("quantity"):
        var, _, kind = str(quantity).partition("_")
        reduce = {
            "max": "max",
            "min": "min",
            "nonfinite_area_fraction": "max",
            "unphysical_area_fraction": "mean",
        }[kind]
        model = float(getattr(group["model"], reduce)())
        truth = float(getattr(group["truth"], reduce)())
        scope = Scope("health", var, FULL_DEPTH, "", np.nan, {})
        metric = kind
        row = _row(
            scope,
            metric,
            period_kind="window",
            period=f"{years[0]}-{years[-1]}",
            value=model,
            truth_value=truth,
            n_years=len(years),
        )
        if kind in ("max", "min"):
            row["units"] = UNITS.get(var, "")
        rows.append(row)
    return rows
