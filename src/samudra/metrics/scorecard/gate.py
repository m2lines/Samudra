# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a scorecard with the champion's, and decide whether the candidate passes.

The *champion* is the best-known model on a track (one dataset version,
resolution and variable set). Its scorecard is kept in the registry. A
candidate is compared with it row by row, paired over the same calendar years:
for each year the difference in "badness" (`card.badness`; lower is better for
every metric) is formed, and a block bootstrap over years gives an interval for
the mean difference. The verdict is:

- ``regression`` if the whole interval says the candidate is worse,
- ``improvement`` if it says the candidate is better,
- ``indistinguishable`` otherwise,
- ``no_interval`` where a metric has no per-year values to pair.

The *gate* then applies hard checks, which never depend on the champion
(the rollout is finite and within physical bounds), and the regression check
on a short list of summary rows. Everything else is reported, not gated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from samudra.metrics import skill
from samudra.metrics.scorecard import card

KEY = ["metric", "scope", "variable", "layer", "channel"]

REGRESSION = "regression"
IMPROVEMENT = "improvement"
INDISTINGUISHABLE = "indistinguishable"
NO_INTERVAL = "no_interval"


def compare(
    candidate: pd.DataFrame,
    champion: pd.DataFrame,
    *,
    bootstrap_samples: int,
    level: float = 0.95,
) -> pd.DataFrame:
    """Paired per-year comparison of every rankable metric the two scorecards share.

    Returns one row per (metric, scope) with both window values, the mean
    per-year difference in badness (candidate minus champion; negative is
    better), its interval, and the verdict.

    Raises:
        ValueError: if the scorecards cover different years, which would make
            the pairing meaningless.
    """
    cand_years = _years(candidate)
    champ_years = _years(champion)
    if cand_years != champ_years:
        raise ValueError(
            f"Candidate scores years {cand_years}, the champion {champ_years}. "
            "Score both over the same window."
        )
    rankable = [m for m, rule in card.BETTER.items() if rule != card.BOUNDED]
    out = []
    for kind, merge_on in (("window", KEY), ("year", [*KEY, "period"])):
        c = candidate[(candidate.period_kind == kind) & candidate.metric.isin(rankable)]
        h = champion[(champion.period_kind == kind) & champion.metric.isin(rankable)]
        merged = c.merge(h, on=merge_on, suffixes=("", "_champion"))
        if kind == "window":
            windows = merged
        else:
            years = merged
    per_year_groups = dict(list(years.groupby(KEY, sort=False)))
    for key, w in windows.groupby(KEY, sort=False):
        row = w.iloc[0]
        metric = str(row["metric"])
        per_year = per_year_groups.get(key)
        diff = None
        if per_year is not None and len(per_year):
            diff = card.badness(metric, per_year["value"]) - card.badness(
                metric, per_year["value_champion"]
            )
        if diff is None or not np.isfinite(diff).all():
            lo = hi = mean = np.nan
            verdict = NO_INTERVAL
        else:
            mean = float(np.mean(diff))
            lo, hi = skill.bootstrap_mean_ci(diff, bootstrap_samples, level=level)
            verdict = _verdict(lo, hi)
        out.append(
            {
                **dict(zip(KEY, key, strict=True)),
                "value": row["value"],
                "value_champion": row["value_champion"],
                "badness_diff": mean,
                "diff_ci_low": lo,
                "diff_ci_high": hi,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(out)


def _years(frame: pd.DataFrame) -> list[str]:
    return sorted(set(frame.loc[frame.period_kind == "year", "period"]))


def _verdict(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return NO_INTERVAL
    if lo > 0:
        return REGRESSION
    if hi < 0:
        return IMPROVEMENT
    return INDISTINGUISHABLE


@dataclass(frozen=True)
class GatePolicy:
    """What a candidate must satisfy. The defaults are the phase-1 policy in #986.

    Attributes:
        max_sst_above_truth: How far the rollout's warmest surface temperature
            may exceed the truth's over the window, in degC.
        min_temperature_below_truth: How far its coldest temperature anywhere
            may fall below the truth's, in degC.
        gated_metrics: Window metrics on layer scopes whose regression against
            the champion fails the gate.
        regression_level: Confidence for gated regressions. Stricter than the
            reported 95%, because many rows are tested at once.
    """

    max_sst_above_truth: float = 2.0
    min_temperature_below_truth: float = 1.0
    gated_metrics: tuple[str, ...] = ("skill_vs_climatology", "bias")
    regression_level: float = 0.99


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def checks(
    candidate: pd.DataFrame,
    policy: GatePolicy,
    champion: pd.DataFrame | None,
    bootstrap_samples: int,
) -> list[Check]:
    """The gate's hard checks, in the order they are reported."""
    health = candidate[candidate.scope == "health"].set_index(["variable", "metric"])
    out = []

    nonfinite = health.xs("nonfinite_area_fraction", level="metric")["value"]
    bad = nonfinite[nonfinite > 0]
    out.append(
        Check(
            "finite",
            bad.empty,
            "every wet cell finite at every step"
            if bad.empty
            else "non-finite values in " + ", ".join(bad.index),
        )
    )

    sst = candidate[
        (candidate.metric == "max")
        & (candidate.scope == "health")
        & (candidate.variable == "thetao")
    ]
    # The surface level dominates the warm extreme, so the variable's maximum is
    # the SST maximum to well within the tolerance.
    model_max, truth_max = (
        float(sst["value"].iloc[0]),
        float(sst["truth_value"].iloc[0]),
    )
    out.append(
        Check(
            "sst_max",
            model_max <= truth_max + policy.max_sst_above_truth,
            f"warmest {model_max:.2f} degC vs truth {truth_max:.2f} "
            f"(limit +{policy.max_sst_above_truth:g})",
        )
    )
    tmin = {k: _cell(health, "thetao", "min", k) for k in ("value", "truth_value")}
    out.append(
        Check(
            "temperature_min",
            tmin["value"] >= tmin["truth_value"] - policy.min_temperature_below_truth,
            f"coldest {tmin['value']:.2f} degC vs truth {tmin['truth_value']:.2f} "
            f"(limit -{policy.min_temperature_below_truth:g}); "
            f"{_cell(health, 'thetao', 'unphysical_area_fraction', 'value'):.2e} "
            "of wet area below -2.5 degC on average",
        )
    )
    smin = {"value": _cell(health, "so", "min", "value")}
    out.append(
        Check(
            "salinity_nonnegative",
            smin["value"] >= 0.0,
            f"freshest {smin['value']:.3f} psu; "
            f"{_cell(health, 'so', 'unphysical_area_fraction', 'value'):.2e} "
            "of wet area below 0 on average",
        )
    )

    if champion is not None:
        cmp = compare(
            candidate,
            champion,
            bootstrap_samples=bootstrap_samples,
            level=policy.regression_level,
        )
        gated = cmp[(cmp.scope == "layer") & cmp.metric.isin(policy.gated_metrics)]
        reg = gated[gated.verdict == REGRESSION]
        out.append(
            Check(
                "no_regression_vs_champion",
                reg.empty,
                f"{len(gated)} gated rows at {policy.regression_level:.0%}; "
                + (
                    "none regress"
                    if reg.empty
                    else "regressed: "
                    + ", ".join(
                        f"{r.metric} {r.variable}/{r.layer}" for r in reg.itertuples()
                    )
                ),
            )
        )
    return out


def _cell(health: pd.DataFrame, variable: str, metric: str, column: str) -> float:
    return float(np.asarray(health.loc[(variable, metric), column]))


def passed(results: list[Check]) -> bool:
    return all(c.passed for c in results)
