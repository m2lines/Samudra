# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a scorecard with the champion's, and decide whether the candidate passes.

The *champion* is the best-known model on a track (one dataset version,
resolution and variable set). Its scorecard directory, including the series
tables, is kept in the registry. A candidate is compared with it on every
metric that has a window value, paired over the same calendar years:

- the point estimate is the difference of the two *window* values, each turned
  into "badness" (`card.badness`; lower is better for every metric), so it is
  exactly the difference of the numbers printed in the two scorecards;
- the interval resamples calendar years, the *same* years for both models in
  each draw, and recomputes both window statistics from the resampled years.

Resampling the window statistic itself matters: a skill score is a ratio of
means, and averaging per-year ratios instead would weight quiet years far more
than their share of the error, and can rank a worse model as better.

The verdict is ``regression`` when the whole interval says the candidate is
worse, ``improvement`` when it says better, and ``indistinguishable``
otherwise. The *gate* then applies hard checks that never depend on the
champion (the rollout is finite and within physical bounds), and the
regression check on a short list of layer-level rows. Everything else is
reported, not gated.

Departure from #986: the issue gates on lead-1 skill. A single rollout has one
sample at lead 1, so that gate waits for the hindcast mode (phase 2); phase 1
gates on window skill against climatology and on bias instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from samudra.metrics import kernels
from samudra.metrics.scorecard import card

KEY = ["metric", "scope", "variable", "layer", "channel"]

REGRESSION = "regression"
IMPROVEMENT = "improvement"
INDISTINGUISHABLE = "indistinguishable"


def compare(
    candidate: dict[card.ScopeKey, pd.DataFrame],
    champion: dict[card.ScopeKey, pd.DataFrame],
    *,
    bootstrap_samples: int,
    level: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired comparison of every window statistic the two models share.

    Args:
        candidate: `card.year_tables` of the candidate.
        champion: `card.year_tables` of the champion.
        bootstrap_samples: Paired year resamples; must be positive.
        level: Confidence level of the interval behind each verdict.
        seed: Seed for the resampling, so identical inputs give identical verdicts.

    Returns:
        One row per (metric, scope): both window values, the difference in
        badness (candidate minus champion; negative is better), its interval,
        and the verdict.

    Raises:
        ValueError: if the year tables cover different years, if there are too
            few years for a block bootstrap, or if no resamples are requested.
            Without an interval no verdict is possible, and a gate that cannot
            tell must not pass.
    """
    if bootstrap_samples <= 0:
        raise ValueError("The champion comparison needs bootstrap_samples > 0.")
    rng = np.random.default_rng(seed)
    tail = (1.0 - level) / 2.0 * 100.0
    out = []
    draws = None
    for key, cand in candidate.items():
        champ = champion.get(key)
        if champ is None:
            continue
        if list(cand.index) != list(champ.index):
            raise ValueError(
                f"Candidate scores years {list(cand.index)}, the champion "
                f"{list(champ.index)}. Score both over the same window."
            )
        n = len(cand)
        if n < kernels.MIN_BOOTSTRAP_BLOCKS:
            raise ValueError(
                f"{n} years is too few for a block bootstrap (need "
                f"{kernels.MIN_BOOTSTRAP_BLOCKS}); widen the window."
            )
        if draws is None:
            draws = rng.integers(0, n, size=(bootstrap_samples, n))
        c_vals, h_vals = cand.to_numpy(), champ[cand.columns].to_numpy()
        c_means = c_vals[draws].mean(axis=1)
        h_means = h_vals[draws].mean(axis=1)
        for metric, stat in card.STATISTICS.items():
            point_c = stat(cand.mean(skipna=False))
            point_h = stat(champ.mean(skipna=False))
            diff = float(card.badness(metric, point_c) - card.badness(metric, point_h))
            boot = np.array(
                [
                    card.badness(metric, stat(pd.Series(cm, index=cand.columns)))
                    - card.badness(metric, stat(pd.Series(hm, index=cand.columns)))
                    for cm, hm in zip(c_means, h_means, strict=True)
                ]
            )
            if not (np.isfinite(diff) and np.isfinite(boot).all()):
                lo = hi = float("nan")
                verdict = "not_finite"
            else:
                lo, hi = (float(v) for v in np.percentile(boot, [tail, 100 - tail]))
                verdict = _verdict(lo, hi)
            out.append(
                {
                    "metric": metric,
                    **dict(zip(KEY[1:], key, strict=True)),
                    "value": point_c,
                    "value_champion": point_h,
                    "badness_diff": diff,
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "verdict": verdict,
                    "level": level,
                }
            )
    if not out:
        raise ValueError("The candidate and champion share no scopes to compare.")
    return pd.DataFrame(out)


def _verdict(lo: float, hi: float) -> str:
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
    comparison: pd.DataFrame | None,
) -> list[Check]:
    """The gate's hard checks, in the order they are reported.

    Args:
        candidate: The candidate's scorecard frame.
        policy: Bounds and gated metrics.
        comparison: `compare` against the champion, at
            `policy.regression_level`; None when there is no champion.
    """
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

    if comparison is not None:
        gated = comparison[
            (comparison.scope == "layer") & comparison.metric.isin(policy.gated_metrics)
        ]
        failing = gated[gated.verdict.isin([REGRESSION, "not_finite"])]
        ok = not gated.empty and failing.empty
        if gated.empty:
            detail = (
                "no gated rows to compare: nothing shows the candidate is not worse"
            )
        elif failing.empty:
            detail = f"{len(gated)} gated rows at {policy.regression_level:.0%}; none regress"
        else:
            detail = "regressed or not finite: " + ", ".join(
                f"{r.metric} {r.variable}/{r.layer}" for r in failing.itertuples()
            )
        out.append(Check("no_regression_vs_champion", ok, detail))
    return out


def _cell(health: pd.DataFrame, variable: str, metric: str, column: str) -> float:
    return float(np.asarray(health.loc[(variable, metric), column]))


def passed(results: list[Check]) -> bool:
    return all(c.passed for c in results)
