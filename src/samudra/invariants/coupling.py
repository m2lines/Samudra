# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The forcing is a physically sensible fit to the ocean state it drives.

These checks catch forcing that is misaligned in time, or rotated or signed
differently from the state, which leaves every field plausible on its own.
"""

import numpy as np
import xarray as xr

from samudra.invariants.core import Violation, weighted_mean


def lagged_correlations(
    forcing: np.ndarray, response: np.ndarray, max_lag: int
) -> dict[int, float]:
    """Correlation of ``forcing[t]`` with ``response[t + lag]`` for each lag in [-max_lag, max_lag]."""
    forcing = np.asarray(forcing, dtype=float)
    response = np.asarray(response, dtype=float)
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = forcing[: len(forcing) - lag], response[lag:]
        else:
            a, b = forcing[-lag:], response[: len(response) + lag]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            result[lag] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return result


def response_peaks_at_lag_zero(
    forcing: np.ndarray, response: np.ndarray, name: str, max_lag: int = 2
) -> list[Violation]:
    """The response correlates best with the forcing of the same window, not the one before or after.

    Pair a flux with the central-difference tendency of what it drives (see
    `budgets.central_tendency`). A peak at lag +1 or -1 means the forcing is
    shifted by one step relative to the state.
    """
    correlations = lagged_correlations(forcing, response, max_lag)
    if not correlations:
        return [Violation(f"coupling.{name}_lag", "too few finite steps to correlate")]
    best = max(correlations, key=lambda lag: correlations[lag])
    if best == 0:
        return []
    shown = ", ".join(
        f"{lag:+d}: {value:.3f}" for lag, value in sorted(correlations.items())
    )
    return [
        Violation(
            f"coupling.{name}_lag",
            f"correlation peaks at lag {best:+d}, not 0 ({shown})",
            {"lag": best},
        )
    ]


def ekman_deflection(
    tauuo: xr.DataArray,
    tauvo: xr.DataArray,
    uo: xr.DataArray,
    vo: xr.DataArray,
    lat: xr.DataArray,
    ocean_area: xr.DataArray,
    min_abs_lat: float = 15.0,
    dims=("y", "x"),
) -> list[Violation]:
    """Surface currents turn right of the wind in the north and left in the south.

    Coriolis deflects the Ekman surface current to the right of the wind stress
    in the northern hemisphere and to the left in the southern. Averaged over a
    hemisphere, the stress-weighted cross product (tau x u) . z is negative in
    the north and positive in the south. A sign error or a swapped component in
    vector rotation flips this.
    """
    cross = tauuo * vo - tauvo * uo
    violations = []
    for hemisphere, selector, expected_sign in (
        ("north", lat >= min_abs_lat, -1),
        ("south", lat <= -min_abs_lat, 1),
    ):
        mean = float(
            weighted_mean(cross.where(selector), ocean_area, list(dims)).mean()
        )
        if not np.isfinite(mean) or np.sign(mean) != expected_sign:
            violations.append(
                Violation(
                    "coupling.ekman_deflection",
                    f"{hemisphere}ern hemisphere mean (tau x u) . z is {mean:.3g}; expected "
                    f"{'negative' if expected_sign < 0 else 'positive'}",
                    {"hemisphere": hemisphere},
                )
            )
    return violations


def stress_and_current_aligned(
    tauuo: xr.DataArray,
    tauvo: xr.DataArray,
    uo: xr.DataArray,
    vo: xr.DataArray,
    ocean_area: xr.DataArray,
    dims=("y", "x"),
) -> list[Violation]:
    """Surface current has a positive component along the wind stress, on average.

    Ekman drift is within 90 degrees of the wind, so the area mean of tau . u is
    positive. A rotation applied to one pair of vectors but not the other breaks this.
    """
    dot = tauuo * uo + tauvo * vo
    mean = float(weighted_mean(dot, ocean_area, list(dims)).mean())
    if np.isfinite(mean) and mean > 0:
        return []
    return [
        Violation(
            "coupling.stress_and_current_aligned",
            f"area-mean tau . u is {mean:.3g}; expected positive",
        )
    ]
