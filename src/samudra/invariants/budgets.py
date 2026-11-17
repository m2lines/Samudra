# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Budgets: heat enters only at the surface, salt is conserved, and mean sea level is removed.

Each budget closes only approximately in the stores (5-day means, nominal layer
thicknesses, float32), so the checks take tolerances. Comparing a budget's
residual across resolutions is sharper than any absolute tolerance: every
resolution comes from the same source, so a residual that changes with
resolution comes from processing.
"""

import numpy as np
import xarray as xr

from samudra.invariants.core import SECONDS_PER_DAY, Violation, weighted_mean


def area_mean_sea_level(
    zos: xr.DataArray, ocean_area: xr.DataArray, dims=("y", "x"), atol: float = 0.01
) -> list[Violation]:
    """Area-mean ``zos`` is about 0 m at each step (CMIP removes the global mean)."""
    mean = np.ravel(np.asarray(weighted_mean(zos, ocean_area, list(dims))))
    bad = np.abs(mean) > atol
    if not bad.any():
        return []
    i = int(np.argmax(bad))
    return [
        Violation(
            "budgets.area_mean_sea_level",
            f"{int(bad.sum())} steps have |area-mean zos| > {atol} m; e.g. {mean[i]:.4g} m",
            {"time": i},
        )
    ]


def central_tendency(series: np.ndarray, step_seconds: float) -> np.ndarray:
    """Central difference of a series of window means, per second; NaN at both ends.

    The difference between means of windows i-1 and i+1 spans the middle of
    window i, so it pairs with window i's mean flux.
    """
    series = np.asarray(series, dtype=float)
    tendency = np.full_like(series, np.nan)
    tendency[1:-1] = (series[2:] - series[:-2]) / (2 * step_seconds)
    return tendency


def heat_budget(
    heat_content: np.ndarray,
    surface_heat_flux: np.ndarray,
    step_days: float = 5.0,
    min_correlation: float = 0.9,
    slope_range: tuple[float, float] = (0.8, 1.2),
) -> list[Violation]:
    """The heat-content tendency tracks the area-integrated surface heat flux.

    Args:
        heat_content: Global (or closed-basin) heat content per step, in J.
        surface_heat_flux: Area-integrated net surface heat flux per step, in W,
            positive into the ocean.
        step_days: Spacing of the window means.
        min_correlation: Lowest acceptable correlation between the two series.
        slope_range: Acceptable regression slope of tendency on flux.
    """
    tendency = central_tendency(heat_content, step_days * SECONDS_PER_DAY)
    flux = np.asarray(surface_heat_flux, dtype=float)
    ok = np.isfinite(tendency) & np.isfinite(flux)
    if ok.sum() < 3:
        return [Violation("budgets.heat_budget", f"only {int(ok.sum())} usable steps")]
    correlation = float(np.corrcoef(tendency[ok], flux[ok])[0, 1])
    slope = float(np.polyfit(flux[ok], tendency[ok], 1)[0])
    violations = []
    if not correlation >= min_correlation:
        violations.append(
            Violation(
                "budgets.heat_budget_correlation",
                f"tendency and surface flux correlate at {correlation:.3f} < {min_correlation}",
            )
        )
    if not slope_range[0] <= slope <= slope_range[1]:
        violations.append(
            Violation(
                "budgets.heat_budget_slope",
                f"regression slope of tendency on flux is {slope:.3f}, outside {slope_range}",
            )
        )
    return violations


def salt_drift(salt: np.ndarray, max_relative: float = 1e-6) -> list[Violation]:
    """Total salt changes by at most ``max_relative`` over the series."""
    salt = np.asarray(salt, dtype=float)
    drift = (np.nanmax(salt) - np.nanmin(salt)) / np.nanmean(salt)
    if drift <= max_relative:
        return []
    return [
        Violation(
            "budgets.salt_drift",
            f"total salt varies by {drift:.3g} (relative), more than {max_relative:g}",
        )
    ]


def residuals_agree(
    residuals: dict[str, float], name: str, atol: float
) -> list[Violation]:
    """A budget's residual is the same at every resolution, to within ``atol``."""
    values = list(residuals.values())
    spread = max(values) - min(values)
    if spread <= atol:
        return []
    shown = ", ".join(f"{key}: {value:.4g}" for key, value in residuals.items())
    return [
        Violation(
            f"budgets.{name}_residual_across_resolutions",
            f"residuals spread by {spread:.4g} > {atol}: {shown}",
        )
    ]
