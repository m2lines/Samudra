# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure metric kernels for observation-based evaluation.

Implements the observation-evaluation metrics as specified in the metrics
document [1], at the revision incorporating its scientific review: instantaneous
(not time-mean) EKE, exact OHC layer-overlap weights, the exact-timestamp guard,
and calendar-year block-bootstrap uncertainty.

[1]: https://docs.google.com/document/d/1cWzIrcajfW-gvoMzV3PU0JDAouVt5xerRmEiBIUbV1s

Every function here is pure: it takes arrays and returns arrays or floats. No
file access, no plotting, no module-level mutable state, no `argparse`. That is
what lets `samudra.eval` and `samudra.viz` share them and get identical numbers.

Two conventions run through the module:

- `e_i(t)` is always *model minus observation* at cell `i` and time `t`.
- Whenever a spatial field collapses to one scalar it is area-weighted over the
  cells where model, observation, and area are all finite.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Seawater properties used for ocean heat content. Held here rather than read
# from data so that OHC values are reproducible against externally published
# numbers, which depend on the choice.
RHO = 1035.0  # kg m^-3
CP = 3850.0  # J kg^-1 K^-1
OHC_REFERENCE = "temperature_integral_relative_to_0_degC"

EARTH_RADIUS_M = 6.371e6
GRAVITY = 9.81
EARTH_ROTATION_RATE = 7.2921e-5

# Geostrophy is ill-conditioned near the equator, where the Coriolis parameter
# vanishes; DUACS comparisons drop this band entirely.
EQUATORIAL_MASK_DEG = 5.0

DEFAULT_BOOTSTRAP_SAMPLES = 10_000

# Least area-weighted fraction of a year's paired cells that must carry data
# before the year may contribute an equal-weight block. Set well below the
# coverage a healthy product gives, so it catches a genuinely starved year
# rather than ordinary gaps.
MIN_YEAR_COVERAGE = 0.5

# Fewest valid samples a cell needs before its residual variance means
# anything. Deseasonalizing removes a per-bin climatology, so a cell with about
# one sample per bin residualizes to exactly zero however variable it truly is.
# 24 admits two years of monthly data; the pentad path has 73 bins a year, so
# any cell that survives at all clears this comfortably.
MIN_RESIDUAL_SAMPLES = 24

# Fewest calendar-year blocks before a percentile bootstrap interval is worth
# reporting. With two blocks it degenerates to the min and max of two numbers.
MIN_BOOTSTRAP_BLOCKS = 5


@dataclasses.dataclass(frozen=True)
class DepthLayer:
    """A half-open depth interval `[min_depth, max_depth)` in metres."""

    min_depth: float
    max_depth: float
    label: str


OHC_LAYERS: tuple[DepthLayer, ...] = (
    DepthLayer(0.0, 700.0, "Surface (0-700m)"),
    DepthLayer(700.0, 2000.0, "Intermediate (700-2000m)"),
)


# --------------------------------------------------------------------------
# Coordinate and time coercion
# --------------------------------------------------------------------------


def coerce_datetime_values(values: np.ndarray) -> pd.DatetimeIndex:
    """Convert a time coordinate to a `DatetimeIndex`, including cftime objects."""
    flat = np.asarray(values)
    if np.issubdtype(flat.dtype, np.datetime64):
        return pd.to_datetime(flat)
    if flat.size == 0:
        return pd.DatetimeIndex([])

    first = flat.flat[0]
    if hasattr(first, "year") and hasattr(first, "month") and hasattr(first, "day"):
        return pd.to_datetime(
            [
                f"{val.year:04d}-{val.month:02d}-{val.day:02d} "
                f"{getattr(val, 'hour', 0):02d}:"
                f"{getattr(val, 'minute', 0):02d}:"
                f"{getattr(val, 'second', 0):02d}"
                for val in flat
            ]
        )
    return pd.to_datetime(flat.astype(str))


def normalize_lon[T: (xr.Dataset, xr.DataArray)](obj: T, lon_name: str = "lon") -> T:
    """Put longitudes on `[0, 360)` and sort, if they are not already."""
    lon = obj[lon_name]
    if np.nanmin(lon.values) < 0:
        obj = obj.assign_coords({lon_name: (lon % 360)}).sortby(lon_name)
    return obj


def require_exact_time_match(
    model: xr.Dataset | xr.DataArray,
    observation: xr.Dataset | xr.DataArray,
    context: str,
) -> None:
    """Fail rather than silently interpolate when model and observation times differ.

    Temporal interpolation between a model and an observation product smooths
    model variability, which deflates variance metrics and inflates RMSE. So it
    is disabled outright: timestamps must line up exactly before any horizontal
    interpolation, and a mismatch is an error the caller has to resolve.
    """
    if "time" not in model.dims or "time" not in observation.dims:
        raise ValueError(
            f"{context}: both model and observation fields must have a time dimension"
        )

    model_time = coerce_datetime_values(model["time"].values)
    obs_time = coerce_datetime_values(observation["time"].values)
    if model_time.equals(obs_time):
        return

    common = model_time.intersection(obs_time)
    model_only = model_time.difference(obs_time)
    obs_only = obs_time.difference(model_time)
    details = [
        f"model samples={len(model_time)}",
        f"observation samples={len(obs_time)}",
        f"common timestamps={len(common)}",
    ]
    if len(model_only):
        details.append(f"first model-only timestamp={model_only[0]}")
    if len(obs_only):
        details.append(f"first observation-only timestamp={obs_only[0]}")
    if not len(model_only) and not len(obs_only):
        details.append("the timestamp order differs")
    raise ValueError(
        f"{context}: model and observation timestamps must match exactly before "
        f"spatial interpolation ({'; '.join(details)}). Temporal interpolation is "
        "disabled to avoid silently smoothing model variability."
    )


def median_time_step_days(index: pd.DatetimeIndex) -> float:
    """Median spacing of a time index, in days; NaN when undefined."""
    if len(index) < 2:
        return float("nan")
    delta_days = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
    delta_days = delta_days[np.isfinite(delta_days) & (delta_days > 0)]
    return float(np.median(delta_days)) if delta_days.size else float("nan")


def complete_calendar_years(
    time: xr.DataArray, max_gap_steps: float = 2.5, min_sample_ratio: float = 0.75
) -> list[int]:
    """Years that span the full calendar with no unexplained hole in the middle.

    Completeness is judged by the largest gap between consecutive samples, not
    by a sample count. A count tolerance has to be loose enough to absorb the
    cadence variation a real product has -- OM4's 5-day calendar carries
    occasional 6-day steps -- and that same looseness then admits a fortnight
    of genuinely missing data. A gap threshold separates the two: it accepts a
    6-day step and rejects a 15-day hole, which is what "complete" is meant to
    mean.

    Args:
        time: The time coordinate to inspect.
        max_gap_steps: Largest interior gap to tolerate, in multiples of the
            median step.
        min_sample_ratio: Fewest samples a year may hold, as a fraction of the
            busiest year in the record.
    """
    index = pd.DatetimeIndex(time.values)
    if index.empty:
        return []
    step_days = median_time_step_days(index)
    if not np.isfinite(step_days) or step_days <= 0:
        return []
    # The year boundary will not land on a sample, so allow the record to start
    # and end up to one-and-a-half steps inside the calendar year.
    boundary_tolerance = pd.Timedelta(days=1.5 * step_days)
    max_gap = pd.Timedelta(days=max_gap_steps * step_days)

    # A gap threshold alone cannot see *thinning*: a year keeping every second
    # sample has gaps of exactly two median steps throughout and would pass,
    # then carry the same equal-weight block as a year with twice the data. So
    # the count is checked too, relative to the busiest year rather than to an
    # assumed cadence.
    counts = pd.Series(1, index=index).groupby(index.year).sum()
    busiest = int(counts.max())

    years = []
    for year in sorted({int(y) for y in index.year}):
        year_index = index[index.year == year].sort_values()
        if year_index.size < 2:
            continue
        if year_index.size < min_sample_ratio * busiest:
            continue
        covers_calendar = (
            year_index.min() <= pd.Timestamp(f"{year}-01-01") + boundary_tolerance
            and year_index.max()
            >= pd.Timestamp(f"{year}-12-31 23:59:59.999999999") - boundary_tolerance
        )
        largest_gap = pd.Timedelta(np.diff(year_index.values).max())
        if covers_calendar and largest_gap <= max_gap:
            years.append(year)
    return years


def require_complete_calendar_years(time: xr.DataArray, context: str) -> list[int]:
    """Return the covered years, failing if the period contains a partial one.

    Equal-year blocks are what make the primary score and its bootstrap
    comparable, so a ragged final year has to be an error rather than a silently
    down-weighted block.
    """
    index = pd.DatetimeIndex(time.values)
    present_years = sorted({int(year) for year in index.year})
    complete = complete_calendar_years(time)
    if complete != present_years:
        incomplete = sorted(set(present_years) - set(complete))
        # This fires only after the rollout, so the message has to carry the fix
        # rather than leave the reader to work out which bound is wrong. The
        # usual cause is an observation product ending days short of the year:
        # the shipped window clears the tolerance by about half a day, so one
        # extra 6-day step in a rebuilt archive is enough to trip it.
        usable = [year for year in complete if year not in incomplete]
        suggestion = (
            f" The last fully covered year is {max(usable)}, so "
            f'rmse_end="{max(usable)}-12-31" would score {len(usable)} years.'
            if usable
            else ""
        )
        raise ValueError(
            f"{context}: uncertainty-enabled primary RMSE requires complete calendar "
            f"years, but {incomplete} {'is' if len(incomplete) == 1 else 'are'} only "
            f"partly covered (data ends {index.max():%Y-%m-%d}).{suggestion}"
        )
    return complete


# --------------------------------------------------------------------------
# Depth geometry
# --------------------------------------------------------------------------


def depth_edges_from_centers(depth_vals: np.ndarray) -> np.ndarray:
    """Reconstruct cell edges from depth centers, for products that ship only centers."""
    depth = np.asarray(depth_vals, dtype=float)
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError("Expected at least two 1D depth centers")
    if not np.all(np.isfinite(depth)) or np.any(np.diff(depth) <= 0):
        raise ValueError("Depth centers must be finite and strictly increasing")
    edges = np.empty(depth.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
    edges[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    return edges


def layer_overlap_thickness(
    depth: xr.DataArray,
    min_depth: float,
    max_depth: float,
    native_dz: xr.DataArray | None = None,
) -> xr.DataArray:
    """Thickness of each native cell inside the half-open interval `[min_depth, max_depth)`.

    Boundary cells are prorated by their true overlap rather than included or
    excluded on the basis of their center depth. Selecting by center is what
    made a nominal "0-700 m" integral actually cover 0-650 m on the OM4 vertical
    grid, where a cell centered at 775 m spans 650-900 m.

    Observation products supply only depth centers, so their edges are
    reconstructed from adjacent centers. Model products additionally supply
    exact native thicknesses; pass those as `native_dz` to recover exact bounds.
    """
    min_depth = float(min_depth)
    max_depth = float(max_depth)
    if (
        not np.isfinite(min_depth)
        or not np.isfinite(max_depth)
        or max_depth <= min_depth
    ):
        raise ValueError(f"Invalid target depth interval [{min_depth}, {max_depth})")

    if native_dz is None:
        edges = depth_edges_from_centers(depth.values)
        lower = xr.DataArray(edges[:-1], dims=depth.dims, coords=depth.coords)
        upper = xr.DataArray(edges[1:], dims=depth.dims, coords=depth.coords)
    else:
        if depth.ndim != 1:
            raise ValueError(
                f"Expected a 1D native depth coordinate, got dimensions {depth.dims}"
            )
        # Accumulate thicknesses from the surface rather than assuming each
        # center sits at its cell's midpoint. The two agree on OM4, whose levels
        # tile exactly, but a uniform edge shift on some other vertical grid
        # would preserve the summed thickness while getting every boundary
        # weight wrong -- so the sum is not a check that would catch it.
        thickness = np.asarray(native_dz.values, dtype=float)
        edges = np.concatenate([[0.0], np.cumsum(thickness)])
        lower = xr.DataArray(edges[:-1], dims=depth.dims, coords=depth.coords)
        upper = xr.DataArray(edges[1:], dims=depth.dims, coords=depth.coords)

    clipped_lower = xr.where(lower > min_depth, lower, min_depth)
    clipped_upper = xr.where(upper < max_depth, upper, max_depth)
    overlap = (clipped_upper - clipped_lower).clip(min=0.0)
    overlap.name = "layer_overlap_dz"
    return overlap


# --------------------------------------------------------------------------
# Area-weighted reductions
# --------------------------------------------------------------------------


def area_weighted_mean(field: xr.DataArray, area: xr.DataArray) -> float:
    """Area-weighted mean over the finite cells of `field`."""
    mask = np.isfinite(field)
    weighted_sum = (field.where(mask) * area).sum()
    weight_sum = area.where(mask).sum()
    if float(weight_sum.values) == 0.0:
        return float("nan")
    return float((weighted_sum / weight_sum).values)


def area_weighted_map_rmse(bias: xr.DataArray, area: xr.DataArray) -> float:
    """Area-weighted RMS of a bias or per-cell RMSE map."""
    mean_square = area_weighted_mean(bias**2, area)
    if not np.isfinite(mean_square):
        return float("nan")
    return float(np.sqrt(mean_square))


def area_weighted_timeseries(
    field: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> xr.DataArray:
    """Collapse a field's spatial dims to an area-weighted time series."""
    mask = np.isfinite(field)
    weighted_sum = (field.where(mask) * area).sum(spatial_dims, skipna=True)
    weight_sum = area.where(mask).sum(spatial_dims, skipna=True)
    return weighted_sum / weight_sum


def area_weighted_pattern_corr(
    model: xr.DataArray,
    observation: xr.DataArray,
    area: xr.DataArray,
) -> float:
    """Centered pattern correlation on the common finite, positive-area mask."""
    model, observation, area = xr.align(model, observation, area, join="inner")
    valid = (
        np.isfinite(model) & np.isfinite(observation) & np.isfinite(area) & (area > 0)
    )
    weights = area.where(valid)
    weight_sum = weights.sum(skipna=True)
    weight_sum_value = float(weight_sum.values)
    if not np.isfinite(weight_sum_value) or weight_sum_value <= 0.0:
        return float("nan")

    model_mean = (model.where(valid) * weights).sum(skipna=True) / weight_sum
    obs_mean = (observation.where(valid) * weights).sum(skipna=True) / weight_sum
    model_anomaly = model - model_mean
    obs_anomaly = observation - obs_mean
    covariance = (weights * model_anomaly * obs_anomaly).sum(skipna=True)
    model_variance = (weights * model_anomaly**2).sum(skipna=True)
    obs_variance = (weights * obs_anomaly**2).sum(skipna=True)
    denominator = cast(xr.DataArray, np.sqrt(model_variance * obs_variance))
    denominator_value = float(denominator.values)
    if not np.isfinite(denominator_value) or denominator_value <= 0.0:
        return float("nan")
    return float((covariance / denominator).values)


def monthly_mean_of_complete_months(
    field: xr.DataArray, time_dim: str = "time"
) -> xr.DataArray:
    """Monthly mean, keeping only months the source actually covers end to end.

    The month at either edge of a record is usually partial. A rollout that
    stops on the 24th contributes 24 days to December while a monthly
    observation product contributes all 31, yet both reduce to a single
    December label -- so the comparison silently pits different quantities
    against each other, and every downstream check counts it as one good month.

    A month is kept when it holds at least as many samples as its length allows
    at the source's own cadence, which admits every month of a monthly product
    and rejects only genuinely short ones in a finer series.
    """
    index = pd.DatetimeIndex(field[time_dim].values)
    monthly = field.resample({time_dim: "MS"}).mean()
    step = median_time_step_days(index)
    if not np.isfinite(step) or step <= 0:
        return monthly

    counts = pd.Series(1, index=index).resample("MS").sum()
    keep = []
    for label in pd.DatetimeIndex(monthly[time_dim].values):
        days = label.days_in_month
        expected = max(1, int(days // step))
        keep.append(int(counts.get(label, 0)) >= expected)

    kept = np.array(keep, dtype=bool)
    if not kept.all():
        dropped = [
            f"{stamp:%Y-%m}"
            for stamp, ok in zip(pd.DatetimeIndex(monthly[time_dim].values), kept)
            if not ok
        ]
        logger.info("  dropping partially covered month(s): %s", ", ".join(dropped))
    return monthly.isel({time_dim: kept})


def field_rmse_over_time(
    sim: xr.DataArray, obs: xr.DataArray, context: str = "field"
) -> xr.DataArray:
    """Per-grid-cell RMSE across the shared time samples."""
    sim_aligned, obs_aligned = xr.align(sim, obs, join="inner")
    if sim_aligned.sizes.get("time", 0) == 0:
        raise ValueError(f"No common time samples for {context} RMSE")
    return cast(
        xr.DataArray,
        np.sqrt(((sim_aligned - obs_aligned) ** 2).mean("time", skipna=True)),
    )


# --------------------------------------------------------------------------
# Velocity, EKE
# --------------------------------------------------------------------------


def apply_equatorial_mask(
    da: xr.DataArray,
    lat_dim: str = "lat",
    equatorial_mask_deg: float = EQUATORIAL_MASK_DEG,
) -> xr.DataArray:
    """Blank the equatorial band, where geostrophic balance does not hold."""
    lat_vals = np.asarray(da[lat_dim].values, dtype=float)
    eq_mask = np.abs(lat_vals) >= equatorial_mask_deg
    eq_mask_da = xr.DataArray(eq_mask, dims=[lat_dim], coords={lat_dim: da[lat_dim]})
    return da.where(eq_mask_da)


def _spans_globe(lon_values: np.ndarray) -> bool:
    """Whether a longitude axis wraps the globe, so its ends are neighbours."""
    values = np.asarray(lon_values, dtype=float)
    if values.size < 2:
        return False
    spacing = float(np.median(np.diff(values)))
    if not np.isfinite(spacing) or spacing <= 0:
        return False
    # A tolerance of one whole cell is precisely the width at which "global"
    # and "one column short of global" stop being distinguishable, and treating
    # a basin as periodic would pad its western edge with eastern-edge data.
    return bool(
        np.isclose(values[-1] - values[0] + spacing, 360.0, atol=1e-6 * abs(spacing))
    )


def _differentiate_lon(field: xr.DataArray, lon_dim: str) -> xr.DataArray:
    """Longitude derivative, wrapping across the 0/360 seam on a global grid.

    `xarray.differentiate` falls back to a one-sided difference at the first and
    last index, which on a global grid means the two meridians beside the seam
    are differenced against the wrong neighbour rather than across it. That
    error propagates into the geostrophic velocity and from there into both the
    velocity RMSE and EKE. Padding one column from the opposite edge makes the
    seam an interior point.

    Regional grids are differentiated as-is: their edges really are edges.
    """
    values = np.asarray(field[lon_dim].values, dtype=float)
    if not _spans_globe(values):
        return field.differentiate(lon_dim)

    padded = _wrap_lon(field, lon_dim)
    # Concatenating single-column pads leaves 1-wide dask chunks at each edge,
    # and `np.gradient` needs at least two points per chunk. Rechunking the
    # padded axis into one block is safe here: it is a single row of longitudes,
    # and the alternative -- a partially rechunked axis -- would reintroduce the
    # same failure at whatever new boundary it created.
    if padded.chunks is not None:
        padded = padded.chunk({lon_dim: -1})
    return padded.differentiate(lon_dim).isel({lon_dim: slice(1, -1)})


def geostrophic_velocity_from_zos(
    zos: xr.DataArray,
    lat_dim: str = "y",
    lon_dim: str = "x",
    equatorial_mask_deg: float = EQUATORIAL_MASK_DEG,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Surface geostrophic velocity from sea-surface height.

    This is what makes the model comparable to DUACS, which reports geostrophic
    velocity derived from altimetric sea level rather than a model velocity.
    """
    deg_per_m = 180.0 / (np.pi * EARTH_RADIUS_M)

    lat_da = zos[lat_dim]
    f = 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(lat_da))
    coslat = np.cos(np.deg2rad(lat_da))

    deta_dy = zos.differentiate(lat_dim) * deg_per_m
    deta_dx = _differentiate_lon(zos, lon_dim) * deg_per_m / coslat

    # `f` vanishes at the equator and `coslat` at the poles, so both quotients
    # blow up there; the equatorial mask below removes the band that matters and
    # the polar rows fall outside every observation product's coverage.
    with np.errstate(divide="ignore", invalid="ignore"):
        u_g = -GRAVITY / f * deta_dy
        v_g = GRAVITY / f * deta_dx

    u_g = apply_equatorial_mask(u_g, lat_dim, equatorial_mask_deg)
    v_g = apply_equatorial_mask(v_g, lat_dim, equatorial_mask_deg)
    return u_g, v_g


def instantaneous_surface_eke(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Eddy kinetic energy at each timestamp, from velocity anomalies about the time mean.

    Instantaneous, not the EKE of the time-mean field: scoring a time-mean EKE
    map against observations measures a bias, not an RMSE, and is not comparable
    with the other per-cell time-RMSE metrics.
    """
    u_anom = u - u.mean("time")
    v_anom = v - v.mean("time")
    return 0.5 * (u_anom**2 + v_anom**2)


def mean_surface_eke(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Time-mean EKE map, for the map figures rather than for scoring."""
    return instantaneous_surface_eke(u, v).mean("time", skipna=True)


# --------------------------------------------------------------------------
# Ocean heat content
# --------------------------------------------------------------------------


def ohc_per_area_layer_maps(
    temp_field: xr.DataArray,
    layers: tuple[DepthLayer, ...] = OHC_LAYERS,
    native_dz: xr.DataArray | None = None,
    depth_name: str = "depth",
) -> dict[str, xr.DataArray]:
    """Ocean heat content per unit area, in J m^-2, for each depth layer.

    Temperature is integrated relative to 0 degrees C (see `OHC_REFERENCE`), so
    these are not IAP anomalies; comparing against published OHC numbers means
    matching both the reference and the volumetric heat capacity.
    """
    out = {}
    for layer in layers:
        overlap_dz = layer_overlap_thickness(
            temp_field[depth_name],
            layer.min_depth,
            layer.max_depth,
            native_dz=native_dz,
        )
        # A depth axis that stops short of the layer would integrate only the
        # levels it has, while the other product integrates the full column --
        # producing an OHC difference dominated by missing water rather than by
        # model error, with no outward sign. A shallow prognostic variable set
        # (say `thermo_dynamic_5`, which reaches 65 m) hits this against the
        # 0-700 m layer.
        covered = float(overlap_dz.sum())
        nominal = layer.max_depth - layer.min_depth
        if not np.isclose(covered, nominal, rtol=1e-6):
            raise ValueError(
                f"Depth axis {depth_name!r} covers {covered:.1f} m of the "
                f"{nominal:.1f} m layer [{layer.min_depth:g}, {layer.max_depth:g}) "
                f"('{layer.label}'). OHC over a partial column is not comparable "
                "against a product that spans the full layer; use a variable set "
                "that reaches the layer bottom, or disable the OHC metrics."
            )
        temp_layer = temp_field.where(overlap_dz > 0)
        # np.isfinite on a DataArray returns a DataArray, but the numpy stubs
        # widen it to ndarray, which has no `dim=` keyword.
        finite = cast(xr.DataArray, np.isfinite(temp_layer))
        # Completeness is required per *cell*, not just per axis.
        # `sum(skipna=True)` treats a masked level as zero heat, so a column
        # whose model and observation bathymetry disagree by one level reports
        # the missing water as a deficit -- about 2e9 J m^-2 for 50 m, which is
        # the size of the metric itself. The axis guard above cannot catch it:
        # that fires when the whole *grid* is too shallow, while this is a
        # single column running out early. Such cells are dropped rather than
        # entering the comparison as fictitious model error.
        wanted = overlap_dz > 0
        complete = (finite & wanted).sum(depth_name) == wanted.sum(depth_name)
        ohc = (
            (temp_layer * overlap_dz * RHO * CP)
            .sum(depth_name, skipna=True)
            .where(complete)
        )
        out[layer.label] = ohc
    return out


# --------------------------------------------------------------------------
# Anomalies: seasonal cycle, trend, residual variance
# --------------------------------------------------------------------------


def pentad_index(index: pd.DatetimeIndex) -> np.ndarray:
    """Map timestamps to 5-day climatological bins on a 365-day calendar.

    Using a no-leap calendar keeps post-February dates in the same bin across
    leap and non-leap years; February 29 joins the February 28 bin.
    """
    dayofyear = index.dayofyear.to_numpy().copy()
    is_leap = np.asarray(index.is_leap_year)
    month = index.month.to_numpy()
    day = index.day.to_numpy()
    dayofyear[is_leap & (month > 2)] -= 1
    dayofyear[is_leap & (month == 2) & (day == 29)] = 59
    return ((dayofyear - 1) // 5 + 1).astype(int)


def has_repeated_calendar_months(index: pd.DatetimeIndex) -> bool:
    """True when every calendar month appears at least twice."""
    counts = pd.Series(index.month).value_counts()
    return bool(len(counts) == 12 and (counts >= 2).all())


def _use_pentad_climatology(index: pd.DatetimeIndex) -> bool:
    """Whether the sampling is fine enough, and repeats enough, for pentad bins."""
    if (
        not np.isfinite(median_time_step_days(index))
        or median_time_step_days(index) > 10
    ):
        return False
    counts = pd.Series(pentad_index(index)).value_counts()
    return bool((counts >= 2).sum() >= 24)


def detrend_linear_dataarray(
    field: xr.DataArray, time_dim: str = "time"
) -> xr.DataArray:
    """Remove a least-squares linear trend independently at each grid cell."""
    if field.sizes.get(time_dim, 0) < 2:
        return field - field.mean(time_dim, skipna=True)

    # Elapsed days, not sample index. The two agree only on a uniform axis, and
    # the axis is not uniform once `monthly_mean_of_complete_months` drops a
    # month -- at which point an index regression silently treats the gap as if
    # no time had passed. `series_linear_trend_per_year` already uses elapsed
    # time; these two must not disagree.
    stamps = pd.DatetimeIndex(field[time_dim].values)
    elapsed = (stamps - stamps[0]).days.to_numpy(dtype=float)
    x = xr.DataArray(elapsed, dims=[time_dim], coords={time_dim: field[time_dim]})
    finite = np.isfinite(field)
    x_mean = x.where(finite).mean(time_dim, skipna=True)
    y_mean = field.mean(time_dim, skipna=True)
    cov = ((x - x_mean) * (field - y_mean)).where(finite).sum(time_dim, skipna=True)
    var = ((x - x_mean) ** 2).where(finite).sum(time_dim, skipna=True)
    slope = xr.where(var > 0, cov / var, 0.0)
    intercept = y_mean - slope * x_mean
    return field - (slope * x + intercept)


def detrended_deseasonalized(
    field: xr.DataArray, time_dim: str = "time"
) -> xr.DataArray:
    """Remove the seasonal cycle and then a linear trend, per grid cell.

    The residual is the quantity of interest: it isolates the variability a
    model either reproduces or smooths away, with the trend and seasonal cycle,
    which are easy to get right, taken out first.
    """
    if field.sizes.get(time_dim, 0) == 0:
        return field
    time_index = pd.DatetimeIndex(field[time_dim].values)

    if _use_pentad_climatology(time_index):
        pentad = xr.DataArray(
            pentad_index(time_index),
            dims=[time_dim],
            coords={time_dim: field[time_dim]},
            name="pentad",
        )
        with_pentad = field.assign_coords(pentad=pentad)
        deseasonalized = with_pentad.groupby("pentad") - with_pentad.groupby(
            "pentad"
        ).mean(time_dim, skipna=True)
        if "pentad" in deseasonalized.coords:
            deseasonalized = deseasonalized.drop_vars("pentad")
    elif has_repeated_calendar_months(time_index):
        deseasonalized = field.groupby(f"{time_dim}.month") - field.groupby(
            f"{time_dim}.month"
        ).mean(time_dim, skipna=True)
    else:
        deseasonalized = field - field.mean(time_dim, skipna=True)

    return detrend_linear_dataarray(deseasonalized, time_dim=time_dim)


def residual_variance_map(
    field: xr.DataArray, time_dim: str = "time", min_samples: int = MIN_RESIDUAL_SAMPLES
) -> xr.DataArray:
    """Variance of the detrended, deseasonalized residual at each grid cell.

    Population variance (`ddof=0`), matching the reference implementation.

    Cells with too few valid samples return NaN rather than a number. Removing
    a per-bin climatology from a cell that holds roughly one sample per bin
    leaves exactly zero by construction, and detrending cannot raise it -- so
    an undersampled cell would otherwise report "perfectly still ocean",
    indistinguishable from a genuinely quiescent point. That zero then enters
    the map RMSE and the pattern correlation as though the model's real
    variability there were pure error.
    """
    residual = detrended_deseasonalized(field, time_dim=time_dim)
    variance = residual.var(time_dim, skipna=True, ddof=0)
    valid = cast(xr.DataArray, np.isfinite(field)).sum(time_dim)
    return variance.where(valid >= min_samples)


def series_without_seasonal_cycle(series: pd.Series) -> pd.Series:
    """Remove a pentad or monthly climatology, falling back to the mean."""
    values = series.astype(float)
    index = pd.DatetimeIndex(values.index)
    if _use_pentad_climatology(index):
        pentad = pd.Series(pentad_index(index), index=values.index)
        climatology = values.groupby(pentad).mean()
        return values - pentad.map(climatology)
    if has_repeated_calendar_months(index):
        climatology = values.groupby(index.month).mean()
        seasonal = pd.Series(index.month, index=values.index).map(climatology)
        return values - seasonal
    return values - values.mean()


def series_without_linear_trend(series: pd.Series) -> pd.Series:
    """Remove a least-squares linear trend from a scalar time series."""
    values = series.astype(float)
    raw = np.asarray(values.to_numpy(), dtype=float)
    finite = np.isfinite(raw)
    if finite.sum() < 2:
        return values - values.mean()

    # Elapsed days, matching `detrend_linear_dataarray` and
    # `series_linear_trend_per_year`. On a gappy axis -- which is exactly what
    # `monthly_mean_of_complete_months` produces -- regressing on sample index
    # treats a missing month as though no time passed, so the scalar and map
    # paths would return different residuals for the same series.
    stamps = pd.DatetimeIndex(values.index)
    x = (stamps - stamps[0]).days.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x[finite], raw[finite], 1)
    trend = pd.Series(slope * x + intercept, index=values.index)
    return values - trend


def series_linear_trend_per_year(series: pd.Series) -> float:
    """Ordinary-least-squares slope in units per year."""
    values = series.astype(float)
    raw = np.asarray(values.to_numpy(), dtype=float)
    finite = np.isfinite(raw)
    if values.empty or finite.sum() < 2:
        return float("nan")
    index = pd.DatetimeIndex(values.index)
    x = (index - index[0]).days.to_numpy(dtype=float) / 365.25
    slope, _ = np.polyfit(x[finite], raw[finite], 1)
    return float(slope)


def series_residual_variance(series: pd.Series) -> float:
    """Population variance of the detrended, deseasonalized scalar series."""
    residual = series_without_linear_trend(series_without_seasonal_cycle(series))
    values = np.asarray(residual.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.var(finite))


# --------------------------------------------------------------------------
# Horizontal regridding
# --------------------------------------------------------------------------


def _wrap_lon(field: xr.DataArray, lon_dim: str = "lon") -> xr.DataArray:
    """Pad a global longitude axis with its wrapped neighbours, or pass through.

    Regional grids are returned untouched: their edges are real boundaries, and
    inventing data beyond them would be worse than dropping a column.
    """
    values = np.asarray(field[lon_dim].values, dtype=float)
    if not _spans_globe(values):
        return field
    left = field.isel({lon_dim: [-1]}).assign_coords({lon_dim: [values[-1] - 360.0]})
    right = field.isel({lon_dim: [0]}).assign_coords({lon_dim: [values[0] + 360.0]})
    return xr.concat([left, field, right], dim=lon_dim)


def _clamped_to(model_lat: np.ndarray, target_lat: xr.DataArray) -> xr.DataArray:
    """Target latitudes pulled inside the model's range, keeping their labels.

    Interpolation leaves NaN wherever the target sits outside the model's
    convex hull. Unlike longitude there is nothing to wrap to at a pole, so the
    choice is between dropping those rows -- by an amount that depends on model
    resolution -- and carrying the nearest model row outward. The gap is at
    most half a model cell, and dropping them silently changes which ocean the
    metric describes.
    """
    clamped = np.clip(
        np.asarray(target_lat.values, dtype=float),
        float(model_lat.min()),
        float(model_lat.max()),
    )
    return xr.DataArray(clamped, dims=target_lat.dims, coords=target_lat.coords)


def model_field_on_obs_grid(
    model_field: xr.DataArray,
    obs_field: xr.DataArray,
) -> xr.DataArray:
    """Interpolate a model field onto an observation product's horizontal grid.

    Horizontal only. Callers must have already established that the time axes
    match exactly (see `require_exact_time_match`); nothing here resamples time.
    """
    rename = {}
    if "y" in model_field.dims:
        rename["y"] = "lat"
    if "x" in model_field.dims:
        rename["x"] = "lon"
    if rename:
        model_field = model_field.rename(rename)
    model_field = normalize_lon(model_field, "lon").sortby("lat").sortby("lon")
    # A global model grid has no true longitude edge, but its first and last
    # centers still sit inside the observation product's outermost centers --
    # DUACS starts at 0.0625 where quarter-degree OM4 starts at 0.125. Plain
    # interpolation calls those observation columns out of bounds and drops
    # them from every reduction, so the metric quietly depends on grid
    # alignment. Wrapping one column from each end makes the seam interior.
    model_field = _wrap_lon(model_field, "lon")
    # Latitude has the same geometry and no wrap to exploit: a 1 degree model
    # tops out at +-89.5 where OISST reaches +-89.875, so the outermost
    # observation rows would fall outside the model's range and be dropped by
    # exactly the resolution-dependent amount the paragraph above rejects.
    # Clamping the *target* pulls those rows onto the model's end row, which
    # extends it poleward rather than inventing a value.
    lat_target = _clamped_to(
        np.asarray(model_field["lat"].values, dtype=float), obs_field["lat"]
    )
    interpolated = model_field.interp(
        lat=lat_target,
        lon=obs_field["lon"],
        kwargs={"fill_value": np.nan},
    )
    # `interp` labels the result with the clamped values; restore the real ones.
    interpolated = interpolated.assign_coords(lat=obs_field["lat"])
    return interpolated.where(np.isfinite(obs_field))


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def interannual_rmse_summary(
    annual_rmse: np.ndarray,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, float | int | str]:
    """Summarize a set of per-year RMSE values with spread and a bootstrap CI.

    The point estimate is the root of the equal-year mean annual MSE, so each
    calendar year carries the same weight regardless of how many samples it
    holds. Resampling whole years (rather than individual timestamps) respects
    the strong autocorrelation within a year.
    """
    values = np.asarray(annual_rmse, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "annual_std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_years": 0,
            "n_bootstrap": 0,
            "uncertainty_method": "unavailable",
            "block_aggregate_rmse": np.nan,
            "primary_aggregation_method": "root_mean_annual_mse_equal_calendar_years",
        }

    annual_mse = values**2
    block_aggregate_rmse = float(np.sqrt(annual_mse.mean()))
    annual_std = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    # Below a handful of blocks a percentile bootstrap is not an interval in any
    # useful sense -- with two years it returns exactly [min, max] of two
    # numbers -- so report no interval rather than one whose label overstates
    # what it knows.
    if values.size < MIN_BOOTSTRAP_BLOCKS or bootstrap_samples <= 0:
        ci_low = ci_high = np.nan
        n_bootstrap = 0
    else:
        # Fixed seed: the confidence interval is a property of the metric, and
        # should not shift between two runs scoring identical data.
        rng = np.random.default_rng(0)
        sampled = rng.choice(
            annual_mse, size=(bootstrap_samples, values.size), replace=True
        )
        bootstrap_rmse = np.sqrt(sampled.mean(axis=1))
        ci_low, ci_high = np.percentile(bootstrap_rmse, [2.5, 97.5]).astype(float)
        n_bootstrap = int(bootstrap_samples)

    return {
        "annual_std": annual_std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_years": int(values.size),
        "n_bootstrap": n_bootstrap,
        "uncertainty_method": (
            "calendar_year_block_bootstrap" if n_bootstrap else "insufficient_blocks"
        ),
        "block_aggregate_rmse": block_aggregate_rmse,
        "primary_aggregation_method": "root_mean_annual_mse_equal_calendar_years",
    }


def rmse_map_with_uncertainty(
    error_squared: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, str],
    context: str,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    min_year_coverage: float = MIN_YEAR_COVERAGE,
) -> tuple[xr.DataArray, dict[int, float], dict[str, float | int | str]]:
    """Per-cell RMSE map, per-year total RMSE, and the uncertainty summary.

    Args:
        error_squared: Squared model-minus-observation error, with a time dim.
        area: Grid-cell area on the same horizontal grid.
        spatial_dims: The two horizontal dim names to reduce over.
        context: Human-readable label used in error messages.
        bootstrap_samples: Block-bootstrap draws; 0 disables the interval.
        min_year_coverage: Least area-weighted fraction of a year's paired
            cells that must carry data for that year to contribute a block.

    Returns:
        The RMSE map, a mapping of year to that year's area-weighted total
        RMSE, and the summary produced by `interannual_rmse_summary`.
    """
    if error_squared.sizes.get("time", 0) == 0:
        raise ValueError(f"No time samples available for {context}")

    years = require_complete_calendar_years(error_squared["time"], context)
    annual_mse_maps = (
        error_squared.groupby("time.year").mean("time", skipna=True).sel(year=years)
    )
    # Coverage is measured on the *data*, not the time axis. A year can keep
    # every timestamp and still be almost entirely NaN -- one missing DUACS day
    # NaNs five consecutive 5-day means everywhere, and ARGO/OISST coverage is
    # genuinely patchy at high latitude and at depth. Counting stamps would let
    # such a year carry a full equal-weight block, which is exactly the
    # invariant the block design rests on.
    finite_samples = (
        cast(xr.DataArray, np.isfinite(error_squared))
        .groupby("time.year")
        .sum()
        .sel(year=years)
    )
    # The denominator is the cells that are *ever* comparable, not every cell
    # on the globe. Land is permanently NaN on these grids, as is the equatorial
    # band the geostrophic mask blanks, so counting them as missing data would
    # score a flawless DUACS year at roughly the ocean fraction of the planet
    # and fail it against any sensible threshold.
    comparable = cast(xr.DataArray, np.isfinite(error_squared)).any("time")
    stamps_per_year = (
        xr.ones_like(error_squared.isel({d: 0 for d in spatial_dims}, drop=True))
        .groupby("time.year")
        .sum()
        .sel(year=years)
    )
    total_samples = comparable * stamps_per_year
    # The map, the scalar score, and the bootstrap all rest on the same
    # equal-year blocks, so the headline number is the one the interval covers.
    rmse_map = np.sqrt(annual_mse_maps.mean("year", skipna=True))
    finite = np.isfinite(annual_mse_maps)
    weighted_sum = (annual_mse_maps.where(finite) * area).sum(spatial_dims, skipna=True)
    weight_sum = area.where(finite).sum(spatial_dims, skipna=True)
    annual_rmse = np.sqrt(weighted_sum / weight_sum)

    # Area-weighted fraction of the comparable cells that carry data, per year.
    # Weighting by area matters: losing a band of tropical cells is a bigger
    # loss of information than the same count near the pole.
    covered = (finite_samples * area).sum(spatial_dims) / (
        (total_samples * area).sum(spatial_dims)
    )

    computed = xr.Dataset(
        {
            "rmse_map": rmse_map,
            "annual_rmse": annual_rmse,
            "weight_sum": weight_sum,
            "covered": covered,
        }
    ).compute()

    coverage = dict(
        zip(years, (float(v) for v in computed["covered"].values), strict=True)
    )
    starved = {
        year: frac for year, frac in coverage.items() if not frac >= min_year_coverage
    }
    if starved:
        detail = ", ".join(f"{year}: {frac:.1%}" for year, frac in starved.items())
        raise ValueError(
            f"{context}: year(s) with too little paired data to carry an "
            f"equal-weight block ({detail}; require "
            f"{min_year_coverage:.0%}). The score and its bootstrap weight years "
            "equally, so a sparse year would distort both. Narrow the window, or "
            "check the observation product's coverage for those years."
        )

    annual_values = np.asarray(computed["annual_rmse"].values, dtype=float)
    return (
        computed["rmse_map"],
        dict(zip(years, (float(v) for v in annual_values), strict=True)),
        interannual_rmse_summary(annual_values, bootstrap_samples=bootstrap_samples),
    )
