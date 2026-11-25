# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Hypothesis strategies for small, physically shaped ocean datasets.

Grids carry cell vertices, masks nest with depth, fields are smooth plus noise,
and time axes use the calendars and five-day windows of the real archives. Keep
sizes small: these feed xESMF and gcm-filters inside property tests.
"""

import cftime
import numpy as np
import xarray as xr
from hypothesis import strategies as st

CALENDARS = ("julian", "noleap", "proleptic_gregorian")
# OM4 nominal interfaces (m); vertical_cell_metadata's fallback.
OM4_INTERFACES = np.array(
    [
        0,
        5,
        15,
        30,
        50,
        80,
        130,
        200,
        300,
        450,
        650,
        900,
        1200,
        1600,
        2100,
        2700,
        3500,
        4500,
        5500,
        6500,
    ],
    dtype=float,
)


def regular_grid(ny: int, nx: int, lon0: float = 0.0) -> xr.Dataset:
    """A global regular lat-lon grid in the pipeline's layout: lon/lat (y, x), lon_b/lat_b (y_b, x_b)."""
    lat_edges = np.linspace(-90.0, 90.0, ny + 1)
    lon_edges = lon0 + np.linspace(0.0, 360.0, nx + 1)
    lat_c = 0.5 * (lat_edges[1:] + lat_edges[:-1])
    lon_c = 0.5 * (lon_edges[1:] + lon_edges[:-1])
    lon_b, lat_b = np.meshgrid(lon_edges, lat_edges)
    lon, lat = np.meshgrid(lon_c, lat_c)
    return xr.Dataset(
        coords={
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
            "lon_b": (("y_b", "x_b"), lon_b),
            "lat_b": (("y_b", "x_b"), lat_b),
        }
    )


@st.composite
def grid_pairs(draw, max_coarse=(6, 12), factors=(2, 3)):
    """A coarse global grid and a fine one whose cells nest exactly inside it."""
    ny = draw(st.integers(3, max_coarse[0]))
    nx = draw(st.integers(4, max_coarse[1]))
    factor = draw(st.sampled_from(factors))
    return regular_grid(ny * factor, nx * factor), regular_grid(ny, nx), factor


def smooth_field(
    rng: np.random.Generator, ny: int, nx: int, waves: int = 3
) -> np.ndarray:
    """Sum of a few random zonal/meridional waves plus white noise, periodic in x."""
    y = np.linspace(0, np.pi, ny)[:, None]
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)[None, :]
    field = np.zeros((ny, nx))
    for _ in range(waves):
        k, m = rng.integers(0, 4), rng.integers(0, 4)
        field += (
            rng.normal()
            * np.cos(k * x + rng.uniform(0, 2 * np.pi))
            * np.cos(m * y + rng.uniform(0, np.pi))
        )
    return field + 0.1 * rng.normal(size=(ny, nx))


@st.composite
def nested_wet_masks(draw, ny: int, nx: int, nlev: int, land: bool = True):
    """(lev, y, x) boolean masks that nest with depth, from a random bottom level per column.

    At least one column is wet to the bottom, so every level has ocean.
    """
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    if land:
        bottom = rng.integers(0, nlev + 1, size=(ny, nx))
        bottom[rng.integers(ny), rng.integers(nx)] = nlev
    else:
        bottom = np.full((ny, nx), nlev)
    levels = np.arange(nlev)[:, None, None]
    return levels < bottom[None]


@st.composite
def ocean_state(
    draw, grid: xr.Dataset, nlev: int = 3, ntime: int = 2, land: bool = True
):
    """thetao, so, uo, vo on (time, lev, y, x) and zos, hfds on (time, y, x), NaN on land."""
    ny, nx = grid.sizes["y"], grid.sizes["x"]
    wet = draw(nested_wet_masks(ny, nx, nlev, land=land))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))

    def field3(mean, scale):
        data = np.stack(
            [
                np.stack(
                    [mean + scale * smooth_field(rng, ny, nx) for _ in range(nlev)]
                )
                for _ in range(ntime)
            ]
        )
        return (("time", "lev", "y", "x"), np.where(wet[None], data, np.nan))

    def field2(mean, scale):
        data = np.stack(
            [mean + scale * smooth_field(rng, ny, nx) for _ in range(ntime)]
        )
        return (("time", "y", "x"), np.where(wet[0][None], data, np.nan))

    interfaces = OM4_INTERFACES[: nlev + 1]
    ds = xr.Dataset(
        {
            "thetao": field3(10.0, 5.0),
            "so": field3(35.0, 0.5),
            "uo": field3(0.0, 0.2),
            "vo": field3(0.0, 0.2),
            "zos": field2(0.0, 0.3),
            "hfds": field2(0.0, 50.0),
        },
        coords={
            "lev": 0.5 * (interfaces[1:] + interfaces[:-1]),
            "dz": ("lev", np.diff(interfaces)),
            "wetmask": (("lev", "y", "x"), wet),
            "time": cftime.num2date(
                np.arange(ntime) * 5 + 2.5, "days since 2000-01-01", "julian"
            ),
        },
    )
    return ds.assign_coords(grid.coords)


@st.composite
def five_day_axes(draw, calendar: str | None = None, max_years: int = 3):
    """OM4-style axes: per year, 73 windows of five days from January 1; stamps at midpoints.

    Starts in a random year and spans up to ``max_years`` year boundaries, so
    leap years are crossed often. Returns (stamps, bounds) as cftime objects.
    """
    calendar = calendar or draw(st.sampled_from(CALENDARS))
    first_year = draw(st.integers(1958, 2020))
    years = draw(st.integers(1, max_years))
    stamps, bounds = [], []
    for year in range(first_year, first_year + years):
        start = cftime.datetime(year, 1, 1, calendar=calendar)
        for i in range(73):
            begin = start + i * np.timedelta64(5, "D").astype(object)
            end = begin + np.timedelta64(5, "D").astype(object)
            bounds.append((begin, end))
            stamps.append(begin + np.timedelta64(60, "h").astype(object))
    first = draw(st.integers(0, len(stamps) - 2))
    last = draw(st.integers(first + 2, len(stamps)))
    return np.array(stamps[first:last]), np.array(bounds[first:last])
