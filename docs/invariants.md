<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Invariants

`samudra.invariants` checks ocean data against physics it must obey: masks nest with depth, water stays above
freezing, heat enters only through the surface, and conservative regridding preserves heat and salt. Use it on a
dataset you built, on a model's rollout, or on the output of a pipeline step. [Ocean physics](concepts/ocean-physics.md)
explains why each invariant holds and gives reference values for the published stores.

The checks ship with `samudra`, so a project that builds its own emulator can add `samudra` as a test dependency and
run them on its own data. They take xarray or NumPy inputs and return plain records, with no training code involved.
Releases up to 0.2.1 on PyPI don't include them yet; until the next release, install from GitHub:

```bash
pip install "samudra @ git+https://github.com/m2lines/Samudra"
```

## How a check reports

Every check returns a list of `Violation`s. An empty list means the data passes. A violation names the invariant,
says what failed with the numbers needed to judge it, and, when it applies, gives one example location:

```python
import xarray as xr
from samudra.invariants import geometry, masks

store = xr.open_zarr(
    "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4.zarr", chunks={}
)
violations = masks.flattened_store(store, time_index=0) + geometry.sphere_area(store.areacello.load())
for violation in violations:
    print(violation)
```

```text
masks.mask_matches_ocean_fraction: mask_4 disagrees with ocean_fraction > 0 in 2 cells at {'y': 161, 'x': 55}
masks.mask_matches_ocean_fraction: mask_8 disagrees with ocean_fraction > 0 in 1 cells at {'y': 168, 'x': 52}
masks.mask_matches_ocean_fraction: mask_13 disagrees with ocean_fraction > 0 in 1 cells at {'y': 171, 'x': 98}
masks.mask_matches_ocean_fraction: mask_14 disagrees with ocean_fraction > 0 in 1 cells at {'y': 174, 'x': 256}
masks.mask_matches_ocean_fraction: mask_15 disagrees with ocean_fraction > 0 in 1 cells at {'y': 175, 'x': 210}
masks.mask_matches_ocean_fraction: mask_16 disagrees with ocean_fraction > 0 in 1 cells at {'y': 167, 'x': 356}
geometry.sphere_area: total area 5.076655e+14 m2 differs from 4 pi R^2 = 5.100645e+14 m2 by 0.470%; implied radius 6356.0 km
```

Both are known deviations in v2026-09. Seven Arctic land cells have an `ocean_fraction` of up to 3 × 10⁻⁷ (#1030),
and the cell areas were computed with Earth's polar radius, 6357 km, instead of its mean radius, 6371 km (#972).

Returning records, rather than raising, lets you collect every failure from several checks before deciding what to do.
In a test, pass the list to `require_no_violations`, which raises `InvariantViolated` (an `AssertionError`) listing all
of them.

## What there is to check

Single-dataset checks hold for any one correct dataset:

| Module | Checks |
| --- | --- |
| `masks` | Masks nest with depth; `ocean_fraction` lies in [0, 1] and doesn't grow with depth; fields are NaN exactly on land. `flattened_store` runs all of these on a store laid out like ours (`thetao_0` … `thetao_18`, `mask_0` …). |
| `bounds` | Temperature at or above the freezing point for its salinity and depth; values within a range; coastal cells not dragged towards a land fill value (`coastal_dilution`). |
| `geometry` | Cell areas tile the sphere; layers tile the water column; columns and total volume match the bathymetry. |
| `timeaxis` | Times strictly increase; OM4's 5-day axis with its 6-day step at each leap-year boundary; forcing and state average over the same windows, in any cftime calendar. |
| `budgets` | Heat-content tendency tracks the surface heat flux; total salt doesn't drift; area-mean `zos` is zero. |

Metamorphic checks compare two versions of the same ocean, such as one field at two resolutions, before and after a
pipeline step, or two releases:

| Module | Checks |
| --- | --- |
| `conservation` | Area, volume, heat and salt integrals agree; variance doesn't grow and no new extremes appear when coarsening; zonal spectra agree at scales both grids resolve. Also builds the area and volume weights and heat and salt content the other checks use. |
| `coupling` | The state responds to its forcing at lag zero, not a step early or late; surface currents turn from the wind the way Ekman drift does. |
| `releases` | Two builds of a store agree everywhere, except in variables a documented change allows to differ. |

Most checks take the names of their dimensions (`y`, `x`, `lev` by default) and work on curvilinear grids. The ones
that need a regular lat-lon grid (`geometry.rectilinear_cell_area` and the zonal spectra) raise on any other grid.

## Checking a model rollout

`samudra eval` writes its predictions to `predictions.zarr`, stacked by depth (`so` with a `lev` dimension). The static
fields that weight the integrals come from the store the model was trained on. This checks that a rollout conserves
total salt about as well as the ocean it emulates:

```python
import numpy as np
import xarray as xr
from samudra.invariants import budgets, conservation, require_no_violations

truth = xr.open_zarr(
    "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4.zarr", chunks={}
)
pred = xr.open_zarr("path/to/predictions.zarr")
volume = conservation.volume_weights(truth.areacello, truth.ocean_fraction, truth.dz)

def total_salt(so):
    return conservation.salt_content(so, volume, ["lev", "y", "x"]).values

truth_so = xr.concat([truth[f"so_{k}"] for k in range(truth.sizes["lev"])], dim=truth.lev)
reference = total_salt(truth_so.sel(time=pred.time))
drift = (np.nanmax(reference) - np.nanmin(reference)) / np.nanmean(reference)

require_no_violations(budgets.salt_drift(total_salt(pred.so), max_relative=2 * drift))
```

The bound comes from the truth over the same period. Total salt in the stores isn't constant: it swings by about
3 × 10⁻⁶ within a year as sea ice freezes and melts, and trends because bottom cells count their full thickness (#973).
A fixed bound tighter than that fails on the truth too. Calibrating every tolerance on the truth first is a good habit
with any of these checks.

## Tolerances and known deviations

Each check takes tolerances, with defaults that suit the published stores. Budgets close only approximately in stored
data (5-day means, nominal layer thicknesses, float32), so tighten or loosen them for your data, and say why in the
call.

The v2026-09 stores fail a few checks for reasons we know: cell areas use the polar radius (#972), volume is about 10%
high (#973), and a handful of land cells have slightly nonzero `ocean_fraction` (#1030). To pin a known deviation in a
test without hiding new ones, mark it as an expected failure of this exception only:

```python
@pytest.mark.xfail(reason="#972: areacello uses the polar radius", raises=InvariantViolated, strict=True)
def test_cells_tile_the_sphere():
    require_no_violations(geometry.sphere_area(store.areacello.load(), rtol=1e-4))
```

With `raises=InvariantViolated`, a network error or a renamed variable still fails the test. With `strict=True`, the
test fails once the deviation is fixed, which reminds you to remove the marker.

This repository's own tests are examples: the pipeline's property tests in `data/tests/test_*_properties.py` and the
audits of the published stores in `data/tests/test_om4_live.py`. [Contributing](contributing.md) covers how we write
and run them.

## API reference

::: samudra.invariants.core
    options:
      heading_level: 3

::: samudra.invariants.masks
    options:
      heading_level: 3

::: samudra.invariants.bounds
    options:
      heading_level: 3

::: samudra.invariants.geometry
    options:
      heading_level: 3

::: samudra.invariants.timeaxis
    options:
      heading_level: 3

::: samudra.invariants.budgets
    options:
      heading_level: 3

::: samudra.invariants.conservation
    options:
      heading_level: 3

::: samudra.invariants.coupling
    options:
      heading_level: 3

::: samudra.invariants.releases
    options:
      heading_level: 3
