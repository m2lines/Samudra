<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# `data/`: the `ocean_preprocessing` subproject

`data/` is a **separate Python project** from the root `samudra` package. It builds the raw and analysis-ready,
cloud-optimized (ARCO) datasets that Samudra trains on and the observation products that `samudra.metrics` scores
against. It has its own environment, dependencies, and tests. Nothing under `data/` is covered by the root
`uv sync`, the root `pytest` run, or root mypy (mypy excludes `^data/`).

## Maintaining it

| Task | Command | Notes |
| --- | --- | --- |
| Create the environment | `cd data && mamba env create -f mamba_env.yaml && mamba activate ocean_preprocessing` | xESMF needs conda-forge, so this is mamba, not uv. Installs the package and the shared test helpers (`src/samudra/test_utils`) in editable mode. |
| Update the environment | `mamba env update -n ocean_preprocessing -f mamba_env.yaml` | |
| Install test dependencies | `mamba install pytest scipy dask pre-commit hypothesis pytest-reportlog` | `mamba_env.yaml` doesn't include them (see `data/README.md`). |
| Run tests | `cd data && python -m pytest tests` | Runs everything, property tests included, at 40 examples each. CI (`.github/workflows/data-tests.yaml`) does the same on Python 3.12 after `pip install -e ./src/samudra/test_utils -e "./data[test]"`. |
| Reproduce the property CI | `mamba env create -p <dir> -f data/.test.yml`, then `pip install --no-deps --no-build-isolation -e ./src/samudra/test_utils -e ./data` from the repository root | The pinned environment the soak and mutation gate use. It can run only the property tests (`tests/test_*_properties.py`), since it leaves out dask, xgcm, and other dependencies of the rest of the suite. |
| Run the mutation gate | `python -m samudra.test_utils.mutations data/tests --output <dir> --seeds 0 1 95920261105 --timeout 60 --budget 600` | From the repository root, in the property environment. Checks that the property tests catch each fault in the `*_mutations.py` catalogs. See "Definition of done" in the root `AGENTS.md`. |
| CLI help | `python -m ocean_preprocessing om4 -- --help` | Also `cm4`, `basin_masks`, `.virtualize_om4`, `.make_norm_datasets`, `.obs_preprocessing`. |

Dependencies live in **four places**: `data/pyproject.toml` (used by the data-test CI), `data/mamba_env.yaml` (used by
developers), and `data/.test.yml` and `data/.test-canary.yml` (used by the property CI). Keep their version pins in
sync when you change any of them.

The runtime is Zarr 3 (for VirtualiZarr), but every store we write must be **Zarr v2**, because the root `samudra`
package pins `zarr<3`.

`data/README.md` documents the pipeline stages, CLIs, and observation workflow. `docs/data.md` lists the published
datasets. Read those before changing a pipeline stage. This file covers only what an agent needs before starting.

## Data access

Everything in the NYU OSN pod (`https://nyu1.osn.mghpcc.org/m2lines-pubs/`) can be read anonymously. OSN is an
S3-compatible object store. With fsspec, set `FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/` and
`FSSPEC_S3_ANON=true`, or open the HTTPS URLs directly. Writing needs credentials. Never delete or overwrite existing
objects in the bucket without asking a maintainer.

The current OM4 training releases:

| Resolution | URL | Grid (y × x) | Size |
| --- | --- | --- | --- |
| 1° | `https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4.zarr` | 180 × 360 | ~100 GB |
| 1/2° | `https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_halfdeg/OM4.zarr` | 360 × 720 | ~400 GB |
| 1/4° | `https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_quarterdeg/OM4.zarr` | 720 × 1440 | ~2 TB |

All three share the same layout. Depth is flattened into channels (`thetao_0` … `thetao_18`), and 2-D variables
(`zos`, `hfds`, ...) have no suffix:

```
>>> import xarray as xr
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 100GB
Dimensions:         (y: 180, x: 360, lev: 19, time: 4745, y_b: 181, x_b: 361)
Coordinates:
    areacello       (y, x) float64 518kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    dz              (lev) float64 152B dask.array<chunksize=(19,), meta=np.ndarray>
    lat             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>
    lat_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>
  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03
    lon             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>
    lon_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>
    ocean_fraction  (lev, y, x) float64 10MB dask.array<chunksize=(19, 180, 360), meta=np.ndarray>
  * time            (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:...
  * x               (x) float64 3kB 0.5 1.5 2.5 3.5 ... 356.5 357.5 358.5 359.5
  * y               (y) float64 1kB -89.24 -88.25 -87.25 ... 87.25 88.25 89.24
Dimensions without coordinates: y_b, x_b
Data variables: (12/100)
    hfds            (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    mask_0          (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    ...              ...
    vo_9            (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    wfo             (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    zos             (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
Attributes: (12/17)
    grid_type:                         gaussian
    m2lines/cli_args:                  /scratch/am16581/Samudra-data-v2026-09...
    m2lines/samudra_git_hash:          https://github.com/m2lines/Samudra/com...
    ...                                ...
```

The `m2lines/*` attributes record provenance (CLI arguments, git hash, creation date). New stores must carry them too.

## Known gotchas

These are the ones we know about. The pipeline changes often, so assume there are others, and add them here when you
find them.

* **`areacello` in the v2026-09 stores is mislabeled and slightly small.** The values are in m², although the `units`
  attribute says `km2`. The label was fixed in code by #928, after these stores were written. The areas were computed
  with Earth's polar radius (6356 km) instead of its mean radius (6371 km; see `r_earth` in
  `ocean_preprocessing/preprocessing.py`), so they are 0.47% too small. Area-weighted means are unaffected, but absolute
  global integrals (total OHC in J, ocean volume) are 0.47% low. Tracked in #972.
* **`areacello` is the full cell area, land included** (the CMIP convention). Weight by `ocean_fraction` at the level
  you integrate: `areacello * ocean_fraction.isel(lev=0)` for a surface integral, and
  `areacello * ocean_fraction * dz` (summed over `lev`) for a volume integral. The boolean wet masks give a partially wet coastal cell its full area, so use them to
  select cells, not to weight them.
* **Per-level channels carry no attributes.** `thetao_0` has an empty `attrs`. Units and names live on the
  dataset-level attributes keyed by the base name (`ds.attrs["thetao"]`) and in `DataLayout.default_metadata` in
  `src/samudra/constants.py`.
* **Velocities are geographic, but the metadata isn't.** Preprocessing rotates `uo`/`vo` and `tauuo`/`tauvo` from the
  tripolar grid's x/y into east/north components, but the `long_name` still says "X"/"Y".
* **Means and snapshots both exist.** `raw/om4_5daily.zarr` holds 5-day means. `raw/om4_5daily_snapshots.zarr` holds
  instantaneous states with 5-day-mean forcings. The v2026-09 releases were built from the means, with `wfo` spliced
  in from the snapshot archive (see the `m2lines/wfo_surgery_*` attributes). Check which variant a store came from
  before comparing results across stores.
* **Calendars differ.** OM4 uses a Julian calendar (`cftime.DatetimeJulian`) and CM4 uses no-leap. Observation
  products are aligned to OM4's 5-day timestamps by a centered rolling mean, never by interpolation.
