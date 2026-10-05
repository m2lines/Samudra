<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation results viewer

A live Panel/Bokeh application for the saved **Obs data report** arrays. The
Python server reads memory-mapped files and sends the selected slice to the
browser. No training runtime or inference is required.

## Views

| View | Fields | Controls and linked plots |
| --- | --- | --- |
| Annual surface forecasts | SST, SSH | Six model/checkpoint choices, three January origins, five-day lead slider/playback through day 365, model/reference/difference maps, click-to-select point time series |
| Initialized ocean interior | Temperature, salinity, zonal/meridional velocity | Fourteen depth centers from 2.5 to 1,850 m, observation/OM4 inputs, model/IAP/OM4/climatology comparisons, linked sections and vertical profiles |
| Monthly ocean heat content | 0–700 m and 700–2000 m layer totals; temperature depth profiles | Month slider, IAP or checkpoint comparisons, map-linked monthly heat series and selected-month temperature profiles |

The six checkpoints comprise mixed training at 50, 500, 2,000 and 8,000
observation updates, plus observation-only training at 8,000 and 16,000 updates.
Origins are January 1 of 2015, 2018 and 2021. Comparisons can also use another
checkpoint. Each browser session owns its controls and plot models.

In the initialized-interior view, origins are labeled `(obs)` or `(om4)` for
their input source and task adapter. The extended bundle contains both sources
for 2015 and 2018; observation examples also include 2021. The 2021 observation
history has no exact timestamp match in the OM4 input series. Observation
examples retain all six checkpoints; OM4
examples use the mixed 8k/8k endpoint. The checkpoint selector lists only
available examples. Velocities are the model's actual U/V fields in m/s.

Maps preserve the irregular Gaussian latitudes using cell quadrilaterals.
The map frames resize at a fixed 2:1 width-to-height ratio, with axes and color
bars outside that frame.
Sections use the actual OM4 layer interfaces. Gray denotes model land/depth
mask; white denotes missing values in wet cells. The two absolute maps share
pooled 2nd–98th-percentile color limits, and differences use symmetric limits.
Limits stay fixed during time/depth stepping by default; the rescale button
updates them explicitly. Values outside the limits remain available on hover.
Sections have separate shared limits computed over the selected full-depth section.

Observation-initialized T/S references include the **preceding December monthly
IAP analysis**, which is context for the initialized five-day state, not
instantaneous truth. IAP and its climatology have no velocity fields, so those
comparison choices are unavailable for U/V. **OM4 gold** is the native full
state for the same final five-day history interval as the OM4-input example.
For observation-input examples, that simulation state is labeled **OM4
contemporaneous context**: it is not observational truth. Observation-task
velocity slots are not directly supervised currents. The
2.5 m temperature slot contains copied available surface inputs. December
climatology uses the report's training-only 1993–2012 analyses. The displayed
RMS/bias are cosine-weighted snapshot diagnostics on paired finite cells, not
the report's full official metric suite. Forecast origin, forecast lead, and
training exposure remain distinct selectors.

Monthly heat content comes from the existing annual report exports. Forecasts
are monthly aggregates of the five-day rollout, compared with IAP monthly
analyses. Values are layer-integrated heat per unit area relative to 0 °C,
displayed in **GJ/m²** (the saved J/m² divided by 10⁹). Only complete model
columns contribute to each layer; missing IAP values remain unavailable.
The extended monthly bundle adds temperature profiles at fourteen depths
(2.5–1,850 m). Click a map to choose a cell, then move the month slider to see
its temperature profile evolve. The layer selector controls OHC maps and the
heat-content series; the temperature profile always shows the full depth range.
Changing the origin preserves the selected calendar month. Model, IAP and
checkpoint comparison choices apply to both heat and temperature.

The original annual exports did not retain monthly depth-resolved temperature.
The extended monthly view therefore uses **regenerated annual forecasts** from
the same frozen checkpoints and verified inputs. Both OHC and temperature come
from that replay, with exact calendar-overlap weighting of five-day intervals.
The local GPU/runtime differs from the original report, so numerical values can
change. The UI labels this distinction; the export receipt quantifies differences
from the report. Use **Monthly data → Original report (heat only)** to return to the original
heat maps and monthly series. Those arrays remain in the bundle under `report_heat`.
Surface and initialized-interior views continue to use their original arrays.
IAP profiles use the prepared monthly analysis on model depths, with missing
values retained; IAP heat totals retain the report's native-depth integration.

## Install and prepare

From the repository root:

```bash
uv sync --project apps/observation_viewer --group dev
uv run --project apps/observation_viewer python apps/observation_viewer/prepare.py \
  --compact /path/to/global-physical-comparison/data/compact \
  --interior /path/to/global-initializer-interior/interior \
  --interior /path/to/global-physical-comparison/scratch-initializer/interior \
  --references /path/to/global-physical-focus/native \
  --output apps/observation_viewer/.data
```

Inputs are the report's existing `.npz` exports and `COMPLETE.json` receipts.
Preparation checks report checksums, grid equivalence, reference equality
between models, and agreement between compact and full-depth initialization
fields. It verifies all output arrays by full read-back and records hashes,
source paths, checkpoint lineage and original receipts in `catalog.json`.
The source catalog is specific to this report; loading an arbitrary Zarr store
is not implemented. Input preparation refuses to overwrite a nonempty output.

The prepared bundle is ignored by Git, and
the server does not depend on the temporary source directories after preparation.
Set `SAMUDRA_VIEWER_DATA` to use a different prepared bundle.

### Extended initialization bundle

`export_interior.py` runs offline in the original report's pinned Samudra
training runtime. Its `--root` contains `contract.json`, which fixes the
producer commit, module hashes, checkpoint and training-manifest hashes, and
the paths/hashes of the six checkpoints' saved annual arrays. It runs only the
mixed endpoint initializer for two matched OM4/observation histories, verifies
reproduction of the saved observation state, and exports the last initialized
T/S/U/V state plus OM4 gold. All other observation states come directly from
saved arrays. No model is trained and no annual forecast is rerun.

Then create a new bundle with the lightweight viewer environment:

```bash
uv run --project apps/observation_viewer python apps/observation_viewer/extend_interior.py \
  --base /path/to/existing/viewer-bundle \
  --export /path/to/initialization-export/output \
  --output /path/to/new/viewer-bundle
```

Extension verifies grid, masks, source hashes, checkpoint lineage, physical
units, and exact preservation of every existing T/S value. It checks all
prepared arrays by read-back and SHA-256 and retains the export source,
contract, and receipt. Keep those data files out of Git. An older base bundle
continues to offer T/S observation examples until it is extended.

### Monthly temperature bundle

`export_monthly_temperature.py` runs offline using the report's historical
Samudra source and a training Python environment. Its contract pins module
hashes, input manifests, normalization statistics and checkpoint runs. The
`--inputs-root` may point to the original scratch tree or a verified local
mirror with the same relative paths. It does no training. For example:

```bash
PYTHONPATH=/path/to/historical/source/src python apps/observation_viewer/export_monthly_temperature.py \
  --contract /path/to/monthly-contract.json --inputs-root /path/to/mirror \
  --base /path/to/existing/viewer-bundle --output /path/to/monthly-export
uv run --project apps/observation_viewer python apps/observation_viewer/extend_monthly_temperature.py \
  --base /path/to/existing/viewer-bundle --export /path/to/monthly-export \
  --output /path/to/new/viewer-bundle
```

The extension verifies every export checksum, month, depth, mask and checkpoint
lineage, and independently integrates the temperature profiles to check that
both displayed heat layers agree. It preserves existing arrays, checks all new
arrays by full read-back, and retains the source, contract and receipts under
`monthly-temperature-provenance/`. Older bundles keep their monthly heat view
and report that temperature profiles are unavailable.

## Run and access

```bash
bash apps/observation_viewer/serve.sh
```

Open `http://localhost:61015/app`. `PORT` defaults to 61015. The launcher binds
only to localhost. For a server running on Spark, open a tunnel on your laptop:

```bash
ssh -N -L 61015:127.0.0.1:61015 spark
```

Then open the same local URL. The browser needs the live server for data
selection and playback. The service must remain running; this launcher does
not configure startup after reboot. Standard Panel resources are served by
the application, while the template may load its frontend assets from a CDN.

If your tunnel uses a different local port, allow that browser origin when
starting the server. For example, for `http://localhost:61997/app`:

```bash
# On Spark:
BROWSER_PORT=61997 bash apps/observation_viewer/serve.sh
# On your laptop (or use the equivalent existing tunnel):
ssh -N -L 61997:127.0.0.1:61015 spark
```

`PORT` selects the server's listening port; `BROWSER_PORT` selects the port in
the browser URL. Both localhost origins are allowed, with no wildcard origins.

### Managed deployment

The deployed user service `samudra-observation-viewer.service` listens on all
IPv4 interfaces at port **61015**, restarts on failure, and is enabled for the
`fomo-bot` user service manager. Its live Git checkout is
`/home/fomo-bot/samudra_viewer`, tracking **`codex/observation-viewer`** in this
repository. The prepared data remains in `apps/observation_viewer/.data` there.

The unit in `deploy/samudra-observation-viewer.service` uses `~/samudra_viewer`
and requires a private `~/.config/samudra-viewer/environment` file. That file
sets `BOKEH_ALLOW_WS_ORIGIN` to the comma-separated allowed browser hosts and
ports, including `localhost:61015` and `127.0.0.1:61015`. Public hostnames and
forwarding configuration belong only in that local file, outside Git.
Preserve it when deploying; update it if the browser hostname or port changes.

Manage the installed service as `fomo-bot`:

```bash
systemctl --user status samudra-observation-viewer
systemctl --user restart samudra-observation-viewer
journalctl --user -u samudra-observation-viewer -f
```

The service and the localhost launcher use the same port; stop the service
before running the launcher manually.

The maintenance skill is versioned in `deploy/samudra-viewer/SKILL.md` and
installed at `/home/fomo-bot/.codex/skills/samudra-viewer/SKILL.md`. It starts
from the bot's fresh Samudra checkout, switches to the viewer branch, and
requires changes to be committed and pushed before pulling into the live
checkout and restarting the service. Do not edit the deployment checkout as
the working copy.

## Verification

```bash
uv run --project apps/observation_viewer pytest \
  -c apps/observation_viewer/pyproject.toml apps/observation_viewer/tests/test_data.py

# With the server running and Playwright Chromium installed:
SAMUDRA_VIEWER_TEST_URL=http://127.0.0.1:61015/app \
  uv run --project apps/observation_viewer pytest \
  -c apps/observation_viewer/pyproject.toml apps/observation_viewer/tests/test_browser.py
```

`CHROMIUM_EXECUTABLE` can select an existing Chromium binary. Browser tests
change actual widgets, verify all displayed map values against the prepared
arrays, check profiles and sections, and check independence of two sessions.
Screenshots are written to the ignored `.screenshots` directory.

This first version focuses on saved report examples. It does not yet browse
individual training batches, arbitrary forecast interior time sequences, 3D
volumes, or diffusion outputs.
