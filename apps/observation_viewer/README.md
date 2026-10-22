<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation results viewer

A live Panel/Bokeh application for the saved **Obs data report** arrays. The
Python server reads memory-mapped files and sends the selected slice to the
browser. No training runtime or inference is required.

## Views

| View | Fields | Controls and linked plots |
| --- | --- | --- |
| Annual surface forecasts | SST, SSH | Model/checkpoint choices, three January origins, five-day lead slider/playback through day 365, model/reference/difference maps, click-to-select point time series |
| Initialized ocean interior | Temperature, salinity, zonal/meridional velocity | Fourteen depth centers from 2.5 to 1,850 m, observation/OM4 inputs, model/IAP/OM4/climatology comparisons, linked sections and vertical profiles |
| Monthly ocean heat content | 0–700 m and 700–2000 m layer totals; temperature depth profiles | Month slider, IAP or checkpoint comparisons, map-linked monthly heat series, colored depth/latitude or depth/longitude sections, and selected-cell temperature profiles |

The original six checkpoints comprise mixed training at 50, 500, 2,000 and 8,000
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

Monthly heat content is integrated from the saved temperature profiles. Forecasts
are monthly aggregates of the five-day rollout, compared with IAP monthly
analyses. Values are layer-integrated heat per unit area relative to 0 °C,
displayed in **GJ/m²** (the saved J/m² divided by 10⁹). Only complete model
columns contribute to each layer; missing IAP values remain unavailable.
The extended monthly bundle adds temperature profiles at fourteen depths
(2.5–1,850 m). Click a map to choose a cell, then move the month slider to see
its temperature field evolve. **Along longitude** shows a depth/longitude
color section at the selected latitude; **Along latitude** shows a depth/latitude
section at the selected longitude. These are spatial slices, not zonal or global
averages. Model, comparison and difference sections sit below the maps, use °C,
and share the initialized-interior view's depth masks and layer geometry. The
selected-cell line profile remains below the heat-content time series.
The layer selector controls OHC maps and the heat-content series; temperature
sections and profiles always show the full depth range.
Changing the origin preserves the selected calendar month. Model, IAP and
checkpoint comparison choices apply to both heat and temperature.

All 23 displayed checkpoints have calendar-month temperature profiles and heat
content from the same frozen annual forecast for each case. Every model uses the
same controls; there is no original/regenerated data selector. Profiles are
computed with exact calendar-overlap weights over the 73 five-day predictions.
The service never runs inference. Historical report arrays, replay comparisons,
checkpoint hashes and generation receipts remain in the prepared catalog for
provenance, rather than appearing as competing UI data modes.
Surface and initialized-interior views retain their saved annual-report arrays.
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
`monthly-temperature-provenance/`. Complete profiles for every model before serving the updated viewer.

### Monthly pre-cooldown and OM4 examples

The monthly-diffusion comparison includes paired eight-draw means and single
member-zero decodes for A, E, their longer-training and replay variants, and
**A-replay-pre-cooldown**. The latter is the fixed EMA checkpoint at 2,000 main
updates, before its observation cooldown and EMA reset. The other six models
retain their fixed cooled endpoints. Single draws come from the same decoder
batches as the means; fresh decoder noise at each readout shares a deterministic
latent trajectory and does not constitute a separate stochastic rollout.

`extend_monthly_precool.py` merges audited offline inference into an existing
monthly catalog without changing any prior arrays:

```bash
uv run --project apps/observation_viewer python apps/observation_viewer/extend_monthly_precool.py \
  --base /path/to/monthly-mean-and-member-catalog \
  --export /path/to/pre-cooldown-inference-export \
  --output /path/to/new-monthly-catalog
```

Pre-cooldown A-replay has observation forecasts for 2015/2018/2021 and OM4
forecasts for 2015/2018, including monthly temperatures and heat integrals.
All seven physical checkpoints have OM4-input initialized T/S/U/V examples
for 2015/2018. Choose an `(om4)` forecast origin in the same selector used for
observations. OM4 histories expose only SST/SSH, with OM4's own normalized
three-field forcing; full-depth gold remains held out. The source's native
Julian calendar is retained. The 2021 OM4 case is excluded because its final
history interval ends one day before the forecast begins. No gap is filled.

OM4 initialization reference fields are the final five-day full simulation
state. Observation initialization references remain preceding-December IAP
context. E's OM4-input examples are out-of-distribution probes: E has no
full-field model-data training. Annual OM4 forecasts are saved only for the
pre-cooldown checkpoint, so switching a cooled model from OM4 initialization
to forecast views falls back to an available observation origin while retaining
the selected model. Origin availability is always specific to model and view.

The current full 77-channel decoder readouts retain substantial diffusion
noise, including initialized surface fields for both input sources. Observation
annual native readouts request only SST/SSH; OM4 native readouts request all 77
channels. Monthly tasks likewise request T/S versus all channels. A fixed-latent
crossed-mask diagnostic reproduced the large noise difference, while increasing
sampling from 32 to 128 steps barely helped. The viewer labels these differences;
it retains the original arrays rather than hiding the full-field failure.
Surface-only decoding of OM4 latents reduces speckle but remains biased, so it
is not a validated substitute for the full model task.

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
individual training batches, arbitrary forecast interior time sequences or 3D
volumes. Saved diffusion outputs and six-lead native-quarter interiors are
available where the selected model has those arrays.

## Additional annual comparison models

The October 8 extension adds **17 non-cooldown models**, alongside the original
six exposure checkpoints: U-global/U-multitask, U-omit-patch, U-patch-loss01,
U-aux01 and its static/seasonal/shuffled/anomaly controls, W-global/W-multitask,
A-global/A-multitask, and the four U-multitask-early variants (coarse/fine,
with/without recurrent latent channels). Each has the same three annual origins.
The model selector and map titles retain the literal report names.

The **What is this model?** section at the bottom of the main area updates with
the selected model. It explains architecture, approximate parameter count,
training data/objective, the selected checkpoint's actual exposure, and the
difference between fixed-budget and validation-selected checkpoints.

The added models expose saved 73-step SST/SSH forecasts, the final inferred
initial T/S/U/V state at fourteen depths, and monthly heat content and temperature
profiles. **Own initialized persistence** is available in the monthly view for
all models, using their own initial temperature profiles and consistent heat
integrals. Surface persistence remains available for the added annual models.
The fine modules used during earlier-quarter-degree training are inactive on
observation rollouts.

`export_annual_profiles.py` runs the missing frozen-checkpoint annual forecasts
in the original audited Torch runtime. It checks source, runtime, checkpoint,
input and calendar contracts and records numerical differences from the saved
annual outputs. `complete_monthly_profiles.py` verifies every export, checks that
independent integration of each temperature profile reproduces its heat content,
and builds a new complete catalog while preserving all historical arrays.
The viewer requires profiles for every model before publishing this update.

New initialized-interior examples are observation-input cases only; existing
OM4-input examples remain unchanged.

`extend_annual_models.py` creates a new bundle from a base catalog and a verified
local mirror of annual report outputs. `inventory.json` records exact remote
paths, SHA256 checksums, byte counts, selected checkpoint metadata and origin
files. It rejects cooldown entries, verifies all source/base bytes, exact grids,
depths, masks, references, calendars and checkpoint lineage, converts heat from
J/m² to GJ/m², and verifies every new array by full read-back. All previous arrays
and provenance are preserved. Keep the source mirror and prepared arrays out of Git.

```bash
uv run --project apps/observation_viewer python apps/observation_viewer/extend_annual_models.py \
  --base /path/to/current-bundle --source /path/to/verified-annual-mirror \
  --output /path/to/new-bundle
```

### Diffusion predictions

`extend_diffusion.py --base BUNDLE --evaluation EVAL --output NEW_BUNDLE
--key UNIQUE_KEY --label LABEL` adds a verified completed diffusion evaluation.
It preserves existing arrays and verifies every base checksum, evaluation receipt,
reference field and coordinate. New arrays are fully read back and checksummed.
The surface view offers all 73 five-day means for both the eight-member mean and
one draw. Draw indices are independently sampled at each lead and do not define
coherent stochastic trajectories. Initialization offers all saved mean physical
fields; draw initialization only includes T at 2.5/550 m and S at 550 m, with
unavailable depths/variables explicitly missing. Monthly-profile views only list
models with saved monthly profiles; no fields are fabricated for diffusion.

### Native quarter-degree selected models

The unified model selector also supports **U-global selected (¼°)** and
**Observation-only selected (¼°)**. Each model routes to its own immutable
catalog; comparisons are restricted to models and references on the same grid.
Selecting a model keeps that model selected when the previous view is unavailable,
returns to its surface view and explains the available views.

The native quarter-degree views retain every 720×1440 grid cell. Maps use raster
row strips with the actual latitude cell boundaries, including the irregular
Gaussian latitude spacing; no interpolation or spatial averaging is applied.
They expose 73 five-day surface forecasts, the 19-depth initial T/S/U/V state,
and saved forecast interiors at days 5, 15, 30, 90, 180 and 365. The forecast
interior view provides depth profiles and sections at those six saved leads.
Own initialized persistence holds each model's actual initializer fixed.

Monthly heat content uses its separate native IAP 360×720 grid and GJ/m² units.
Its zero-centered longitude cell is drawn identically on both sides of the
periodic map seam; coordinates and values are neither shifted nor resampled.
Monthly temperature profiles and interior observational truth were not exported
for these models and are not fabricated or inferred from the six saved states.
Existing one-degree models retain their available monthly profiles and all
original diffusion controls.

A new root catalog can register child catalogs without changing its old arrays:

```json
{
  "extra_catalogs": ["native-quarter"],
  "extra_catalog_hashes": {"native-quarter": "SHA256_OF_CHILD_CATALOG"}
}
```

These keys extend the existing schema-version-1 catalog; model IDs must be
unique across catalogs, and supplied child-catalog hashes are verified.
`SAMUDRA_VIEWER_CATALOGS` can provide additional local catalog paths separated
by the platform path separator for preview testing. Prepared arrays remain
outside Git. `prepare_quarter.py` creates the native bundle from verified saved
evaluation files; no model inference is performed by the viewer.

```bash
python prepare_quarter.py --config /path/to/verified-config.json --output /path/to/new-native-bundle
```

The exact extraction configuration is retained in the prepared bundle.


### Monthly latent/diffusion replay comparison

`prepare_monthly_replay.py` builds a separate complete child catalog for **A, E,
A-extend, E-extend, A-replay and E-replay** from frozen annual inference exports.
It requires exact replay of the published surface/OHC/latent outputs, verifies
checkpoint and payload hashes, and independently integrates the retained monthly
temperature profiles to both OHC layers. It does not train or select models.

```bash
uv run --project apps/observation_viewer python apps/observation_viewer/prepare_monthly_replay.py \
  --export /path/to/verified-monthly-export \
  --context-catalog /path/to/existing-verified-viewer-bundle \
  --output /path/to/new-monthly-child-catalog
```

Register the resulting directory through `extra_catalogs` and
`extra_catalog_hashes` in a new immutable root bundle. Existing models remain in
the same selector. Each entry includes annual surface fields, a decoded initial
interior, calendar-month OHC and its exact underlying temperature profiles, plus
its own decoded initialized persistence. All three January starts are retained.

The initial reconstruction uses a separate eight-draw decoder noise stream and
does not change the forecast. Full-field initial decoding is an exploratory
readout: the observation reconstruction objective directly supervises only
surface temperature/height, and E has no full-field model-data supervision.
IAP initial T/S comes from the existing checksum-verified preceding-December
monthly context; it is not instantaneous interior truth and contains no U/V.

Use `tests/test_monthly_replay_browser.py` against an actual server to exercise
all six models' surface, initialization, monthly heat, temperature-profile and
persistence controls over WebSockets.


To add paired single draws, use an export that retains `member_` fields and
certifies every mean field unchanged against the prior complete profile export:

```bash
uv run --project apps/observation_viewer python apps/observation_viewer/prepare_monthly_replay.py \
  --export /path/to/paired-single-draw-export \
  --context-catalog /path/to/existing-verified-viewer-bundle \
  --output /path/to/new-combined-monthly-catalog \
  --readout draw1 --paired-mean-catalog /path/to/prior-six-mean-catalog
```

This preserves the original six mean IDs and arrays, labels them **8-draw mean**,
and adds six **1 draw** entries. Both versions live in the same comparison
catalog, so users can compare their maps and profiles directly. A single draw
is member zero from the very same eight-member decoder batch; each readout
samples fresh decoder noise from the original stream. The latent trajectory is
shared and deterministic, with no sampled physical fields fed back into it.
Monthly top-layer temperature retains the annual operator: calendar-weighted
native SST, using the selected member rather than the mean for single draws.
OHC is independently integrated from each selected readout's own monthly
temperature profiles, and persistence holds its own decoded initial fields.
The published score curves continue to evaluate eight-draw mean fields.
