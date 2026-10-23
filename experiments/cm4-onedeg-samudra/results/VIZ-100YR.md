<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# CM4 century rollout visualization

**Completed:** CPU-only Grace job **144455** and all Slurm steps exited `0:0`
in **7h53m34s**, with peak resident memory about **84.8 GiB**. The exact executed
source is [2b988487d](https://github.com/m2lines/Samudra/commit/2b988487d63e9ae69a8135789eab14626251b874).
Across the four allocated attempts, runtime totaled **9h01m06s**, below the
original 12-hour budget; no GPUs were allocated.

The pipeline compared the completed epoch-40 EMA forecast from Beta job 214351
against the fixed CM4 1° reference, over all **7,298 forecast dates** from
**0251-01-16 through 0351-01-01**, at the 19 source depths. All **25 substantive
steps** completed: time series, heat-content and salinity diagnostics, profiles,
distributions, ENSO diagnostics, maps, and movies. Five observation steps skipped
explicitly because no observational comparison was configured.

## Outputs and verification

- **107 PNG figures**, totaling 218,555,680 bytes. Every PNG passed the launcher's
  integrity check and an independent full pixel decode with `PIL.Image.load()`.
- **Four 1920×1080 MP4s**, each with **7,298 frames** and duration **486.533333 s**
  (8m06.5s), verified using FFprobe. Total movie bytes: 6,437,745,025.
- Three CSV tables: global heat-content/salinity slopes, depthwise heat-content
  slopes, and basin heat-content slopes.
- Exact no-leap forecast/reference timestamp equality, source depth centers, and
  layer thicknesses passed before processing. `COMPLETE.json` records the final
  dates, steps, PNG count, and movie metadata.

The movie preset is designed for two forecast runs. With this single configured
run, it repeats the same forecast in both prediction panels; these are not two
independent models. Temperature and salinity each have a surface-state movie
and a bias movie.

A diagnostic limitation is visible in the global heat-content figure: the
forecast does not reproduce the reference's upward trend over this interval.
The pipeline's fitted slope is 0.05547935 ZJ per five-day sample for CM4 and
-0.01994271 for the forecast, a ratio of -0.35946. These are plot diagnostics,
not a new checkpoint-selection or forecast-skill metric. The previously measured
negative/extreme local salinity values remain a physical limitation; successful
rendering does not establish physical realism.

## Artifact locations

Run root:
`/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/viz-100yr-20261022-attempt4`.

Output directory: `cm4-100yr-epoch40-ema/`, with `Temperature/`, `Salinity/`,
`Timeseries/`, `OHC/`, `PDFs/`, `ENSO/`, `Metrics/`, and `Movies/` subdirectories.
The four movie filenames are:

- `Movies/thetao_surface_map_movie.mp4`
- `Movies/thetao_surface_map_bias_movie.mp4`
- `Movies/so_surface_map_movie.mp4`
- `Movies/so_surface_map_bias_movie.mp4`

The run root also contains `figures-and-tables.tar.gz`, containing the figures,
tables, config, logs, and completion audit, with movies excluded. Its checksum
is recorded in `figures-and-tables.sha256`. A local copy was downloaded and
verified against that checksum for access from the originating conversation.
Source/config archives, hashes, allocation, runtime package versions, and logs
are retained. Completion callbacks and reminders have been retired.

## Input adaptations and runtime

`run_viz.py` calls the existing plotting implementation with two input adaptations:
the visualization config currently parses OM4 Julian/noon dates, so the launcher
rebuilds its date-only bounds as CM4 no-leap/midnight dates before selection;
and it selects the five physical forecast fields plus reference masks/areas,
excluding bounds and averaging metadata from physical weighted calculations.
It verifies agreement with the plotting layout's source depths/thicknesses.
No prediction values or plotting algorithms were modified.

The job used the same pinned ARM container as evaluation, with an archived
source overlay, 64 CPUs, and 384 GiB requested memory. On CPU-only Grace,
`NVIDIA_VISIBLE_DEVICES=void` disables unnecessary driver injection. A writable,
user-namespace container layer installed FFmpeg and matching `python3-tk`;
package versions were recorded, and both delayed `xarrayutils.plotting` imports
were checked before the large data reductions.

Earlier attempts and their artifacts are preserved:

| Job | Outcome | Allocation runtime |
| --- | --- | --- |
| 144384 | NVIDIA driver injection failed before Python on CPU-only Grace | 1m43s |
| 144392 | Missing FFmpeg preflight failed before Python | 1m38s |
| 144395 | 11 steps/87 figures completed, then delayed Tkinter import failed | 1h04m11s |
| 144455 | Full pipeline and artifact audits passed | 7h53m34s |

The final attempt restarted all steps in a separate directory because later
steps consume values held in the earlier steps' process memory. The previous
process state was unavailable after failure.
