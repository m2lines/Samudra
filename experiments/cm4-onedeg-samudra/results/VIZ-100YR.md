<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# CM4 century rollout visualization

Run the standard visualization steps against the completed epoch-40 EMA forecast
from Beta job 214351 and the matching fixed CM4 1° reference. The comparison uses
all 7,298 forecast dates, 0251-01-16 through 0351-01-01, with the 19 source depths.
It includes the standard time series, heat-content and salinity diagnostics,
profiles, distributions, ENSO diagnostics, maps, and four full-record surface
movies. Observation comparisons are unconfigured and their steps skip normally.

`run_viz.py` uses the existing plotting implementation with two input adaptations:
the visualization config currently parses OM4 Julian/noon dates, so the launcher
rebuilds its date-only bounds as CM4 no-leap/midnight dates before selection;
and it selects the five physical forecast fields plus reference masks/areas,
excluding bounds and averaging metadata from physical weighted calculations.
It verifies exact reference/forecast timestamp equality and agreement with the
plotting layout's depth centers/thicknesses. No prediction values are modified.

The CPU-only Grace job requests 64 CPUs, 384 GiB RAM, and at most 12 hours. It
uses the same pinned ARM container as the evaluation with an archived source
overlay. An exit callback plus durable reminder provides completion follow-up.
`progress.json` records completed steps; `COMPLETE.json` is written only after
all configured steps finish, every generated PNG decodes, and all four movies
have the expected 7,298 frames. Figure and movie generation is not evidence of
physical skill; the rollout's salinity extrema remain a known limitation.

Run root:
`/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/viz-100yr-20261022-attempt3`.
Figures are under `cm4-100yr-epoch40-ema/`.

Grace job **144384** was submitted via Alpha on 2026-10-22, using source
[0b72ec219](https://github.com/m2lines/Samudra/commit/0b72ec219).
The archive checksum was verified before extraction; the completion hook was
staged and authenticated before job release. Quiet monitoring and a two-hour
progress check are installed. Results remain pending.

The first allocation, job 144384, failed before Python started after 1m43s:
Enroot attempted NVIDIA driver injection on a CPU-only Grace node. The site
`98-nvidia.sh` explicitly exits without driver setup when
`NVIDIA_VISIBLE_DEVICES=void`; the replacement exports that supported setting
and passes it into the container. It retains the same image, input stores, and
plotting implementation, uses a separate attempt directory, and has an 11h55m
limit to stay within the original 12-hour allocation budget. First-attempt
logs are preserved. Replacement Grace job **144392** was released with source
[5b3b8b98b](https://github.com/m2lines/Samudra/commit/5b3b8b98b).
Its separate completion hook and quiet monitor are installed; the old callback
is retired. Results remain pending.

Job 144392 resolved NVIDIA initialization but failed after 1m38s at the movie-tool
preflight, before Python started. The container lacks an FFmpeg executable on its
PATH. Attempt 3 uses a writable container with user-namespace root mapping
(no elevated host permissions) to install FFmpeg if absent, records package
versions and installation logs, and preserves the pinned Python environment.
Its 11h50m limit plus prior failed allocations remains below 12 hours. All prior
artifacts are retained. Replacement Grace job **144395** uses source
[bc816d33d](https://github.com/m2lines/Samudra/commit/bc816d33d).
Bring-up confirmed FFmpeg/FFprobe installation and entry into the Python
visualization launcher. Installed package versions and logs are retained.
Its callback and quiet monitor are active; both failed-attempt callbacks are
retired. Plotting and final artifact verification remain pending.
