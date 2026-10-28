# Final EMA ensemble members and CRPS maps

All eight B/C members are shown alongside their means and deterministic A.
The example is the January 2014 forecast: initialization corresponds to the
December 27–31, 2013 five-day mean; day 30 is January 26–30, 2014. These are
saved predictions from the completed runs, with no additional inference.

## Member fields

| Variable | Initialization | Day 30 |
|---|---|---|
| SST | [Map](results/maps/sst-initial.png) | [Map](results/maps/sst-day30.png) |
| SSH | [Map](results/maps/ssh-initial.png) | [Map](results/maps/ssh-day30.png) |
| Interior temperature, level 9 | [Map](results/maps/thetao9-initial.png) | [Map](results/maps/thetao9-day30.png) |
| Interior salinity, level 9 | [Map](results/maps/so9-initial.png) | [Map](results/maps/so9-day30.png) |

The top row contains observations and A; the second row contains B/C means;
the remaining rows contain members 1–8. Interior instantaneous references are
not available and are explicitly marked absent. Monthly interior targets would
not be valid instantaneous gold fields here.

Within each model, member indices follow the same forecast realization from
initialization through day 30. Between B and C, equal member indices should not
be interpreted as matched physical realizations.

Panels use shared limits per variable/time, native cells with nearest-neighbor
rendering, and no smoothing. Display limits use pooled 1st/99th percentiles;
colorbar arrows indicate clipped tails. Exact limits are in
[display-limits.json](results/maps/display-limits.json). Initialization and day-30
figures have independently chosen limits, so use colorbar values for time comparisons.

## Pointwise fair CRPS terms at day 30

| Variable | January example | All nine validation origins |
|---|---|---|
| SST | [Terms](results/maps/crps-sst-2014-01.png) | [Terms](results/maps/crps-sst-nine-origins.png) |
| SSH | [Terms](results/maps/crps-ssh-2014-01.png) | [Terms](results/maps/crps-ssh-nine-origins.png) |

Columns show mean member absolute error, the positive spread subtraction, and
fair CRPS (first column minus second). Rows show A, B, C and C−B. In the final
column's C−B map, red means C is worse and blue means C is better. In the spread
column, red instead means C has a larger spread subtraction; it is not by itself
an indication of worse forecasts. Each term has its own color scale, shared
across A/B/C. Differences have separate symmetric scales.

The nine-origin maps average scores at each cell over available reference
origins. They do not apply latitude weights to pixel values. Applying the
cosine-latitude and valid-origin weights reproduces all six aggregate fair CRPS
scores from the report (checked numerically). Missing support is gray. Display
tails are clipped, while the saved arrays retain full values.

Exact spatial arrays: [January](results/maps/crps-terms-2014-01.npz),
[nine origins](results/maps/crps-terms-nine-origins.npz). Arrays A/B/C have axes
(term, variable, latitude, longitude), with terms ordered error/spread/CRPS,
variables SST/SSH, and units °C/cm. `count` records valid origin counts.

[Scores and experiment report](results.md). Reproduce with
`scripts/plot_stochastic_member_maps.py` and the saved final EMA member files.
