<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra OM4 V2 + Ocean Heat Corrector

Same model as [`samudra_om4_v2`](../samudra_om4_v2/README.md), with the
revived `OceanHeatCorrector` turned on (see #224/#822, revived on branch
`yongquan/heat-corrector`) and the pre-correction-supervision /
imbalance-penalty fix from
[Chapman, Schreck & Sha (2026)](https://arxiv.org/abs/2607.18416).

Requires a data source with `hfgeou` and `sea_surface_fraction` static
variables -- run the preprocessing pipeline with `--ocean_static_path`
pointing at `ocean_static_no_mask_table.zarr` first (see
`data/ocean_preprocessing`), then point `--experiment.data_root` at that
output.

This is a validation config, not a production baseline: intended for a
small/fast resolution (e.g. twodeg) to check the corrector's imbalance
metric over training, not to reproduce the full-resolution baseline.
