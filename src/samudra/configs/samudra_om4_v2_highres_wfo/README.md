<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra OM4 V2 High-Res + Freshwater Flux (wfo)

Same model as [`samudra_om4_v2_highres`](../samudra_om4_v2_highres/README.md)
(reduced batch size, instance norm, bfloat16, for handling higher spatial
resolutions), trained with freshwater flux (`wfo`) added to the boundary
forcing -- see [#832](https://github.com/m2lines/Samudra/issues/832) and
[#833](https://github.com/m2lines/Samudra/issues/833). Requires the
freshwater-augmented OM4 datasets published under `Samudra/v2026-09/` or later
(point `--experiment.data_root` there); those datasets carry `wfo` but not
`hfds_anomalies`, so this is not a single-variable-controlled comparison
against the `samudra_om4_v2_highres` baseline.

Intended for 1/2-degree and 1/4-degree runs (point `--experiment.data_root` at
the resolution you want). For 1-degree, use
[`samudra_om4_v2_wfo`](../samudra_om4_v2_wfo/README.md) instead.
