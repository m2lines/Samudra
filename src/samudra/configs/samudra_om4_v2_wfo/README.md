<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra OM4 V2 + Freshwater Flux (wfo)

Same model as [`samudra_om4_v2`](../samudra_om4_v2/README.md), trained with
freshwater flux (`wfo`) added to the boundary forcing -- see
[#832](https://github.com/m2lines/Samudra/issues/832) and
[#833](https://github.com/m2lines/Samudra/issues/833). Requires the
freshwater-augmented OM4 datasets published under `Samudra/v2026-09/` or later
(point `--experiment.data_root` there). These configs retain all baseline
boundary inputs (`tauuo`, `tauvo`, `hfds`, `hfds_anomalies`) and append `wfo`.
The loader computes `hfds_anomalies` and its normalization statistics from
`hfds` when the anomaly is not stored in the dataset.

Model and training settings match the corresponding baseline preset. For
comparison with existing results, also check data version, resolution and
global batch size (per-GPU batch size × GPU count × gradient accumulation).
Four-boundary-input checkpoints from the earlier WFO config are not
compatible with this five-input preset without explicit weight conversion.

Intended for 1-degree runs. For 1/2-degree and 1/4-degree, use
[`samudra_om4_v2_highres_wfo`](../samudra_om4_v2_highres_wfo/README.md)
instead.
