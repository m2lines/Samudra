<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# samudra-multi OM4

These configurations are for the samudra-multi (encoder-processor-decoder) model,
trained on OM4 data. The intent is for this to support heterogeneous
resolutions and modalities.

The `identity_*.yaml` configurations run the fixed-sample identity diagnostic with
`python -m samudra.identity`. They use the production model and Rust loader at one
resolution per job and preserve MSE, spectrum, and patch-seam evidence in the run
output directory.
