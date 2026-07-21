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

`train_1deg_mse_updates.yaml` is the full-data, single-step promotion config. Its
defaults assume four GPUs and effective global batch 32. Do not submit it until a
candidate passes the two-seed proxy gate and a third fixed proxy seed confirms the
finalist. Independent one-GPU proxy jobs use
`train_1deg_mse_stratified_updates_proxy.yaml` with gradient accumulation overridden
to 16.

`train_1deg_1cell_direct_mse_updates.yaml` pins the promoted one-degree direct-head
baseline. Its defaults assume eight GPUs and effective global batch 32. The paired
proxy and identity diagnostics must establish the direct heads before promotion;
the architecture has no residual or encoder-to-decoder skip path.
