<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra OM4

These configurations are our current WIP Samudra-style (UNet) model,
trained on OM4 data.

The current preset uses instance normalization, bfloat16, and dynamic MSE loss
with a maximum channel-weight ratio of 20. Current Samudra and mini use
360-day (roughly one-year) rollout validation each epoch and select the best
checkpoint by rollout RMSE. Current Samudra, mini, and multi run post-training
evaluation over saved periodic checkpoints and the final EMA checkpoint.
Multi retains one-step-loss checkpoint selection because multi-source rollout
validation is not supported.

These three training presets use the Rust loader for local OM4 stores; install
it with `uv sync --extra rust`. Current Samudra and mini use the shared
`data/om4_rust.yaml` config with `tau_hfds` forcing (no derived heat-flux anomaly
channel), including during evaluation. This changes the input channel count,
so checkpoints trained with `tau_hfds_hfds_anom` need their original data config.
The historical v1, v2, and v2_highres presets retain their existing settings.
