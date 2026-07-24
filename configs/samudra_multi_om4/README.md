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

`identity_cross_1_halfdeg.yaml` exercises one shared encoder/decoder on the four
same- and cross-resolution routes between the one- and half-degree products. It
uses current-timestamp destination labels and balances its fixed samples equally
across routes; the nominal 40 epochs correspond to 1,280 optimizer updates on one
GPU, so adjust epochs when changing data-parallel world size.

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

The iterable-inverse proxy and full-data configs set
`train_processor_depths: [1, 2, 4]`, `steps: [4]`, and a separate boundary
encoder. The prognostic state is encoded once. A batch selected at depth N then
calls the boundary encoder and shared latent processor N times with one aligned
forcing state per call, and is supervised against the physical `t+N` label. The
decoder output is never fed back through the state encoder. Validation reports the
configured physical leads, currently `[1, 2, 4]`, plus lead-matched persistence
and forcing ablations. There is no legacy decode/re-encode training path. The depth
list is an initial experiment range, not a permanent limit on the supported
contract.
