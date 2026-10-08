<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Additional paired seeds on Torch

Seeds 16 and 17 each compare control with constant stochastic depth 0.1. The
training source remains the original merged experiment source; only seed,
platform, per-GPU batch size, names, and runtime paths change.

Each run uses one eight-RTX6000 host, 128 CPUs, batch size 2 per GPU and gradient
accumulation 2: effective batch 32, matching the initial four-GPU batch-4 pair.
The September 2026 1° OM4 release is staged separately from Torch's older v3
copy and verified before submission. Rust loading and tau_hfds are preserved.

Use the x86 PhysicsNeMo image for original main revision 0b53202488d18724d5cf7d306230fb72c2779528,
with a checksum-verified code overlay built from this experiment branch. The
overlay builder must verify its dependency files against the container.

Final epoch-70 and EMA rollouts and OM4 metrics run after training. Observation
metrics are a separate CPU follow-up, with the original wet masks restored as
in the first pair. No visualization is requested for this wave. Results belong
on PR #915; hardware/world-size differences must be retained in provenance when
comparing against seed 15 on Beta.
