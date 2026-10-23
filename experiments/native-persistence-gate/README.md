<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native feature-conditioned persistence gate

Experiment/results branch, not intended to merge. Train one 1-degree Samudra
from scratch. Existing control: Beta job 210664, W&B ocean_emulators/default/uaei2vbr,
source a20c444849ab39575089f8e653bb128dd2b473a6. No new control run.

A 1x1 projection of the final full-resolution U-Net features produces independent
sigmoid gates for all 77 channels and two leads. Let p be the ordinary Samudra
3x3 decoder forecast and x the latest input state, repeated for both leads:

    prediction = p + sigmoid(G(h)) * (x - p)

All parameters train jointly. The gate starts at sigmoid(-6)=0.002473. Existing
parameter initialization and the subsequent RNG stream are preserved. There is
no frozen backbone or separate postprocessor. Unlike the earlier correction pilot,
this native blend has no additional residual head: the trainable decoder p supplies
the unconstrained learned forecast. The gate blends this with persistence.
Land masking follows blending. `pred_residuals=True` is rejected with this gate to
prevent adding history twice. Default gate=False preserves the original path.

Matched recipe: samudra_om4, Rust loader with 8 concurrent reads and device prefetch,
tau_hfds (no hfds anomalies), InstanceNorm, bfloat16 backbone/float32 output,
stochastic depth off, seed15, four GPUs x batch4 x accumulation2 = effective32,
70 epochs, Adam LR0.0006 cosine, capped dynamic MSE limit20, four model calls per
training example (40 days), same source/splits/normalization and EMA implementation.
Selection uses 360-day validation RMSE. Dynamic training losses are not comparable
between models; compare fixed validation/rollout metrics and physical errors.

Use the same pinned Beta container/native-loader build and migrated project data
as the historical control. Configuration audit allows only gate, name, W&B project,
and post-evaluation path differences. Test gate routing for 1/2 output frames,
masking, paired initialization/RNG, feedback gradients and optimizer reload.
GPU qualification tests full-sized 180x360 two-call gradients before production.

Completion criteria: successful 70 epochs and configured post-training evaluation;
compare against existing control on fixed 5/10-day physical RMSE and texture plus
360-day rollout metrics. Report single-seed uncertainty and any runtime/protocol
mismatch. Do not declare success from training loss or short-range maps alone.
