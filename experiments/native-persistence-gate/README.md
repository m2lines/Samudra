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

## Candidate-aware gate ablations

Two additional from-scratch runs use the same seed-15 recipe. `direct` applies
one affine 1x1 projection to `[h, latest state, forecast block]`; `nonlinear`
applies a 1x1 projection to 64 channels, GELU, then a 1x1 output projection.
Both finish with sigmoid and the same forecast/persistence blend. All 77 latest
state channels and both 77-channel candidate forecasts are visible to every gate.
There is no target access or gradient detachment. Differences are not appended
because a linear head can already form them from the two candidates.

For the 1-degree model, gate input width is 280 + 77 + 154 = 511. The direct
projection has 78,848 parameters; the nonlinear gate has 42,778 parameters.
The original feature-only gate has 43,274. This compares functional forms,
not exactly matched parameter counts. Final projection weights start at zero
and bias at -6 for every arm; the nonlinear hidden layer starts randomly and
receives gradients after the first output-projection update. Construction
preserves the common backbone/decoder initialization and global RNG stream.
No additional residual head is introduced in these two ablations.

The scripts perform full-sized GPU qualification before training; checkpoint
selection and evaluation stay fixed. Existing original-gate and ungated runs
remain controls. Final results require both new runs and matched diagnostics.
