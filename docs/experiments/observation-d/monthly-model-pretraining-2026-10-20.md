<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Model-only pretraining, then observation adaptation

Question: do the model datasets and multitask initializer help when observations
are introduced only during the final adaptation stage, rather than jointly
throughout training?

Status: **26 targeted tests and repository checks passed**. GPU qualification
**215045 is running** on Beta; its compute-node callback check passed. Production
waits for all four arms to qualify. Pinned producer: `a1e74e188`. Run root:
`beta:/projects/ny/lz1955/multiscale/jrusak/runs/monthly-pretraining-20261020`.
No new scientific results yet.

This is a separate four-run comparison alongside the
[longer-training and replay wave](monthly-replay-2026-10-20.md). No replay or
latent-size regularizer is introduced here.

## Runs and data exposure

| Run name | Arm | First 2k optimizer updates | Initializer tasks during pretraining | Final 1k updates |
|---|---|---|---|---|
| P-OM4-surface | A | 100% OM4 | Surface only | 100% obs, surface inputs |
| P-OM4-multitask | B | 100% OM4 | 50% surface, 25% missing variable family, 25% spatial gap | 100% obs, surface inputs |
| P-mix-surface | C | 50% OM4 / 50% CM4 | Surface only | 100% obs, surface inputs |
| P-mix-multitask | D | 50% OM4 / 50% CM4 | 50% surface, 25% missing variable family, 25% spatial gap | 100% obs, surface inputs |

All four start from the same random weights and seed 1729. An update accumulates
eight examples, matching the [original A–E pilots](monthly-latent-pilots-2026-10-16.md):
16k model examples, followed by 8k observation examples. A/B receive 2k OM4
updates; C/D receive 1k OM4 and 1k CM4 updates. **No observation training updates
occur before update 2001.** Fixed observation validation during pretraining
does not perform optimizer updates; its scores are diagnostic only. Final
checkpoint selection is restricted to the observation cooldown.

This reuses the original A–D matrix with observations removed from main
training. It does not include a CM4-only arm. Relative to those original joint
pilots, the total update budget is matched, but the number of model updates is
doubled and observation updates are halved. This tests the practical allocation
and sequencing recipe; it cannot attribute a difference solely to ordering.

## Model, objective and sources

Use the same 48,702,016-parameter architecture: separate masked physical-history
initializer, deterministic two-slot latent processor and physical diffusion
decoder. Global 1° fields are 180×360; latent slots are 128×45×90 each. Histories
contain nineteen five-day bins. Multitask changes only the initializer's
available input fields; the processor still evolves a full latent state.
Pretraining target fields and forecast windows are the same across tasks.

Model loss remains 0.5 full-field monthly denoising + 0.5 full-field native
five-day denoising + 0.1 initial reconstruction. The observation cooldown retains
0.8 monthly temperature/salinity denoising + 0.2 future SST/SSH denoising + 0.1
initial surface reconstruction, including interior supervision but no explicit
OHC-integral loss. No soft latent bound, latent norm penalty or architecture
change is added.

OM4 and CM4 retain their original native calendars, training cutoffs, masks and
three own-model forcings (`tauuo`, `tauvo`, `hfds`). CM4 retains **OM4
normalization**, including forcing scales. Sources on Beta are:

- OM4: `/projects/ny/lz1955/multiscale/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3`,
  with the original byte-verified packed physical fields.
- CM4: `/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg/CM4.zarr`,
  the previously transferred and fully read-back-verified store.
- Obs: `/projects/ny/lz1955/multiscale/jrusak/data/obs_full_range/d-observation-pilot/samples`.

The learned eight-variable ERA5 adapter is unused during model-only pretraining
and first learns during the observation cooldown. The model datasets use their
native forcings directly, as in the original pilots. Existing observation/OM4
normalization and data splits remain fixed for comparability.

## Schedule, validation and execution

AdamW, clipping at 1, main LR 1e-4, EMA half-life 100 updates. At 2k, retain Adam
moments, reset EMA to raw weights and linearly decay LR from 1e-4 to 1e-6 over
the 1k observation-only cooldown. No optimizer, decoder, initializer or forcing
adapter is reinitialized at the transition.

Keep nine-origin monthly validation every 250 updates, eight diffusion members,
32 sampler steps and the existing five-component raw-RMSE/climatology blend.
Report surface-only, upper/deep OHC and raw/EMA endpoint metrics alongside the
full blend. Save uncooled 2k and cooled 3k checkpoints. Compare with the original
joint A–D and obs-only E at 3k, explicitly reporting source-update counts. Repeat
the same fixed-endpoint three-origin annual rollout diagnostics to check whether
short-lead improvements survive through day 365; do not select on annual tests.

All four qualifications must pass before production: original source and
normalization hashes, every applicable initializer task on each model source,
finite observation adaptation, exact model/EMA/Adam/RNG serialization across an
observation update, and real diffusion evaluation. Qualification explicitly
checks that model training supplies no ERA5-adapter gradients and that observation
adaptation does. Diagnostic weights never initialize production.

Use one four-GPU Beta node for the four production processes, with online W&B
and `async-job-followup` quiet monitoring. Queue estimates are advisory; waiting
is acceptable. Retain producer hashes, qualification artifacts and GPU accounting.
