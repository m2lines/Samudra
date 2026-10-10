<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Cheap continuation wave: diagonal structure and long-rollout instability

Approved scope: six short continuations of the cooled 8k-observation + 8k-OM4
checkpoint, with fixed-budget evaluation and a report before any later wave.
The model evolves a deterministic latent state; diffusion only decodes physical
fields. Consequently, decoded grain cannot feed back into the processor, and
physical-field replay would train the wrong recurrence.

| Arm | Spatial objective | Half of OM4 microbatches |
|---|---|---|
| control | Existing pixel + nearest-neighbor gradient losses | Original 30-day objective |
| multiscale | Add four-direction, multi-scale target-gradient matching | Original 30-day objective |
| replay | Existing losses | Six-step segments of model-generated latent chains |
| multiscale-replay | Multi-scale addition | Same latent replay |
| unroll12 | Existing losses | 60-day differentiable rollout, score last six leads |
| latent-jitter | Existing losses | Perturb encoded latent by 5% of per-channel RMS before original rollout |

All arms start from exactly the same weights, use fresh AdamW moments, effective
batch eight, one paired seed, and 128 updates (64 observation / 64 OM4), with
snapshots at 32/64/128. The learning rate decreases from 1e-5 to 1e-6. This small
LR restart, task mix and optimizer restart apply equally to the control. Results
must be compared against both that continuation control and the untouched parent.
Training and evaluation keep the full-latitude mask, zero-filled missing inputs,
existing normalization, 0.5-correlated diffusion noise, two training members,
32 sampling steps, and eight evaluation members. No retraining from scratch.

The spatial addition compares target and predicted increments in x, y, and both
diagonal directions at lags 1, 2, 4 and 8 cells. Differences are divided by lag
(and sqrt(2) for diagonals), with mean-over-scale weighting and total directional
weight two, then coefficient 0.5. Every traversed grid cell must have valid
supervision. Longitude wraps; latitude does not. These are grid-coordinate
derivatives, not gradients per kilometer. The original gradient term remains,
so this tests an added multi-scale constraint, not a strength-matched replacement.
OM4 uses squared gradient error of the clean denoising estimate; observations
use fair CRPS on member increments. This rewards matching spatial structure
rather than unconditional smoothness.

Replay retains the original objective for four of eight OM4 microbatches. The
other four use separate latent chains, first burned in 30/90/180/300 days without
gradients. Each use backpropagates through the next six steps and scores six
aligned two-field denoising targets, then retains the detached ending latent.
Chains reset after four uses or at 360 days, and a reused state was generated
two optimizer updates earlier (OM4 then observations). The entire possible trajectory must fit
inside the existing training source (1975–2013). Target interiors never enter
the encoder or processor. This is truncated backpropagation with a small online
replay buffer, not full-year backpropagation. It changes lead exposure and drops
initial reconstruction/completion on replay microbatches; report that difference.
The longer-unroll alternative scores the last six steps of a twelve-step graph,
retaining the same forecast denoiser-call count but more processor backpropagation.

Evaluation: use the original nine validation months for member maps, CRPS,
coverage/spread and point/spectral metrics; use the fixed three annual January
starts (2015/2018/2021) for 30/365-day RMSE and member fields. Annual test results
are exploratory diagnostics, not checkpoint selection. Report SST/SSH/T/S member
and mean maps with identical scales, spatial increment errors and directional
power, and the original climatology-normalized four-component RMSE. Compare
latent norms and error-versus-lead where possible. Lower error with collapsed
member variability or over-smoothing is not sufficient evidence of success.
Save and analyze early checkpoints if a run fails; do not relabel a partial run
as a completed 128-update comparison.

Also run three held-out OM4 year-long trajectories, decoding eight members at
0/30/90/180/365 days and recording latent RMS at every step. Repeat these OM4
probes for the untouched parent on the same Beta data/runtime. This separates
model-world stability changes from transfer to observation initialization.

Compute: prefer one Beta production node, four independent GPU workers, scheduling
six arms as workers finish. Initial ceiling is one hour on four GPUs for smoke
qualification plus sixteen production hours on four GPUs (68 allocated GPU-hours
maximum), including evaluation. Queue estimates are provisional. Source archives,
Slurm IDs, checkpoint hashes, data checks and W&B URLs live in the operational
manifest under `outputs/diffusion-interventions-v1/` and on project storage.

Related evidence informing this wave:

- [Weidong's methods note](https://docs.google.com/document/d/1rNUAMYNeuKvyFAF4vRPETN12DnRe_MalAcYQlUYA298/edit)
  distinguishes CRPS calibration, memberwise variability and forecast error;
  low-error stochastic-interpolant runs substantially underrepresent temporal
  variance. This motivates retaining spread and variability diagnostics.
- [The longer-unroll/replay discussion](https://openathena.slack.com/archives/C08CYM42DT3/p1791317695835179)
  and [subsequent annual-rollout observation](https://openathena.slack.com/archives/C08CYM42DT3/p1791418935575869)
  motivate the twelve-step arm. These are preliminary results from a different
  deterministic architecture, not proof of benefit for this latent model.
- [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376)
  introduces training on model-generated inputs with truncated gradients to
  address rollout distribution shift. Our latent replay adapts that rationale.
- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)
  uses perturbed training inputs to reduce accumulated error. The 5% latent
  perturbation is our heuristic adaptation, not the paper's parameterization.
- [Sobolev Training for Operator Learning](https://arxiv.org/abs/2402.09084)
  motivates derivative supervision; it does not establish that these particular
  diagonal artifacts arise from missing derivative constraints.
- [PDE-Refiner](https://arxiv.org/abs/2308.05732) motivates checking fidelity across
  frequencies. Its denoising participates in physical evolution, unlike this
  model's readout-only diffusion, so it is not a drop-in stability remedy here.

Qualification: Beta job 211727 passed both an OM4 and an observation update
for all six arms, with finite gradients and checkpoint writes. The original
256-update proposal was reduced to 128 **before production** because measured
observation updates took 279–440 seconds (including first-call overhead). This
preserves all six comparisons and room for evaluation inside the same compute
ceiling. These smoke losses are not evidence of scientific improvement.
