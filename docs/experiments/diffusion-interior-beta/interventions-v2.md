<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Alternative spatial and stability objectives: two-day campaign

Authorized October 10: spend one to two days testing alternative remedies for
visible diagonal artifacts and annual divergence, following the
[first continuation pilot](interventions-v1-results.md). This campaign has a
192 allocated GPU-hour ceiling including qualification, failures and evaluation.
Prefer Beta. Target a report within 48 hours; queue delays will be reported.
No colleague reservation or unrelated job will be used or displaced.

Eight independent arms start from the exact cooled 8,000-observation + 8,000-OM4
parent, not from selected first-pilot winners. Each receives 256 fresh-AdamW
updates (128 observations and 128 OM4), effective batch eight, paired seed
271829, cosine LR 1e-5 to 1e-6. Snapshots: 32/64/128/256. The original 128-update
pilot is context, not the matched optimization control: its LR schedule differs.

| Arm | Intervention |
|---|---|
| control | Original objectives, longer matched continuation |
| replay | First-pilot latent replay, for the longer budget |
| replay-bound | Replay plus a soft penalty on excessive per-channel latent RMS |
| pushforward | Fresh model-generated 30/90/180/300-day states each use, without reusing a buffer |
| curvature | Add target second-difference supervision in four directions |
| block | Add scores of spatial averages over 3/7/15-cell squares |
| curvature-replay | Curvature supervision plus replay |
| block-replay | Spatial-average supervision plus replay |

Curvature uses offsets 1/2/4 cells along x, y and both diagonals, dividing second
differences by lag squared. These are grid-coordinate features, not physical
curvatures per kilometer. Block features use unit-mass square averaging kernels.
Both preserve periodic longitude and exclude any stencil crossing land, a
missing target or a latitude boundary. Features are averaged and added with
coefficient 2 to the original loss, which remains unchanged. Empty-support
features contribute zero. OM4 compares features of the predicted clean denoised
field to truth using channel-balanced MSE with the original noise weighting;
observations use fair CRPS of member features. These are target-matching losses,
not unconditional smoothing. Block scores may improve large-scale coherence
without removing diagonal texture; that outcome would reject this intervention
as an artifact remedy. Neither intervention directly constrains all joint
spatial dependencies, even though it supplements pointwise supervision.

The bound adds 0.1 times the mean squared excess of per-channel spatial latent
RMS above twice the corresponding encoded-origin RMS (floor 0.1). Reference
scales are detached and retained with each replay chain. This is a soft training
penalty on all six differentiable future states; there is no inference clipping.
Its arbitrary scale and pressure on latent representation are deliberate pilot
limitations. Smaller latent norms alone are not success: physical error must
improve without sacrificing short-lead skill.

Replay retains four original OM4 microbatches and replaces four with six-step
segments from four detached latent chains. Fresh pushforward shares the initial
burn-in leads but resets every use, so its lead distribution differs from reused
chains as well as its freshness. All trajectories and targets stay inside the
original training source. Replayed/pushforward samples omit the initial
reconstruction/completion objective; this known first-pilot confound remains.
No target interior is fed into the encoder or processor. Observation training
is unchanged except for the explicitly specified spatial-feature additions.

Fixed data/model: full-latitude observation mask, zero-filled missing inputs,
existing normalization, ERA5/OM4 forcing, same architecture, 0.5-correlated
noise, two training draws, 32 denoising steps. Full training must log to online
W&B group `diffusion-interventions-v2`. All eight scientific runs have one seed.

Evaluation uses the same nine validation months and 2015/2018/2021 annual starts,
eight draws and 32 steps. Evaluate the prespecified final 256 checkpoint for all
arms; do not pick a checkpoint using annual test error. Preserve early weights
for diagnostics/recovery. Report 30/365-day climatology-normalized four-component
RMSE, per-origin/component errors, native OM4 latent growth and physical error,
member and mean SSH/SST/T/S maps, directional structure, target-increment errors,
scored regional spectra, CRPS, spread/RMSE and coverage. Compare individual
members as well as ensemble means. Any sharper-looking or smoother field must
be assessed against truth and calibration. Annual tests are now repeatedly
consulted exploratory cases, not a fresh confirmatory holdout.

Execution: isolate outputs under Beta project `runs/diffusion-interventions-v2`;
reuse verified read-only inputs via links to v1. A short eight-arm GPU smoke test
must pass before a single four-B200 production allocation starts. Reserve up to
40 production hours (160 allocated GPU-hours), leaving budget for qualification
and routine failure recovery. Register durable completion callbacks and hourly
or two-hour fallback checks. After the fixed comparison, synthesize results and
update draft PR #895. Any extra work must fit both the time target and remaining
192-hour allocation ceiling; no open-ended extension is authorized by this plan.

Related rationale: [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376)
uses model-generated inputs and truncated gradients to address rollout
instability. Our fresh-pushforward arm adapts that idea to deterministic latent
evolution. [Scheuerer and Hamill](https://repository.library.noaa.gov/view/noaa/22327/)
show why multivariate verification needs dependence-sensitive scores. Our
linear-feature CRPS is a simpler diagnostic/training intervention, not their
variogram score and not a claim to resolve joint calibration. The earlier
[related-work discussion](interventions-v1.md) remains relevant.

Initial submissions: GPU qualification **211872** (`beta_test`, up to one hour,
four B200s), production **211873** (`beta`, after successful qualification,
up to 40 hours, four B200s). Immutable runtime source:
`interventions-v2-13ef3dcb046b`. Operational state and full file fingerprints:
`outputs/diffusion-interventions-v2/manifest.json` locally. The production
scheduler initially estimated October 10 at 21:20 UTC; this is not a guarantee.
Completion callbacks and one-/two-hour fallback reminders are registered for
both jobs. Twenty-one focused CPU tests passed before submission.

The report generator accepts `--updates 256 --arms control replay replay-bound
pushforward curvature block curvature-replay block-replay`; the first-pilot
128-update contract remains its default. Add explicit spatial-sector diagnostics
and spatial-arm spectrum panels to the final v2 report rather than relying only
on the default first-four-arm spectral panel.

## Checkpoint learning curves (requested during the run)

Add identical annual evaluations of every arm at 32/64/128 added updates; reuse
the already-scheduled final 256-update evaluations and common untouched parent.
The x-axis shows both total added updates and their half-sized OM4/observation
counts. These are checkpoints along one 256-update cosine-LR schedule, not
independently optimized training budgets. Do not splice in v1's 128-update
endpoint as if it came from this schedule.

The extra 24 evaluations keep three origins, eight members, 32 sampling steps,
full-latitude support and unchanged metric definitions. Completed v1 annual
runs took about 30.4 minutes each: roughly 12 GPU-hours expected for this sweep.
Queue one four-B200 evaluation allocation after production, capped at seven
hours (28 allocated GPU-hours). Combined reservation ceilings are 160 production
+ 28 learning-curve evaluation + the actual 0.999 smoke GPU-hours, under 192.
Any recovery must respect the remaining campaign budget.

Use `scripts/report_diffusion_learning_curves.py` for 30/365-day blended curves,
physical component tables, per-origin results and SST-error-versus-forecast-lead
panels at each checkpoint. Distinguish delaying divergence from reducing errors
throughout the year. Report marginal improvement per doubling. Arithmetic
512/1024-update scenarios repeat the observed 128→256 fractional gain; also show
no-further-improvement. They are conditional scenarios, not convergence claims,
confidence intervals, or authorization for larger training runs. Repeated annual
cases remain exploratory. The current training processes and budgets are unchanged.

## Clarified stability objective and variable-timestep follow-up

The immediate goal is comparable degradation over a year to the deterministic
reference, not matching or beating its absolute RMSE: architecture and other
hyperparameters differ. Report absolute errors together with excess error above
day 30 and growth relative to day 30, per component and origin. Ratios alone can
reward a poor short-lead forecast, so retain short-lead skill and all absolute
values. Distinguish a stable but inaccurate trajectory from catastrophic growth.

Assess physical plausibility separately from agreement with the realized future:
compare early and late individual-member fields using fixed color scales, field
means and spatial variability, extremes, directional spectra and increments.
Account for seasonality using the corresponding reference distributions; do not
require January and December to have identical statistics. Include ensemble-mean
results and calibration alongside members. Independently sampled readouts do not
define coherent member trajectories, so temporal increments also require an
explicit sampling convention. These diagnostics should reveal whether late
fields look implausible without needing a pointwise truth overlay.

None of the eight submitted arms changes the five-day ocean timestep. A proposed
follow-up is supervised 5/10-day evolution with an additional composition loss:
two five-day advances should agree with one ten-day advance under the same
ordered forcing history. Five and ten days use existing OM4 targets without
inventing intermediate 2.5-day truth. The user approved a small follow-up;
implementation and execution details are in [the timestep pilot](timestep-pilot.md).

The current processor shifts two latent slots, and the decoder reads both.
Simply supplying a different timestep would leave slots at different times:
two five-day calls finish with [t+5,t+10], whereas a naive ten-day call would
produce [t,t+10]. As the user clarified, only the newest day-ten slot needs to
match; the older slots should not be compared. The implemented extension shares
the dynamics weights, with a new ordered-forcing/interval conditioning adapter.
Physical auxiliary supervision uses the short-path day-five slot alongside the
direct day-ten slot so the existing decoder still sees five-day spacing.
Consistency gradients reach both paths and the original five-day processor.
This tests composition regularization, not arbitrary continuous timesteps.

Suggested matched arms: ordinary five-day continuation; joint supervised
five-/ten-day training without consistency; the same joint training with latent
composition consistency. Keep replay, data, optimizer schedule and evaluation
fixed across arms, and report both examples and GPU cost since compute differs.
Evaluate all arms with the ordinary five-day rollout to establish benefit to the
existing task. Compare deterministic latents rather than independent diffusion
draws; sampled physical outputs need a coupled-noise or distributional comparison.

[Deep-OSG](https://arxiv.org/abs/2302.03358) studies variable-step semigroup-aware
learning for autonomous systems. Our forced system needs matching forcing paths
and calendar times, not an autonomous semigroup assumption. Composition alone
does not guarantee stability: even exponentially growing dynamics compose
exactly. Physical supervision and annual plausibility diagnostics remain essential.

## Directional report diagnostic

Use `scripts/report_diffusion_directional.py --root <evaluation-directory>
--arms <arm-names> --output <output-directory>` alongside the scored radial
spectra. It checks the nine validation origins, coordinates, masks, references
and NPZ fingerprints, and rejects incomplete regional ocean boxes. All eight
members are included. The same planar detrending and periodic Hann taper are
used, but this diagnostic uses grid-cell distances and a full two-dimensional
Fourier transform, not the physical-distance bins used in the blended score.

Report power in 2-4, 4-8 and 8-16-cell wavelength bands, excluding frequencies
outside the radial Nyquist circle. Also report the power-weighted mean of
`cos(4 theta)`, where theta is the Fourier wavevector angle in image-grid
coordinates. Positive coefficients favor grid axes; negative coefficients favor
diagonals. Compare against the reference coefficient in each band and region:
a negative difference means relatively more diagonal power than the reference.
Pool raw powers and angular moments across cases before dividing, separately
for ensemble-mean fields and the average of individual-member powers. Zero-power
orientation is undefined, not isotropic. The CSV retains absolute power and
coefficients alongside reference ratios/differences.

This is an exploratory texture diagnostic, not an additional selection score
or a unique artifact detector. Ocean structure, finite Fourier lattices and
tapering also affect angular power; a reference difference alone does not prove
an artifact. Interpret it together with individual-member maps, reference
gradients, radial spectra, calibration and annual stability.
