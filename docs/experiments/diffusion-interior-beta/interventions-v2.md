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
