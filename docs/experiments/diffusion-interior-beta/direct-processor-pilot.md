<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Direct next-latent prediction pilot

**Status:** complete. Three arms finished their 128-update continuations and all
forecast evaluations; direct without warm-up failed at mixed update 3.
See the [final report, maps and numerical results](direct-processor-results.md).
The warmed direct conversion fails on observational annual rollouts, whereas
further residual replay-bound training improves annual error from 6.28 to 3.80.
Total execution used 28.11 allocated GPU-hours; monitoring is retired.

User-authorized follow-up to the [latent-amplitude diagnostics](coastal-decoder-results.md).
Test whether replacing the persistent residual update
`next = latest + processor(history, forcing, context)` with
`next = processor(history, forcing, context)` reduces annual drift while
retaining short-lead skill. History still shifts to `(latest, next)`. The
processor's internal residual blocks remain unchanged; there is no outer
identity bypass or implicit addition in the direct variant. Direct prediction
does not mathematically guarantee stable recurrence.

The existing matched deterministic 8k/8k reference already predicts the next
output state directly. Its versioned
[evolution](https://github.com/m2lines/Samudra/blob/4c058b3ec144a646589bb4b833f74f8f654f06f4/src/samudra/experiments/surface_state.py)
returns the network output times the mask, and its
[forecast loop](https://github.com/m2lines/Samudra/blob/4c058b3ec144a646589bb4b833f74f8f654f06f4/src/samudra/experiments/observation_model.py)
shifts that prediction into the state without adding the prior state. Internal
U-Net residual blocks are a separate detail. This is a relevant difference from
the diffusion model, but not a controlled explanation of their stability gap.

## Four matched small continuations

All arms start from the previous **replay-bound** checkpoint at 8,128 OM4 and
8,128 observation updates (global step 16,256). Hold the existing data, global
latitude loss mask, effective batch eight, correlated diffusion noise, 32-step
sampler, replay and soft latent bound fixed. Do not introduce inference RMS
clipping or GroupNorm changes in this experiment.

| Arm | Persistent update | Processor warm-up |
|---|---|---|
| residual | latest plus predicted delta | None |
| direct | predicted next state | None; immediate removal of addition |
| residual-warm | latest plus predicted delta | 256 processor-only teacher updates |
| direct-warm | predicted next state | Same 256 processor-only teacher updates |

Immediately removing the addition reinterprets the old delta prediction as a
complete state, so this can cause transfer failure independent of long-term
stability. Warm-up trains the processor to reproduce the frozen parent's
short-step next latent on training-era OM4 examples. It scores only the newly
predicted slot, not the trivially shifted history slot. The encoder and diffusion
decoder remain frozen during warm-up. No long, unstable teacher rollout or
scientific validation/test target is used. The residual-warm arm is a procedural
control: its initial teacher loss should be near zero.

Warm-up uses a separate cosine learning rate from 1e-3 to 1e-5 and zero weight
decay. Record pre/post relative next-latent MSE and short autoregressive agreement
on distinct examples drawn from the training era. If the direct warm-up does
not recover that mapping, a negative continuation result remains inconclusive
about direct prediction as an architecture.

Afterward all arms use fresh AdamW optimizers, identical paired data/noise seeds,
and **128 mixed updates: 64 OM4 + 64 observation**, cosine learning rate 1e-5 to
1e-6. Replay plus the existing soft amplitude penalty is retained in every arm.
Final physical-supervision counts are **8,192 OM4 + 8,192 observation**; warm-up
exposure and compute are reported separately. Retain checkpoints at 32/64/128
added updates. Selection is the fixed final budget, not the best annual test score.

## Evaluation and interpretation

Use the existing validation, three-origin native OM4 and three-origin annual
observation evaluations, with eight draws and 32 sampling steps. Compare day-30
and day-365 blended error, component errors, individual members and ensemble
means, latent RMS trajectories, physical extremes and coast-proximate artifacts.
The principal question is graceful degradation over the year, not beating the
deterministic reference's absolute score.

Architecture metadata and a persistent direct-mode marker must survive checkpoint
save/load. All three evaluators use the same strict architecture-aware loader;
unmodified residual-only loaders reject direct checkpoints instead of silently
adding the residual connection back. Unit tests check this contract and the
zero-head distinction: persistence for residual prediction versus zero next
state for direct prediction.

This is one paired training seed and a short continuation. A direct arm failing
to learn the new output parameterization does not rule out training it from
scratch. A lower mean error alone does not establish plausible members, calibrated
spread or stable long-term behavior.

## Execution

One Beta production node, four B200s, one arm per GPU. Qualification checks the
full 256-update warm-up where applicable, two mixed updates, and architecture-aware
checkpoint reload. Production allocation is at most eight hours. Total pilot
ceiling including smoke tests and recovery: **40 allocated GPU-hours**. A Slurm
accounting gate checks previous actual/reserved allocations before launch.
Use online W&B for production. Register durable completion follow-up and an hourly
fallback reminder before ending the launch turn.

Campaign root:
`/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-direct-v1`.
Data and parent checkpoint are verified links to existing project storage; no
new data preparation or split changes are required.

Sources: [architecture](../../../src/samudra/experiments/diffusion_direct.py),
[trainer](../../../src/samudra/experiments/diffusion_interventions.py),
[runner](../../../scripts/run_diffusion_direct.py),
[Slurm launcher](../../../scripts/slurm_diffusion_direct_beta.sbatch).

Initial qualification job **212175** finished with three passing arms and one
failure, after 21m41s on four B200s (**1.44556 allocated GPU-hours**). It used an
abbreviated two-update warm-up. In `direct-warm`, teacher-forced relative latent
MSE fell from 0.947 to 0.598, but day-30 autoregressive relative MSE rose from
0.951 to 53.843; the subsequent native replay objective became nonfinite before
the first mixed update. The residual, direct-without-warm-up, and residual-warm
arms passed their two mixed updates and strict checkpoint reloads. This is a
failure of the abbreviated conversion, not a result for the planned full warm-up.
Its immutable runtime source was
`direct-v1-362478529fa8`, from commit `39dde12a6`; 28 targeted tests and all
repository hooks passed. Preserve these outputs under `smoke-212175/` and rerun
qualification using the actual 256-update conversion, without changing the
production recipe.

Replacement qualification **212178** completed all four arms and strict reloads
with exit 0 in 30m02s on four B200s (**2.00222 GPU-hours**). Runtime source
`direct-v1-b673a95fcd25` uses the same production recipe, with the full conversion
also applied during qualification. All arms completed two mixed updates and
saved 8,129 OM4 / 8,129 observation counts.

| Warm-up assessment on reserved training-era OM4 origins | Before | After 256 updates |
|---|---:|---:|
| Direct: teacher-forced next-slot relative MSE | 0.947320 | 0.005984 |
| Direct: day-30 autoregressive relative latent MSE | 0.951350 | 0.019181 |
| Residual: both relative errors | 0 | 0 |

These are latent agreement scores against the frozen teacher, **not physical
forecast errors**. They establish substantially better short-trajectory transfer
than the abbreviated conversion, not annual stability or transfer to the
observation-forcing task. The first mixed native loss remained much higher in
direct-warm (75.70) than in the residual control (0.4683); the un-warmed direct
arm's initial native loss was finite but approximately 9.64e15. Passing numerical
qualification does not mean these models already forecast well.

Production **212190** uses that qualified archive and starts each arm afresh
from the agreed parent; it does not resume the two-update qualification weights.
Both qualification jobs count toward the ceiling: **3.44778 GPU-hours consumed**,
plus at most 32 reserved for production, totaling **35.44778 of 40** before any
recovery. The initial late-night queue prediction was superseded by the actual
immediate start. A durable scheduler watcher and one-hour fallback reminder
are registered for production.

All four production workers confirmed online W&B initialization. Live runs:
[residual](https://wandb.ai/ocean_emulators/default/runs/diffint-direct-residual),
[direct](https://wandb.ai/ocean_emulators/default/runs/diffint-direct-direct),
[residual-warm](https://wandb.ai/ocean_emulators/default/runs/diffint-direct-residual-warm),
[direct-warm](https://wandb.ai/ocean_emulators/default/runs/diffint-direct-direct-warm).
The two warm arms have begun their processor-only updates; these are separate
from the 128 mixed updates and are not yet forecast results.

### First production check: approximately 20:05 UTC

| Arm | Completed mixed updates | Latest observation training loss | Latest OM4 training loss |
|---|---:|---:|---:|
| residual | 23 | 0.04773 | 0.57630 |
| residual-warm | 21 | 0.04640 | 0.48588 |
| direct-warm | 21 | 0.29135 | 0.90624 |
| direct | 2; stopped on update 3 | 0.77840 | 9.64e15 |

All three continuing arms have saved recovery checkpoints at update 16.
The warmed direct observation loss has fallen from approximately 0.656 to
0.291, but remains above the controls. These are training losses on varying
batches, not matched held-out scores or evidence of annual stability.

The un-warmed arm passed the two-update qualification but stopped at gradient
clipping on the third update. The recorded exception identifies a **nonfinite
total gradient norm**; it does not distinguish nonfinite individual gradient
entries from overflow while computing the norm. The very large initial native
loss and failed conversion are retained as evidence, not silently repaired by
disabling the guard or changing that arm's objective. The other three arms
continue unchanged; the runner will report an overall failure after they finish
because it preserves the failed arm's exit status. Check individual results and
completed evaluation artifacts rather than treating that aggregate exit as an
infrastructure failure requiring a complete rerun.

Recent speed is approximately 141–150 seconds per mixed update, suggesting
roughly another 4–4.5 hours of training, plus evaluations. This is an estimate;
the eight-hour allocation limit and forty-GPU-hour pilot ceiling remain fixed.

### Second production check: approximately 21:06 UTC

The residual arms have completed 49/128 mixed updates and direct-warm 45/128;
all continuing arms have finite logged losses and gradient norms. Each has a
verified `step-0032.pt` snapshot. The residual recovery checkpoints are at
update 48; direct-warm's latest recovery checkpoint is at update 32.

At the same observation update 44, training loss is **0.178296 direct-warm**,
**0.046561 residual**, and **0.046559 residual-warm**. The direct-warm loss is
lower than at the previous check, but this temporal comparison uses different
batches and does not establish a held-out gain. The two residual controls remain
nearly identical on this matched update. No final forecast evaluation has
completed. Recent throughput suggests **3–3.5 hours of training remaining**,
followed by evaluations. The un-warmed direct arm remains stopped; no extra
training was launched. The durable completion watcher is active and the hourly
fallback check is rearmed.

### Third production check: approximately 22:07 UTC

Residual and residual-warm have completed **75/128** mixed updates, and
direct-warm **69/128**. All three have saved `step-0064.pt` and update-64
recovery checkpoints (8,160 OM4 / 8,160 observation updates including their
parent). Logged training losses and gradient norms remain finite in these
continuing arms. The un-warmed direct arm remains stopped at its update-3
failure; its two completed-update records alone do not capture that failure.

At matched observation update 68, training loss is **0.149306 direct-warm**,
**0.045597 residual**, and **0.045601 residual-warm**. Direct-warm continues to
reduce its training loss, with the same varying-batch caveat as above; held-out
forecast results remain pending. Estimated remaining training is **2–2.5 hours**,
plus evaluation, with approximately five hours left in the allocation. The
completion watcher is active and the hourly fallback is rearmed. No additional
compute was submitted.
