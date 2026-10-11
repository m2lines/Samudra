<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Direct next-latent prediction pilot

User-authorized follow-up to the [latent-amplitude diagnostics](coastal-decoder-results.md).
Test whether replacing the persistent residual update
`next = latest + processor(history, forcing, context)` with
`next = processor(history, forcing, context)` reduces annual drift while
retaining short-lead skill. History still shifts to `(latest, next)`. The
processor's internal residual blocks remain unchanged; there is no outer
identity bypass or implicit addition in the direct variant. Direct prediction
does not mathematically guarantee stable recurrence.

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

One Beta production node, four B200s, one arm per GPU. Initial smoke test checks
two warm-up updates where applicable, two mixed updates, and architecture-aware
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
