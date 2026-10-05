<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Three-day joint-training and mechanism investigation

Authorized 26 September 2026 at 13:30:45 UTC. **Report deadline: 29 September
2026 at 13:30:45 UTC (09:30:45 America/New_York).** The deliverable is conclusions
about performance and/or mechanism, not a requirement to exhaust compute or
complete an unbounded search. Finalize the report within that wall-time cap.

## Scope and priorities

1. Qualify a Samudra-2-backbone-matched evolution model: widths
   [280, 380, 480, 520], capped GELU, expansion two, zonally periodic upsampling,
   and the reference output kernel. Retain affine InstanceNorm, no running stats,
   and fixed loss initially. This is architecture matched, not a reproduction of
   the BatchNorm/dynamic-loss reference recipe. Keep the initializer fixed in
   architecture so the processor change is explicit.
2. Use one training loop with scratch-observation, sequential OM4/observation,
   and scheduled mixed OM4/observation arms. The sequential arm is a control for
   ordering; the mixed arm retains OM4 in its observation-heavy tail. Retain
   reconstruction as an auxiliary objective, rather than a separate observation
   reconstruction/adaptation phase. Start with the existing shared output/state
   interface and forcing adapter; extra decoders/task embeddings are conditional
   follow-ups, not automatic additions.
3. Diagnose existing completed checkpoints while the new comparison runs:
   case-specific initial-state information versus seasonal structure, perturbation
   sensitivity at short/long leads, and bias/seasonal versus anomaly errors.
   Use training-derived replacements; never derive climatology or tune
   interventions from held-out outcomes. Avoid raw state swaps between
   independently trained latent representations.
4. If time permits, test whether richer OM4 supervision is responsible for the
   benefit using observation-pattern versus full-state supervision on matched
   OM4 examples. Do not sacrifice completed main comparisons/reporting for this
   optional extension.

**Excluded:** the known velocity/preprocessing ceiling investigation and
quarter-degree training. Stay on the existing cheap one-degree grid; the user
intends quarter-degree work after the mechanism is better understood.

## Qualification and resource bounds

Use Torch/RTX by preference, seed1729 initially, existing verified local datasets,
and no repeat bulk transfer. Check actual scheduler capacity, quota and throughput
before production. Initial operational ceiling: **120 allocated GPU-hours**, with
all qualification/calibration/retry allocations charged. This is a new bounded
campaign, not leftover allocation from a prior wave. Do not infer authorization
for new seeds or unrelated searches. Production update counts are set from
qualification throughput and remaining wall time, then pinned before submission.
Reserve the final six hours for completing evaluations, analysis and reporting;
do not launch a job whose requested duration exceeds the remaining deadline.

Maintain approximately matched OM4 and observation exposure between sequential
and mixed arms. Record both task update counts, effective batch, loss weights,
optimizer policy and GPU-hours; sampling proportions and loss scales must not
silently double-count a task's intended weight. Preserve early checkpoints,
OM4 retention curves and observation learning curves. Separate final-budget raw
weights from observation-validation-selected weights.

Selection remains the fixed v3 integrated-plus-spatial-spectral observation
validation score. Report components, persistence, 30-day and annual held-out
metrics. Annual and intervention diagnostics do not alter the selected checkpoint.
No held-out tuning; one-seed limitations and confounded comparisons are explicit.

## Execution record

At start, Torch authenticated multiplexing works and the user queue is empty.
Scratch usage is 4.04 TB of 5 TB (approximately 0.96 TB free). Prior InstanceNorm
results and all source checkpoints remain intact. The old monitor timer stays
disabled; monitoring uses interruptible waits and meaningful updates, with
previously authorized Slack blocker alerts if user action is required.


### Initial implementation and diagnostic checkpoint

Backbone producer `b9d6ebad9f240e5fd92f6e0bb86c4d774ccfe6ea` passed 18 tests
and pre-commit checks. Build 18580732 completed; GPU qualifications 18580735 and
18580736 are pending group CPU capacity. The new processor has 83,872,357
parameters. Annual bias/anomaly producer `6d8a8d26b` passed two tests; CPU retry
18580810 completed after correcting the interpreter used by failed 18580782.
[Interim findings](three-day-findings-2026-09-26.md) already distinguish reduced
bias from year-end predictive skill against training climatology.

Implementation next: a single optimizer/loop for scratch, sequential and mixed
controls; no reconstruction-only phase or best-weight reload at a task boundary.
Preserve identical per-task sample order and exact per-task exposure counts
across sequential/mixed arms. A deterministic cumulative schedule can implement
an OM4-heavy to observation-heavy mixture while retaining OM4 at the end. Keep
legacy model loading/evaluation unchanged; checkpoint manifests now carry
`evolution_architecture` and strict qualification checks. New loop qualification
and resume tests are required before production. Production counts/rates remain
unfrozen until real throughput is measured.

### Single-loop implementation, 26 September 13:57 UTC

Producer `50630959c6a8aed89fcc0152cab28b16f8c377d3` implements the single-loop
trainer with one AdamW optimizer, activation checkpointing, exact task counts,
and epoch-shuffled sample streams indexed by each task's own update count. A
sequential/mixed pair therefore receives the same examples in the same order
within each task. For a mixed run with N OM4 and M observation updates, the
cumulative observation count at fraction f of the run is
`floor(M * (f/4 + 3*f*f/4))`. This schedule requires an overall observation
fraction at most 4/7; the final production fraction will retain OM4 in the tail.
The code supports a per-task learning rate and per-task warm-up; rates and
production counts remain subject to qualification, not inferred from this formula.

Both tasks retain reconstruction as an auxiliary loss. OM4 updates bypass the
ERA5 adapter; observation updates train all three modules. Gradient checks fail
on missing core gradients or unexpected adapter gradients during OM4 updates.
Checkpoints include optimizer moments, task/global counts and random-generator
states. No best-weight reload or optimizer reset occurs at the task boundary.
Held-out evaluation has an explicit selected-only mode for fresh runs, retaining
selected-state persistence and training climatology without loading historical
D source weights into the larger backbone.

Twenty-nine targeted tests passed, including sample/exposure matching, task-switch
optimizer continuity, CPU restart equivalence, nonfinite rejection, architecture
and evaluation compatibility. All commit hooks passed. Build **18581034** submits
fresh producer-matched OM4 and observation qualifications, followed by a mixed
probe dependent on both succeeding. The probe performs three OM4 and two
observation updates plus a repeated next-update comparison after checkpoint
restoration; its extra work counts toward allocated GPU time. Production also
requires this probe's matching qualification marker. Earlier backbone jobs are
superseded only if still pending, preserving their submission records. No
production results are claimed yet.

Build 18581034 subsequently completed in five CPU seconds. Its new jobs are
18581035 (OM4 qualification), 18581036 (observation qualification), and 18581037
(mixed probe, dependent on both). The first two are pending group CPU capacity;
the probe is pending dependencies. Original 18580735 / 18580736 were cancelled
while pending with zero elapsed allocation. All new-campaign GPU allocations
remain zero at this checkpoint.

### Prespecified mechanism comparisons

Diagnostic producer `d67e600319222fea87d731c65d0b6e04774a285f` adds five annual
conditions on each previous InstanceNorm selected checkpoint: unmodified;
initial hidden state replaced with that model's training seasonal mean;
initial velocity slots alone replaced with their training seasonal mean;
hidden-state seasonal replacement immediately after day 30; and future ERA5
forcing replaced with a training-only monthly climatology. Both SST/ADT state
slots at both history times remain unchanged in the hidden-state replacements.
The forcing intervention retains the original initialization/history inputs.

Seasonal latent means come from all 243 training initializer outputs, grouped by
the last history frame's calendar month. Forcing means use only the training
NPZ inputs, deduplicating identical timestamps. Each model uses its own latent
means; no cross-model state swaps or assumptions of physical identifiability.
The day-30 replacement can be outside the evolved-state distribution and will
be interpreted as a sensitivity diagnostic, not proof of a particular physical
mechanism. These fixed comparisons use the same three previously inspected
annual origins; they do not tune or reselect checkpoints.

Six mechanism/annual tests passed, including observed-slot preservation,
intervention timing and unmodified inputs. Runtime baseline forecasts must also
match the native forecast path exactly. Commit checks passed. CPU build 18581137
will submit the two one-hour-capped diagnostic jobs, charged to the new campaign's
120 GPU-hour ceiling. Training remains pinned to its separate producer 50630959c.

### H200 capacity recovery, 26 September 14:07 UTC

The RTX partition had idle nodes but its shared QOS CPU limit prevented starts.
A complete H200 preemption `sbatch --test-only` request was accepted with an
immediate start estimate. The pending requests were cancelled before allocation
and replaced using the same code, model/data arguments, one-hour caps and
explicit dependency graph. Routing and preemption/requeue settings were verified
with `scontrol`; no scientific protocol changed.

| Work | H200 job | Observed state at 14:07 UTC |
| --- | ---: | --- |
| OM4 larger-backbone qualification | 18581191 | Running; verified producer loaded, OM4 cache warming |
| Observation larger-backbone qualification | 18581192 | Running; pinned container/code loaded |
| Mixed-task/resume probe | 18581193 | Pending both qualification successes |
| Previous scratch checkpoint mechanisms | 18581194 | Running; training seasonal-state extraction reached 26/243 |
| Previous OM4-initialized checkpoint mechanisms | 18581195 | Running; training seasonal-state extraction reached 26/243 |

These replace RTX jobs 18581035/18581036/18581037/18581146/18581147,
respectively. Diagnostic build 18581137 completed. The five requested caps total
five GPU-hours within the 120-hour campaign ceiling; actual allocated time will
be accounted from Slurm, including every retry. Running allocations and startup
outputs do not yet establish successful qualification or scientific results.

### Follow-up diagnostic: continuously remove hidden carryover

After the initial-state intervention showed little year-end effect, a fixed
follow-up will reset hidden state to its training seasonal mean **before every
forecast step**, either for all non-SST/ADT slots or only the internal velocity
slots. Each step still carries the model's own predicted SST/ADT history; future
observations are never supplied. This tests the role of evolving hidden carryover,
which a one-time initialization intervention cannot remove.

Run these two conditions plus an unmodified paired baseline on each existing
selected checkpoint and the same three annual origins. Reuse training seasonal
means only after verifying checkpoint, training manifest, training statistics and
mean-file hashes. This is an explicitly exploratory follow-up prompted by the
first diagnostic, not a checkpoint-selection rule or test-tuned correction.
Continuous resetting can impose an artificial stabilizing constraint; results
will not establish a physically interpretable latent state. Model-specific mean
initialization versus processor-weight effects remain only partly separated.

### Qualified production protocol — frozen before submission

Producer **dd3a05b07554ce0c1c9f589643d2afb764ab1440** passed fresh observation
fitting job 18582538 and mixed-loop/resume job **18582539**. The latter completed
six OM4 and four observation updates, including switches back from observations
to OM4. Weights, optimizer moments and RNG state were serialized/restored exactly.
Native repeated-update weight RMSE was 2.308e-7; disk-restored replay RMSE was
2.278e-7. Maximum differences were approximately 1.00e-4 for both, consistent
with the measured native CUDA arithmetic variation rather than a checkpoint-state
loss. All gate artifacts and the earlier failed probe are preserved.

The 100-update OM4 qualification measured 191 training seconds; observation
micro-update fitting took approximately 0.66 seconds on a repeated cached
example. Mixed-probe cold observation batches were slower because they loaded
new examples. Full OM4 cache initialization on the retry node took several
minutes and peaked near 78 GiB host RSS; it was allowed to finish without restart.

| Literal new model name | Ordering | OM4 updates | Observation updates | Total updates |
| --- | --- | ---: | ---: | ---: |
| Samudra2 scratch | Observation-only training from random weights | 0 | 16,000 | 16,000 |
| Samudra2 sequential | OM4 first, then observations; one optimizer throughout | 8,000 | 8,000 | 16,000 |
| Samudra2 mixed | Deterministic OM4-heavy to observation-heavy mixture | 8,000 | 8,000 | 16,000 |

All arms: seed 1729, effective batch eight (eight single-example microbatches),
AdamW with learning rate 1e-4 for both tasks, weight decay 0.01, gradient norm
cap one, no warm-up, InstanceNorm, fixed loss and reconstruction auxiliary weight
0.1. No standalone reconstruction/adaptation phase, optimizer reset or selected
weight reload occurs within training. Sequential/mixed receive identical example
streams within each task. The mixed schedule's observation fraction rises from
approximately 12.5% to 87.5%, retaining OM4 in its tail.

Each update has task-dependent cost; 16,000 updates is rough compute matching,
not a FLOP-equivalence claim. Report per-task exposure and actual allocated
GPU-hours. Primary final checkpoints are selected by the unchanged nine-origin
integrated-plus-spectral observation score over the whole arm. Fixed-budget raw
checkpoints supply additional comparisons, including scratch at 8,000 observation
updates against the two 8,000-observation arms. Do not label that raw checkpoint
as the validation-selected best through 8,000 updates.

Validate every 100 global updates and at observation milestones 10, 25, 50, 100,
250, 500, 1k, 2k, 4k, 6k, 8k (plus 12k/16k for scratch). Preserve raw milestone
weights/optimizer and selected weights with their global/per-task counts. Record
OM4 validation retention for all arms, including scratch, without using it for
observation checkpoint selection. Scratch receives no OM4 training gradients.

Use one H200 per arm, up to 28 allocated hours each; internal training cap is
27 hours. Absolute training cutoff is **29 September 07:30:45 UTC**, reserving
six hours before the report deadline. Requested production caps total 84
GPU-hours; actual prior allocations, ongoing diagnostics, evaluations and retries
must remain within the 120-hour campaign ceiling. No new seed is included.
