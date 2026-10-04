<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Separating regional initialization quality from gradient interference

**Production is running.** Job 209686 started October 4 at 2:17:31 a.m. ET on
`b1-14-s1-dgx-01-c04`. At 2:40 a.m., all four arms had finite training updates
and their first observation validation. This is confirmed execution, not a
queue estimate; final comparisons await the completed budgets and evaluations.

Predeclared October 4 after completion of the [first mechanism ablations](extent-ablations-2026-10-03.md),
within the user's autonomous iteration window ending Monday around 9 a.m. ET.
The strongest patch variant on validation was true-state initialization
(0.7356 versus 0.8293 for the original native task), but omission remained better
(0.6866). That evidence motivates separating input quality, initializer gradient
routing, auxiliary initializer losses, and task weighting. The test cohort is
used descriptively; it does not choose these interventions or checkpoints.

This is one further four-arm, one-seed comparison, alongside the already
qualified [accessory/capacity wave](extent-representation-2026-10-03.md). It may
overlap that wave on a second four-GPU node, for at most eight production GPUs.
No existing run or source pin changes.

## Exact tasks and controls

All four arms use seed 1729 and exactly 4,000 optimizer updates: 1,000 global
OM4, 1,000 native quarter-degree OM4 patches and 2,000 global observation updates.
Reuse the verified 1° OM4 v3 global store, v2026-09 quarter-degree five-day-mean
patch cache and fixed observation samples. Global fields are 180×360; native
patches are 128×128 with 64×64 scored interiors. The model evolves 77 physical
channels through six nominal five-day steps on OM4 tasks. Observation tasks
retain six or seven forecast bins according to calendar-month overlap for OHC,
with surface metrics at 5, 15 and 30 days. No future state boundary conditions
or fine-resolution fields enter observation inference.

| Literal name | Initial state on native patch tasks | Native objective and gradient routing | Primary comparison |
|---|---|---|---|
| U-patch-detach | Same learned initializer and masked surface inputs as U-multitask | Six-step forecast loss only; initializer evaluated without gradients; processor gradients retained | Against U-patch-truth: same gradient routing/objective, learned rather than true initial states |
| U-patch-forecast | Same learned initializer and masked surface inputs as U-multitask | Six-step forecast loss backpropagates through both initializer and processor; omit native reconstruction/completion losses | Against U-patch-detach: whether forecast gradients through the initializer help or hurt; against U-multitask: remove only native auxiliary initializer losses |
| U-patch-loss01 | Original learned initialization and native task | Multiply the entire native loss by 0.1 before backward/clipping; keep LR 1e-4 | Against U-patch-lr01: reducing gradient contribution versus reducing parameter step size |
| L-patch-truth | Two true full native physical states | Six-step forecast loss only, no native initializer gradients | Against L-multitask and L-global: does true-state patch training also help the bounded local processor? |

The three U arms have 62,967,680 parameters; L-patch-truth has 35,681,936.
All retain the same 31.28M-parameter U-Net initializer. The local processor's
four-cell one-step receptive radius conditions on supplied states and geometry;
its shared initializer remains nonlocal. The global and observation tasks,
including learned initialization and their reconstruction/completion losses,
remain unchanged in every arm. No artificial visibility mask is removed from
the learned-initializer patch variants.

Reference definitions: **U-multitask** and **L-multitask** are the original
shared-initializer native-patch arms; **U-global** and **L-global** replace native
updates with additional global OM4 updates. **U-patch-truth** is the completed
U-Net true-initial-state ablation; **U-patch-lr01** is the completed native LR
1e-5 ablation, which retains unscaled native gradients in Adam's moments.

The loss-weight arm also changes the interaction with clipping: when both the
original and scaled gradient norms exceed the threshold, normalization can
erase much of the scale difference. Log both scaled and unscaled native losses
and the pre-clip gradient norm. This is a controlled training intervention,
not an assertion that loss weight and task frequency or LR are equivalent.

## Fixed training, selection and qualification

Keep the progressive mixed schedule, per-task sample seeds, accumulation of
eight examples, AdamW LR 1e-4, weight decay 0.01, clipping at 1, and no warmup.
Checkpoints use the same frozen integrated-plus-spectral global observation
validation composite. Report the same 96 monthly test forecasts and initialized
persistence with common test-climatology normalization. One seed limits the
strength of causal/generalization claims; repeated test reporting is exploratory.

Use a new immutable producer, fresh fitting qualifications for U and L, and a
real joint/resume probe for each arm. The new qualification contract binds
initialization mode and native loss scale. Probes must verify initializer
gradients are absent for detached/true-state native tasks and present for the
forecast-only/shared variants. Unit tests check actual gradient routing and
forecast-only values, including exclusion of auxiliary losses.

Production requests one beta node, four GPUs, 144 CPUs, all node memory and a
16-hour cap. Admit only if each short probe extrapolates below 12 training hours;
the per-arm training loop stops safely at 14 hours, preserving a checkpoint if
needed. This leaves startup/evaluation allowance and aims to finish comfortably
before Monday review. No queue-chasing resubmission or additional seed is planned.

Results and exact job/source provenance will be appended after qualification.

## Submission, October 4 at 1:56 a.m. ET

Producer `b8092783bf1e47b6ac670251feba5fbd2902a5bd`; source archive SHA-256
`fc9d5196a9dc9fa262453499e69724a2b400eb3ab33d6ab0e7113f6d8f3ee778`.
Local checks passed 24 targeted tests, Ruff, mypy for four touched Python files,
shell syntax and diff checks. All existing production uses its earlier immutable
producer. Fresh qualifications are required under the extended contract.

| Stage | Arm/family | Job |
|---|---|---:|
| Fit | Shared U-Net family | 209674 |
| Fit | Local family | 209675 |
| Joint/resume probe | U-patch-detach | 209676 |
| Joint/resume probe | U-patch-forecast | 209677 |
| Joint/resume probe | U-patch-loss01 | 209678 |
| Joint/resume probe | L-patch-truth | 209679 |

Both fitting allocations were running at this snapshot; the four probes were
pending on their corresponding fitting job. Each qualification requests one GPU,
16 CPUs, 128 GiB and up to two hours on `beta_test`/`test`. Production has not
been submitted. Root:
`/projects/ny/lz1955/multiscale/jrusak/runs/2026-10-04-extent-initializer`.
[Exact receipts](artifacts/extent-initializer-2026-10-04/qualification-jobs.json).

## Qualifications passed; production queued at 2:17 a.m. ET

All six qualification jobs completed with exit 0. Fitting loss decreased from
1.703 to 0.283 for U and 1.556 to 0.952 for L, with all required component and
completion gradients present. Each joint probe verified exact serialization
and restoration, and replay within the unchanged native-nondeterminism test.
The contracts explicitly record `detach`, `forecast`, `shared` with loss scale
0.1, or `truth`, as appropriate. Peak GPU memory was 71.6–71.7 GiB, including
the global-data cache.

| Arm | Probe seconds/update | Extrapolated hours for 4,000 updates |
|---|---:|---:|
| U-patch-detach | 9.89 | 10.98 |
| U-patch-forecast | 9.62 | 10.69 |
| U-patch-loss01 | 9.82 | 10.91 |
| L-patch-truth | 6.14 | 6.82 |

Every arm passes the predeclared 12-hour training extrapolation gate. Startup,
validation and evaluation are additional; per-arm training caps remain 14 hours
within the 16-hour allocation. Qualifications used **0.860 allocated GPU-hours**.

Production **209686** is pending beta priority with successful `afterok`
dependencies on all four joint probes. It requests the declared four GPUs,
144 CPUs and full node memory. A dry-run start estimate was **October 4 around
3:46 a.m. ET**, not a confirmed start. This allocation may overlap the already
running representation wave 209608. There are no failed attempts or retries.

[Qualification evidence](artifacts/extent-initializer-2026-10-04/qualification-results.json)
and [exact production receipt](artifacts/extent-initializer-2026-10-04/production-jobs.json).

## First production validation, October 4 at 2:40 a.m. ET

The job started earlier than the tentative queue estimate, without resubmission.
U-patch-detach reached 147 updates, U-patch-forecast and U-patch-loss01 reached
146, and L-patch-truth reached 164. Their observation validation composites at
100 updates were respectively 2.6141, 2.5989, 2.4499 and 2.5664. These early
values establish functioning training and evaluation, not a result at the
planned 4,000-update budget. Production overlaps the representation wave as
declared, with eight GPUs allocated in total. No recovery or protocol change
was required.
