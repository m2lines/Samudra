<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# C with structured completion tasks

Authorized 2026-10-12 after review of the [first-wave results](RESULTS.md). The user confirmed the
50% forecast / 25% surface-to-interior / 12.5% missing-variable / 12.5% spatial-gap mixture and a matched
forecast-only control. This is an experiment/results record, not intended to merge. Qualification and
submission status are recorded below; this plan alone is not evidence of a running job.

The question is whether allocating part of C's training to conditional reconstruction produces useful
interior/completion distributions and improves, preserves or harms its ordinary forecast skill. This uses
C's marginal almost-fair CRPS objective. "Structured" describes the masks/tasks, not D's transformed-field
losses, denoising score matching, or diffusion.

## Two matched arms

| Arm | Task distribution | Initialization and training |
| --- | --- | --- |
| F: forecast-only | 100% forecast | Seed 15, from scratch, 70 epochs |
| M: multitask | User-confirmed mixture above | Same seed, initial parameters, contexts, updates and four calls/window |

Both use the same 84M-parameter stochastic ConvNeXt U-Net plus a shared **94,864-parameter** conditioning
projection. C's original checkpoint/results remain a historical reference; F controls for the added
interface and changed validation policy. Training retains four members, almost-fair CRPS alpha 0.95,
two-input/two-output states, four calls per window, effective batch 32, existing normalization,
InstanceNorm, bf16, EMA, and the Rust loader with tauuo/tauvo/hfds and no hfds_anomalies. Other optimizer
settings are inherited from `common.yaml`.

The model uses fixed channel slots for history at nominal days −5/0 and requested states at +5/+10.
Physical values and availability masks distinguish missing fields from normalized zero. Output request
masks say which entries are to be inferred. Known output values, history availability, output availability,
and requests feed a 1×1 projection into the existing prognostic input. Its weights start at zero: tests
require the initial forecast to match C exactly under the same random draws. Both arms learn this projection.
There are no separate initializer/processor models or task-specific output heads.

One task is drawn independently per rank minibatch from a separate CPU RNG. Each window still uses four
joint two-state calls. The forecast task autoregresses each member's own history for all four calls. The
three reconstruction tasks instead use four consecutive true target pairs, supplying only the allowed
same-time fields at each pair; historical ocean states are unavailable for those tasks:

- **Surface:** supply only SSH and top-level temperature at the two output dates, request all remaining wet
  fields. This is synthetic OM4 surface-to-interior completion; SST/SSH are conditioning, not scored targets.
- **Variable:** uniformly choose U, V, temperature, salinity or SSH; hide that family at both dates and supply
  all other target fields. The same family is hidden through the four calls.
- **Gap:** hide a rectangle one quarter of the grid height by one quarter of its width, across all channels
  and both dates. Center it on a randomly selected wet surface cell, clamp latitude and wrap longitude.
  Supply fields outside the gap. The same gap is used through the four calls.

Known values are copied at output and excluded from loss. Requested-entry CRPS is normalized within each
active channel, retaining C's original wet fraction for that channel and averaging active channels. When
all wet channels/cells are requested, this equals C's original objective. Full-state forecasts never receive
future ocean fields or reconstruction targets. All tasks use labels from the same permitted training split.

This matches total training windows, updates and model calls, **not the number of forecast examples**:
M uses about half as many forecast windows as F. Task counts and objective contributions are logged per
epoch. Runtime/GPU-hours are measured; reconstruction's shorter gradient paths can change cost. A null or
negative result could reflect lost forecast exposure as well as task interference. No equal-compute or
equal-forecast-exposure superiority will be inferred from this pair alone.

## Evaluation and success criteria

Training data, validation and test boundaries remain those in the first-wave manifest. Existing normalization
stores are reused as requested. Model selection never sees 2014-10-10 onward test data.

Both arms now run an **eight-member, 360-day validation rollout every epoch**, from the first validation
origin, scoring nominal days 10/20/40/90/180/360. Validation-selected EMA minimizes their mean empirical CRPS;
the primary comparison remains final epoch-70 EMA. This differs from C's first-wave 90-day training-time
validation and is shared by F/M. One-step forecast validation remains; fixed completion validation masks
use up to four minibatches per rank at epochs 1, 5, 10, …, 70, with eight members.

Final evaluation compares the same thirty quarterly test origins, seven leads and sixteen members used in
wave one, with RMSE, empirical/fair CRPS, ranks, spread and coverage by variable/depth. Add a fixed, time-held-out
completion evaluation for all three reconstruction tasks, scoring only requested entries and preserving masks,
target dates, members and native-pixel maps. F on unfamiliar masks is a transfer diagnostic, not a trained
task-specific reconstruction baseline. The main causal comparison is F versus M on ordinary forecasting.

Use continuous-rollout diagnostics to check the observed salinity jump problem and sampled OHC variability.
Observation scoring remains secondary, with each product's verified cadence and members/means separated.
Completion success on OM4 does not establish observational initialization, real Argo-profile calibration,
or forecasting with unknown future surface fields. Do not select favorable masks, dates or members after seeing outputs.

## Execution and monitoring

Use Torch H100s following the user's request to use available Torch capacity. The initial plan used Beta;
its held qualification job 212521 was cancelled before allocating resources. Torch's H100 queue estimates
were earlier than RTX/H200 production capacity on October 12. Qualification uses one H100; both production
arms request two H100s with batch 4/rank and accumulation 4, retaining effective batch 32. The corresponding
four-GPU Beta recipe uses accumulation 2. These rank counts change sampling/RNG streams relative to historical
C; the two new arms remain matched to one another.

Reuse Torch's `sd-seeds-base-0b5320248.sif` container, whose `uv.lock` and `pyproject.toml` are byte-identical
to this source, with its installed x86 Rust loader. Bind an immutable source snapshot read-only; compare
dependency files again inside every allocation. Data is `/scratch/jr7309/data/om4-v2026-09/om4_onedeg`.
Compare metadata, coordinates, masks, all normalization chunks, and sampled ocean chunks against the Beta
first-wave store before submission; this is a sampled cross-host check, not a new full-store verification.

The initial estimate of **90–110 allocated B200 GPU-hours** for both runs plus initial evaluation was based
on C's measured 43.45 training GPU-hours and validation overhead. It is not an H100 timing estimate.
Qualification measures actual memory/runtime and all task branches before releasing full runs. Preserve
immutable source snapshots, resolved configurations, allocation records and actual GPU-hours.

Full runs require online W&B before training. CPU-only `afterany` callbacks and hourly reminders cover
success/failure and queued jobs. Preemption requeues from the latest completed epoch without consuming the
final callback. Recover straightforward failures and report blockers. Later experiment waves still need approval.

Implementation: `multitask_model.py`, `train_multitask.py`, `prepare_multitask.py`, `multitask_torch.sbatch`
and the alternative `multitask_beta.sbatch`.
Local checks cover exact initial C forecast parity, mask leakage under nonzero learned conditioning, copied
known values, requested-only gradients, dry cells, task-RNG replay and checkpointed backward equivalence.
