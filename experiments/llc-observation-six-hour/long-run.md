<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Longer training with observation-only cooldown

Continue both completed six-hour experiments from their raw 2,000-update
checkpoints to 10,000 main-phase updates, then 1,000 observation-only cooldown
updates. These are total budgets, including the existing 2,000 updates. Each
arm performs 9,000 additional updates. The mixed main phase continues alternating
LLC and observations; the control continues using observations throughout.

The cooldown follows the deterministic OM4 preset
[`deterministic_mixed_large.yaml`](../../../src/samudra/configs/observations/deterministic_mixed_large.yaml):
linear learning-rate decay from 1e-4 to 1e-6 over 1,000 OBS-only updates,
with EMA reset at the phase boundary. The first cooldown update uses a slightly
reduced learning rate and the last uses 1e-6. The longer 10,000-update main budget
is this experiment's choice; the referenced OM4 preset uses 4,000 main updates.
The snapshot experiment continues without replay or monthly supervision.

| Setting | Mixed | OBS-only control |
| --- | --- | --- |
| Main phase | 10,000 alternating LLC/OBS updates | 10,000 OBS updates |
| Cooldown | 1,000 OBS updates | 1,000 OBS updates |
| Total sampled windows | 40,000 LLC + 48,000 OBS | 88,000 OBS |
| GPUs | Four GB200 | Four GB200 |
| Effective global batch | Eight windows | Eight windows |
| Per GPU | Microbatch one, accumulate two | Microbatch one, accumulate two |
| Requested production wall time | 16 hours | 10 hours |

Architecture, six-hour step, 29-frame initializer history, forcing, losses,
normalization, data splits, seed and eight fixed validation origins remain
unchanged. Validation runs every 100 updates and scores 24/48/72-hour forecast leads.
Main-phase LR remains constant at 1e-4. Milestones at updates 4,000, 6,000,
8,000 and 10,000 save checkpoints; they do not change LR or rollout duration. Prescribed future ERA5 weather
continues to make this a hindcast experiment. The short-run results remain
[recorded separately](results/mixed-versus-obs-only.md).

## Checkpoint continuity and qualification

The new output directories preserve the original checkpoints. A one-time
migration verifies the parent checkpoint SHA-256 and original code manifest.
It permits only the reviewed removal of the snapshot cooldown validation guard,
main/cooldown budgets, milestone schedule and the future cooldown scheduler.
The short qualification may additionally change validation and save cadence.
All other code, data, runtime and scientific settings must match. The migration
preserves model, optimizer, EMA, source exposure, sampler and all four ranks'
random states. It resets the best-score tracker for the extended trajectory.
Both old and new manifests and the allowed changes are recorded in lineage.json.
Subsequent restarts load the new output's latest checkpoint.

Each arm qualifies from its own original 2,000-update checkpoint: four more main
updates, checkpoint/resume, then four OBS-only cooldown updates. Qualification
checks cumulative source exposure, 24/48/72-hour validation, the phase flag and final
learning rate. Those qualification updates are discarded; production continues
from the original parent in separate outputs. A CPU test checks exact model and
EMA agreement with uninterrupted training after restarts at the phase boundary
and halfway through cooldown.

Parent runtime: `83df2678f91c29aba4187e79fdc9e8e46bfe0d54`.
Parent checkpoints are `outputs/*-six-hour-v1/saved_nets/main-2000.pt` under the
respective original EAI run directories. SHA-256 values:

- Mixed: `fc197be1591fd25bfabb2b5b5e0dabbabffd8e7e8dc442fd78c24d72d81d0294`.
- OBS-only: `b8a717b1b443c506d5835d825b7cd9124e011eec859b0439dd31ca8a317070a7`.

## Comparisons and execution

Compare both EMA models at exactly update 10,000 before cooldown and update
11,000 after cooldown, using the same SST/SSH metrics and persistence support.
Report best-selected checkpoints separately. Selection minimizes the equal
channel/lead mean of observation normalized RMSE over the eight fixed origins.
The new best-score tracker spans validations at updates 2,100 through 11,000
across both phases; it is not reset at cooldown. EMA itself is reset there. The OBS-only 6,000-update checkpoint
matches the final mixed run's 48,000 OBS draws, but has different optimizer and
EMA histories and has not undergone cooldown. Treat that as a secondary exposure
comparison. Before/after cooldown changes combine additional updates, source
selection, learning-rate decay and EMA reset; they do not isolate those effects.

These continuations retain the original single seed. They are not independent
replicates. LLC validation retains full-state persistence's advantage in initial
interior information, and the raw LLC Eta/observed ADT discrepancy remains.

Use standard EAI QoS and online W&B, with separate outputs and completion
callbacks. Estimate about 11–12 hours of remaining mixed computation and 6–7
hours for the control from the initial measured throughput, excluding queue time.
Qualification and production have an atomic gate so only one allocation performs
a given arm's qualification. No further arms are included in this budget.

Submitted on November 20, 2026: mixed production job `234626` and OBS-only
production job `234629`, both under standard QoS. Short test-queue qualification
jobs `234627` and `234630` share each arm's qualification gate. All four jobs
were pending at submission; actual GPU qualification and continued training
remain unverified. Completion callbacks and separate monitors own both arms
through the 11,000-update endpoint. Queue estimates are provisional.

Deployed runtime commit: `3416705c5`. Both archives have SHA-256
`2b73b7884d94932bc1cdd17ffae4d335fe357ab006674e43a5f159ee3f3fad21`.
Local checks passed 27 snapshot, pipeline and configuration tests, adversarial
review, the document's cold-reader review, and all pre-commit hooks. The new EAI
run directories end in `llc-obs-six-hour-long-20261120` and
`obs-only-six-hour-long-20261120` under the existing project run root.
