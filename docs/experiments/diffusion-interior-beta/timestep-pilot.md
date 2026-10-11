<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Five-/ten-day endpoint-consistency pilot

User-authorized follow-up to the [stability campaign](interventions-v2.md).
Question: can supervising a second time interval, and requiring consistency
between one ten-day advance and two five-day advances, reduce annual drift in
the ordinary five-day forecast? Success means plausible late fields and slower
error growth without degrading short-lead skill, not beating the differently
configured deterministic model's absolute RMSE.

## Matched arms

| Arm | Added OM4 loss |
|---|---|
| control | None; ordinary five-day continuation |
| supervised | 0.5 times ten-day physical denoising loss |
| consistency-weak | Same physical loss + 0.1 times endpoint consistency |
| consistency-strong | Same physical loss + 1.0 times endpoint consistency |

All start from the same cooled 8,000-observation + 8,000-OM4 parent, receive
128 updates (64 each source), effective batch eight, seed 271829, fresh AdamW,
cosine LR 1e-5 to 1e-6, gradient clip 1 and weight decay 0.01. Retain snapshots at
32/64/128; final evaluation uses the prespecified 128 checkpoint. No replay or
new spatial loss is added. Original objectives remain unchanged; the auxiliary
is added to every OM4 microbatch using the same origin and surface availability.
The supervised arm receives extra target scoring, so it is needed to distinguish
that effect from composition consistency. Report actual GPU time as well as
matched updates; added objectives have extra compute.

## Endpoint semantics

The existing five-day processor shifts [z(t-5),z(t)] to [z(t),z(t+5)]. A shared
ten-day path returns [z(t),z(t+10)] and receives the ordered forcing/calendar
blocks at both five-day intervals. Its dynamics backbone and output head are
the same weights as the five-day path. A zero-initialized conditioning adapter
receives both blocks and the interval; the residual is multiplied by two.
At initialization this is only an approximation, not a known ten-day integrator.
Five-day calls retain the exact original computation and do not use the adapter.

Compare only the newest latent slot against two ordinary five-day calls; do not
compare their different historical slots. The penalty is squared endpoint
difference divided by detached short-endpoint per-channel spatial RMS (floor
0.1), averaged over batch, channels and cells. Both branches receive gradients;
neither is a frozen teacher. Normalization does not guarantee against collapse
or excessive growth, so retain physical supervision and inspect physical fields.

The decoder reads two slots trained at five-day spacing. For the extra physical
loss, pair the short-path day-five state with the direct-path day-ten state and
supervise against OM4 at those two times. This uses a short-path historical
scaffold, not a standalone variable-spacing physical decoder. No target interior
is passed to the initializer or processor. This is composition regularization
of the existing dynamics, not a claim of arbitrary-timestep inference.

Use the same full-latitude masks, source splits, normalization, observation
targets and forcings as the current campaign. Noise correlation remains 0.5,
with two training draws and 32 diffusion sampling steps. Diffusion sampling
steps and the ocean's five-/ten-day interval are separate quantities.

## Evaluation and provenance

Run the unchanged five-day evaluators: nine validation months; native OM4
rollouts; annual observation starts January 2015/2018/2021, eight members and
32 denoising steps. Report day-30/day-365 four-component normalized RMSE,
component errors versus lead, absolute and relative growth from day 30, latent
RMS, member/mean maps and seasonally contextualized field statistics. Retain
CRPS, spread/error and coverage. Compare member spectra and directional artifacts;
a smooth but inaccurate or underdispersed field is not a stability success.
Annual cases are repeatedly consulted exploratory tests, not a fresh holdout.

Saved `model` weights load strictly into the existing five-day evaluator. Save
the auxiliary `timestep_adapter` separately, including in snapshots, and preserve
its optimizer state for training resume. A GPU smoke test must verify actual
backpropagation for all four arms and standard evaluator compatibility before
production starts. Full runs log online to W&B group
`diffusion-interventions-timestep`.

Beta outputs: `/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-timestep-v1`.
Reuse verified data and parent checkpoints via read-only input links. Pin and
fingerprint source and container. Four arms each occupy one B200 in a four-GPU
allocation; two-update smoke capped at one hour, production including evaluations
capped at eight hours. Source, jobs, callbacks and accounting are recorded in
`outputs/diffusion-timestep-v1/manifest.json` locally.

## Shared budget

Keep the combined campaign below 192 allocated GPU-hours. Existing production
211873 now has a 31-hour cap (124 GPU-hours), reduced from 40 hours; its observed
progress projects completion before that cap. Keep 28 hours reserved for the
already-submitted learning-curve evaluations. Existing qualification consumed
0.999 GPU-hours. This pilot reserves at most four for qualification and 32 for
production, for a combined maximum of 188.999. Admission checks use Slurm
accounting and reject overruns; recoveries must use remaining budget. Existing
training updates and evaluation coverage are unchanged. Monitor all jobs with
durable callbacks and one-/two-hour fallback reminders.
