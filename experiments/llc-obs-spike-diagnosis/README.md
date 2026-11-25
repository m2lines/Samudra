<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Diagnosing the loss spikes in the LLC + observation runs

This is an experiment record, not intended to merge. It continues
[#1037](https://github.com/m2lines/Samudra/pull/1037), which trained a
surface-initialized ocean model on coarsened LLC4320 snapshots and observed
SST/SSH (OISST, DUACS) at six-hour steps. Both of its long runs, mixed and
observation-only (OBS-only), had transient validation spikes
([training-curve stability](../llc-observation-six-hour/results/training-curve-stability.md)).
We want to know what causes them and fix them before comparing training mixtures,
because a spike adds noise to every comparison made after it.

Written by Claude Opus 5.5 (Claude Code), driving this work while Jesse is on leave.

## What the logs already show

W&B logs one total gradient norm per update. Every validation spike follows a
burst of large gradient norms, and every burst is on an OBS update:

| Run | Updates | Gradient norm (run median) | OBS forecast loss (run median) |
| --- | --- | --- | --- |
| Mixed | 3304–3314 | up to 8.2 (0.12) | up to 0.21 (0.0024) |
| Mixed | 6660–6676 | up to 6.4 | up to 0.19 |
| OBS-only | 4871–4877 | 1.8 → 211 → **3745** (0.05) | 0.007 → 2.0 → **56** (0.0017) |

The surface-completion loss depends only on the initializer network, and it
stays at its usual level while the forecast loss runs away over several
consecutive updates. In the mixed run, the LLC loss degrades 10–20 updates after
the OBS burst. Gradient clipping (at norm 1) rarely acts, since typical norms are
0.05–0.12.

**Working hypothesis.** The spikes are an instability of the 12-step forecast
rollout that builds up in the weights, on the observation path, where only one
step in four is supervised. The same unconstrained sub-daily behavior may
cause #1037's growing daily cycle and its 30-day error growth.

## Plan

### Step 0: diagnostics without training

- `tide_check.py`: how much of LLC's surface variance (SSH, SST, surface
  velocity) is sub-daily, including tides, and how LLC SSH compares with DUACS
  ADT as a midnight snapshot versus a daily mean.
- `forecast_oscillation.py`: the within-day range of #1037's 30-day forecasts by
  forecast day, against LLC truth, and whether the oscillation sits at one cycle
  per day everywhere (forced or aliased) or follows the inertial frequency
  (2|sin(latitude)| cycles per day).

### Step 1: replay the spikes

Training is deterministic, and #1037 saved full checkpoints at milestones,
including the next update's sample plan and all four ranks' RNG state. So a run
resumed from a milestone with the same code, container and world size should
reproduce the original updates exactly, spike included.

| Window | Parent checkpoint | Replays through |
| --- | --- | --- |
| `obs-4877` | OBS-only `main-4000.pt` | 5000 |
| `mixed-6676` | mixed `main-6000.pt` | 6800 |
| `mixed-3308` (if needed) | mixed 2,000-update `main-2000.pt` | 3400 |

`fork_replay.py` rewrites a copy of the parent for each arm. It permits code
differences only in the files that add the probe and the interventions, and
config differences only in the listed interventions, then records everything
in `lineage.json`. The **probe** (`samudra.observations.probes`) logs, per update:

- gradient and update norms for each network component
- each example's window, corruption seed and loss parts
- rollout diagnostics: the RMS step-to-step surface change for each of the 12
  steps, the within-day range, and for LLC the same statistics of the truth
- every 50 updates in the control, raw and EMA weights

A CPU test checks that enabling the probe leaves every weight bit-identical.
`stability.py` then measures, for each saved weight snapshot, how much the
12-step rollout amplifies a perturbation (linearized, by forward-mode
differentiation) and the largest singular value of the 12-step propagator.

**Stage 1: reproduce, then vary the data order.** Arms `control`, `reseed-1`
and `reseed-2` for `obs-4877` and `mixed-6676`. The reseeded arms resume the
same weights and optimizer state but draw different training windows.

**Stage 2: one intervention each,** from the same weights and data order as the
control: `clip-0.3` (clip at roughly the 99th-percentile normal gradient norm),
`beta2-0.95` (Adam's second moment adapts faster), `guard-5x` (skip an update
whose loss exceeds 5× the running median of the last 50; fail after 20
consecutive skips), and `lr-5e-5`.

## Pre-registered predictions and decision rules

1. **Reproduction gate.** `check_reproduction.py` finds zero difference between
   the control and the original run in gradient norm, loss parts and validation
   score at every update. If the control diverges before the spike, the replay is a
   new trajectory: we report the divergence, and Stage 2 compares arms only
   statistically, across reseeds.
2. **Weights or data.** If the hypothesis holds, the reseeded arms also spike
   within about 150 updates of the original, because the instability is in the
   weights. If only the control spikes, a specific batch triggers it; the probe's
   per-example records then identify the windows, and we inspect those dates.
3. **Mechanism.** If the hypothesis holds, in the control the OBS-task tangent
   growth at step 12 and the propagator norm rise over the 100–300 updates before
   the spike, relative to the window's first snapshots, while LLC-task values
   stay flat. The rollout diagnostics show the step-to-step surface change
   growing with step k during the burst. If the stability measures stay flat
   until the burst update, the hypothesis is wrong.
4. **Interventions.** If the hypothesis holds: clipping shrinks the burst
   but does not prevent it, because Adam rescales the step anyway; the lower
   learning rate delays or prevents it; the guard skips repeatedly and fails
   (itself evidence that the weights are the cause); β₂ = 0.95 reduces the
   damage afterwards. An intervention *fixes* a window if no update's OBS
   forecast loss exceeds 10× its trailing 50-update median, and the validation
   score at the window's end is no worse than the control's last score before
   the spike.

Replays test fixes locally, in a window where we know a spike happens. They do
not show whether a fix holds from scratch, or what it costs in forecast skill.
That needs from-scratch runs with two seeds, which need separate approval.

## Cost

These are estimates from #1037's measured throughput on four GB200s
(2.5 s per OBS-only update, 4.7 s per mixed update), plus validation every 100
updates: about 4 GPU-hours per `obs-4877` arm and 5 per `mixed-6676` arm.

| Stage | Runs | GPU-hours |
| --- | --- | --- |
| 1 | 3 arms × 2 windows | ≈ 28 |
| 2 | 4 interventions × 2 windows | ≈ 37 |
| Stability analysis | 2 controls × 20 snapshots | ≈ 2 |

The control's weight snapshots take about 1.5 GB each, about 30 GB per window,
kept on EAI and not in git.

## Running

```bash
# On Beta, with access to ny_lz1955_multiscale.
export SPIKE_RUN_ROOT=/projects/ny/lz1955/multiscale/<user>/runs/spike-replay-<date>
export SPIKE_CODE=$SPIKE_RUN_ROOT/code/<commit>   # git archive of this branch
sbatch experiments/llc-obs-spike-diagnosis/replay.sbatch obs-4877 control
```

## Status

- 2026-11-25: tooling written and CPU-tested (`tests/test_spike_probe.py`).
  Waiting for Beta access to `ny_lz1955_multiscale`. `forecast_oscillation.py`
  is written against the export code and has not yet run on Jesse's bundle.
