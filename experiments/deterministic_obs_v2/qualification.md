<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Deterministic observation qualification, 2026-11-03

All five model arms passed the original bounded GPU qualification on Torch. Its
loader spends 70–80% of replay-observation update time and 96–97% of OM4 update
time in data preparation/loading. The subsequent Rust/prefetch/DDP work is tracked
in the [scaling qualification](scaling.md). Full scientific training
has not been submitted; these six-update checks do not establish forecast skill.

The implementation uses the Samudra configuration harness from
[PR #955](https://github.com/m2lines/Samudra/pull/955), a full Samudra evolution
U-Net and direct conditioning on eight ERA5 fields. The [experiment design](README.md)
defines the observation products, losses, initializer choices and selection rule.
All arms accumulate eight monthly examples per optimizer update. Each example
forecasts six or seven five-day steps, weighted by calendar overlap for monthly
targets. Replay supplies detached two-frame ocean states from an earlier model
forecast and continues them through the next month. A–D use four fresh and four
replay examples; E uses eight fresh examples. Mixed arms alternate
observation and OM4 updates during main training and use observations throughout
cooldown. The small initializer has widths 128/192/256/384 and the large one
280/380/480/520; the evolution U-Net always uses the latter widths.

Array `21786733` completed all five tasks with exit code zero on NVIDIA RTX PRO
6000 Blackwell Server Edition GPUs. Each task used one GPU, eight CPU cores and
96 GiB host memory under preemption-only routing. No other GPU family was
qualified in this attempt. Code was frozen at
`024177bc7b612042275381f3470d37e795e5d396`, with the dependency-matched
PhysicsNeMo container from `4f86be45a09cb9eba1b51b6273e0189941b02495`.

All arms passed exact next-update checkpoint restoration, including model,
optimizer, EMA, RNG and replay state across the cooldown boundary. Uncooled
checkpoint forks remained isolated. Every annual validation completed 73 steps
and 12 complete months, scored 29 observed channels and produced finite metrics.
The no-replay arm retained empty replay pools. Arm D also passed twelve-month
model-generated replay chains for both sources, independently checked global
normalization moments and exact daily-versus-materialized sample parity for
1993-05 and 2001-01. This tests explicit checkpoint restoration; a forced Slurm
preemption/requeue cycle and cross-GPU numerical equality were not exercised.

The [receipts](receipts/qualification-v2/) preserve configs, manifest hashes,
complete metrics, replay chains, timing records and Slurm accounting. Full
manifests and checkpoints remain under
`/scratch/jr7309/deterministic-obs-v2/qualification-v2/` on Torch.
W&B runs are linked
in the table. The times below exclude the first update and count the repeated
resume-check update once. Mixed arms supply only two measured OM4 updates each;
the ranges are short-run observations, not statistical confidence intervals.

| Arm and W&B run | Main data / initializer | Parameters | Obs seconds/update | OM4 seconds/update | Peak allocated GPU GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| [A](https://wandb.ai/ocean_emulators/observational-transfer/runs/6pbyhxj2) | obs / small | 117.5M | 32.7–39.7 | — | 5.15 |
| [B](https://wandb.ai/ocean_emulators/observational-transfer/runs/ulh92r5z) | mixed / small | 117.5M | 26.8–28.0 | 147.4–149.8 | 5.54 |
| [C](https://wandb.ai/ocean_emulators/observational-transfer/runs/v2iz3dko) | obs / large | 169.8M | 28.3–29.7 | — | 6.43 |
| [D](https://wandb.ai/ocean_emulators/observational-transfer/runs/y6mc7gkn) | mixed / large | 169.8M | 28.1–28.9 | 115.1–149.5 | 6.61 |
| [E](https://wandb.ai/ocean_emulators/observational-transfer/runs/evlr19jg) | mixed / large, no replay | 169.8M | 11.9–12.2 | 144.5–145.2 | 6.53 |

A's larger loading delay makes it slower than C despite its smaller initializer.
Different nodes, cache state and concurrent filesystem traffic confound these
timings. They do not establish an architecture speed ranking. The timed loading
block includes decoding, aggregation and device transfer; it is not a measured
GPU utilization percentage or a breakdown of disk versus CPU cost.

Annual validation, including an improved-best checkpoint, took 23.0–25.3 seconds.
Ordinary checkpoint writes took 2.81–4.21 seconds; restores took 1.49–2.06 seconds.
D's checkpoint with full replay capacity and populated Adam moments was 4.17 GiB
and took 5.68 seconds to write. Slurm reported peak process RSS of 29.3–52.6 GiB
across the qualification tasks, which include CPU checkpoint comparisons.

The [runtime estimates](receipts/qualification-v2/runtime-estimates.json) use the
measured per-source minimum and maximum, checkpointing every 25 updates and
annual validation every 250 updates. Each screen has 4,000 main updates plus
1,000 observation-only cooldown updates: 5,000 observation updates for A/C, or
3,000 observation plus 2,000 OM4 updates for B/D/E. Validation timing already
includes its best-checkpoint save, which is counted once. Full replay-capacity
checkpoint cost is measured only for D; the other receipts contain smaller pools.

| Arm | Screen GPU-hours, observed timing range | Screen GPU-hours, with 30% allowance | Additional GPU-hours if selected, with 30% allowance |
| --- | ---: | ---: | ---: |
| A | 45.7–55.4 | 59.4–72.1 | 178.1–216.2 |
| B | 104.6–106.8 | 136.0–138.8 | 407.9–416.5 |
| C | 39.6–41.6 | 51.5–54.1 | 154.4–162.4 |
| D | 87.8–107.6 | 114.1–139.8 | 342.4–419.5 |
| E | 90.5–91.1 | 117.6–118.4 | 352.8–355.3 |

The five screens total 369–403 GPU-hours rounded up, or 479–524 with the explicit
30% planning allowance. Parallel arms can shorten elapsed time; a screen's
GPU-hours equal its execution hours on one GPU. Queue delays, unknown preemption
frequency and changes in long-run replay/data-access behavior are excluded. The
allowance is a planning choice, not measured uncertainty coverage.

The winner adds 12,000 main updates and three separate 1,000-update cooldowns
after its screen. Its total is 16,000 main plus four 1,000-update cooldowns,
including the screen's cooldown. For D, that is 456.5–559.3 GPU-hours including
the allowance, or 19.0–23.3 days if all work executes serially on one GPU.
Its remaining work after the screen is 342.4–419.5 GPU-hours. Intermediate
cooldown branches can run alongside the continuing main trajectory.

The initial qualification led to these follow-up decisions. The
[scaling work](scaling.md) implements the updated loader and DDP choice:

1. Prepare a complete, verified float32 OM4 frame cache for the current 2,812
   native five-day intervals, whose bounds extend from 1975-01-26 through
   2013-08-04, preserving existing calendar gaps. Also cache observation
   five-day aggregates with their exact original interval alignment. The OM4
   physical arrays would occupy 2,812 × 77 channels × 180 × 360 cells × 4 bytes,
   about 56.1 GB uncompressed. Existing caches must not reduce the eligible window pool or
   change targets, masks or normalization. The observation cache is now prepared;
   OM4 currently uses direct Rust reads, so an additional OM4 cache remains an
   option if measured throughput requires it.
2. Qualify eight-way DDP together with asynchronous reads. The user requested
   eight RTX GPUs and measured utilization above 50%. Keep the global batch at
   eight to preserve the optimizer-update budget.
3. Set checkpoint cadence from the improved timings. Current writes are cheap,
   but 25 updates can represent roughly 34 minutes of mixed training. A roughly
   ten-minute checkpoint interval and a forced scheduler-interruption check
   would better qualify long preemptible execution.
4. Decide whether confirmation needs additional seeds or validation origins.
   The implemented screen uses seed 1729 and one annual origin, 2013-11-01.
   The five-arm screening budget and 16k winner extension do not include extra
   seeds or an independent test-origin campaign.

Successful qualification consumed 2,895 allocated GPU-seconds (0.804 GPU-hours).
Including the [failed logging attempt](receipts/attempt-21785317.json), the total
was 0.891 GPU-hours. CPU data preparation and container preparation are separate.
Both callbacks for the original qualification are retired. Later scaling jobs
are recorded separately.
