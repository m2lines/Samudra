<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Launch and qualification, 2026-10-17

Training job **214219** was submitted and released to Beta's production queue
with four GPUs, 144 CPUs, full host memory, standard QoS, and a 24-hour limit.
It trains for 70 epochs from scratch with the configuration in this experiment.
At submission, it was pending; no completed training result is claimed here.

Executed source: [79fcd103c](https://github.com/m2lines/Samudra/commit/79fcd103cb3794e27a8d9dbd1541f4c843d38143),
based on main [f1e713220](https://github.com/m2lines/Samudra/commit/f1e7132209ea5d4839b5b2c37f078f2c276c3f5f).
Later commits adding this run record do not change the archived source used by
the job. The native extension's Rust source matches that source exactly; its
verified ARM64 build was copied from the earlier current-Samudra Beta control.

## Dataset readiness

The full CM4 data-store transfer was verified by download comparison on
2026-10-16: 131,466 objects, 140,827,879,002 stored bytes, no differences.
The complete 14,600-index chunk sets of all nine physical variables were also
checked before this launch. Both full-record normalization stores were copied
on 2026-10-17 and independently compared by full download: 36 matching files
each, no differences. All three stores reside together in project storage.

## Qualification

Job **214211** completed with exit `0:0` in 7 minutes 32 seconds on
`b3-11-s1-dgx-03-c01`, with one NVIDIA GB200 GPU. Most startup time was the
container import; maximum reported step RSS was 7,978,688 KiB.

- All 81 canonical channels matched Xarray exactly, including NaN placement,
  at train positions `[5839, 1, 0, 1]` and validation positions `[1459, 1, 0, 1]`.
  These exercise slice offsets, split endpoints, ordering, and duplicate reads.
- Native reads were used for both training and ordinary validation.
- Split sizes were 5,840 / 1,460 / 7,300 records, with the expected disjoint
  no-leap dates. Only timestamps were inspected for the test period.
- Every normalization mean and standard deviation was finite; every std was
  positive.
- The actual training path completed one optimizer update with all four
  autoregressive calls and `wfo` forcing. Ordinary validation and a configured
  360-day validation rollout completed with finite scalar metrics.

No CM4-specific Rust loader changes were required. These are execution and
reader-parity checks, not trained-model skill results. Local validation also
passed 106 CM4/Rust tests and all pre-commit hooks.

## Tracking and follow-up

W&B is configured online in `ocean_emulators/default`; its run URL will be
recorded after actual training startup. An authenticated completion hook is
installed in the batch script, with compute-host connectivity checked before
training. A durable two-hour fallback reminder and quiet monitoring agent also
check scheduler/accounting state, covering termination before the batch hook can
run. Monitoring reports verified startup, completion, or an actionable failure.

The persistent run directory is
`/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017`.
It retains the exact source archive/hash, native binary hash, expanded training
config, full qualification report, allocation records, logs, and checkpoints.
