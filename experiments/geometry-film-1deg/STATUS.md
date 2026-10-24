<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Completed: 2026-10-24

All three scratch runs and all six final evaluations completed successfully.
See [results](results/RESULTS.md) for the matched comparison.

| Arm | Slurm job | Final state |
| --- | --- | --- |
| Control | 216666 | COMPLETED, exit 0:0 |
| Input concatenation | 216667 | COMPLETED, exit 0:0 |
| Geometry FiLM | 216668 | COMPLETED, exit 0:0 |

All jobs execute source commit `4462fa1f4` (model implementation `05c490f55`),
with source/native SHA-256 read-back verification. Each requests one full Beta
node: 4 GPUs, 144 CPUs, full memory, standard QoS, 24-hour limit. The earlier held
jobs 216656–216658 were cancelled before allocation to replace their copied
batch scripts with the credential-free monitoring harness.

Local checks: 45 tests passed across geometry, metric scoring, blocks, periodic
boundaries, gradient detaching and post-training evaluation (2 skipped,
10 expected failures). All commit hooks passed, including mypy, Ruff, config
schemas, secret scanning and REUSE. Full-shape GPU qualification passed before production training in every arm.

All three monitoring registrations are retired after completion verification.
No remote callback credentials were transferred.

Runtime and log paths, evaluation criteria and the shared date-split correction
are documented in README.md. This is an experiment/results record, not intended
to merge.
