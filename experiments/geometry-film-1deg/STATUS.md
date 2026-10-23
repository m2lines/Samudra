<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Submission status: 2026-10-23

All three scratch runs are released into the production Beta queue. No training
results or GPU qualification results exist yet.

| Arm | Slurm job | State at submission verification |
| --- | --- | --- |
| Control | 216666 | PENDING |
| Input concatenation | 216667 | PENDING |
| Geometry FiLM | 216668 | PENDING |

All jobs execute source commit `4462fa1f4` (model implementation `05c490f55`),
with source/native SHA-256 read-back verification. Each requests one full Beta
node: 4 GPUs, 144 CPUs, full memory, standard QoS, 24-hour limit. The earlier held
jobs 216656–216658 were cancelled before allocation to replace their copied
batch scripts with the credential-free monitoring harness.

Local checks: 45 tests passed across geometry, metric scoring, blocks, periodic
boundaries, gradient detaching and post-training evaluation (2 skipped,
10 expected failures). All commit hooks passed, including mypy, Ruff, config
schemas, secret scanning and REUSE. Full-shape GPU qualification runs at the
start of each job and must pass before production training.

Three quiet local monitors are registered with six-hour scheduler checks; no
remote callback credentials were transferred. Completion/failure detection can
lag by six hours. No completion ETA is set while queue timing is uncertain.

Runtime and log paths, evaluation criteria and the shared date-split correction
are documented in README.md. The branch is committed locally. Automatic approval
review blocked pushing it to the public repository because it includes cluster
and storage paths; draft PR publication awaits explicit user approval.
