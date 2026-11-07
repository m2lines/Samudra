<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# First-wave submission, November 7, 2026

The first wave is running on NYU Torch. GPU qualification passed and the first two
array arms produced complete artifacts. Remaining tasks were quota-limited at the
launch check; there is no reliable completion ETA yet. These are engineering
checks, not results about task competition.

| Receipt | Value |
| --- | --- |
| Training source | [90ef3a7c011f8bf2c4d2d012f8ce3f8bcb9155cf](https://github.com/m2lines/Samudra/commit/90ef3a7c011f8bf2c4d2d012f8ce3f8bcb9155cf) |
| Draft experiment PR | [967](https://github.com/m2lines/Samudra/pull/967) |
| Qualification | Torch job 22199882, COMPLETED 0:0, 23 seconds |
| Main array | Torch 22199995, indices 0-222, concurrency limit 4 |
| Array completion callback | CPU job 22200213, afterany on array 22199995 |
| Run root | `/scratch/jr7309/precision-competition-20261107-90ef3a7c0` |
| W&B group | `precision-competition-20261107`, project `ocean_emulators/samudra` |
| First completed arm | [Auxiliary-only control](https://wandb.ai/ocean_emulators/samudra/runs/pc-a8aac7297a7a042c) |

All seven qualification cases reached 30 updates with finite evaluation metrics.
The device was an NVIDIA RTX PRO 6000 Blackwell. Peak allocated GPU memory was
2.81 GiB; measured update times ranged from 0.0048 to 0.0481 seconds, with FP64
slowest. The pilot does not establish sustained end-to-end throughput for every
configuration. The first two full arms reached their endpoints, with about 28
seconds of measured training combined; evaluation and startup are additional.

The immutable code tar SHA-256 is
`af1eff19fd53534f06229f9cffb8f4954578895cf0e0ced3becccb809e46fc29`.
The manifest SHA-256 is
`d6f969ac4ca43f10cbe80639992f4fd1e5463e71351d314f01cabe5fcd18067e`.
The code tar was verified on staging and made read-only; each job verifies the
manifest before execution. The source commit is fixed even if this branch later
adds reports or alternative launchers.

Torch uses the existing container
`/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif`.
Observed runtime: Python 3.12.3, PyTorch
`2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2. The job account is
`torch_pr_347_lzanna`; the preemption/requeue comment enables broad compatible GPU
routing, including L40S. The first observed pending reason was `QOSGrpGRES`.

Engaging authentication was refreshed and verified. An existing Engaging container
had the same Python/PyTorch/CUDA versions. Its advanced-account qualification trial,
28226260, remained queued and was cancelled before starting (CANCELLED, zero elapsed).
No main training was submitted there. The Engaging launcher is retained as a
contingency. The trial did not consume GPU runtime.

Ten focused CPU tests and all repository pre-commit checks passed before submission.
GPU checks establish execution and memory feasibility, not convergence or scientific
validity of the hypothesis. Training caps and deferred comparisons are specified in
[the protocol](README.md). Monitor the exact matched-step, seed-paired penalties and
auxiliary accuracy before interpreting the screen. Allocated GPU-hours must include
qualification, failed attempts, validation and startup separately from measured
training time.
