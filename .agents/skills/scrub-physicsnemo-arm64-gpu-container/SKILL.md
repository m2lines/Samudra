---
# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

name: scrub-physicsnemo-arm64-gpu-container
description: Nightly local ARM GPU scrub that pulls the latest main PhysicsNeMo arm64 container and runs CUDA-marked tests outside GitHub Actions.
schedule_cron: "0 17 3 ? * *"
schedule_tz: America/New_York
---

# scrub-physicsnemo-arm64-gpu-container

Use this skill for the nightly ARM GPU container validation scrub.

## Purpose

- Validate the latest `main` ARM64 PhysicsNeMo container on a real local ARM machine with a modern NVIDIA GPU.
- Keep this outside GitHub Actions. Cloud ARM GPU instances currently expose unsupported NVIDIA T4G-class hardware for this stack.
- Exercise the GPU path that CI cannot cover cheaply: pull the latest `25.11-arm64-latest` image and run CUDA-marked tests locally.

## Required Host

- `aarch64` Linux host.
- NVIDIA GPU visible through `nvidia-smi`.
- Docker usable by the current user, or through non-interactive `sudo docker`, with NVIDIA container runtime support.
- Repository checkout available with this script from `main`.

## Run

From the repository root:

```bash
scripts/container/scrub_arm64_gpu_container_tests.sh
```

The script defaults to:

```text
ghcr.io/open-athena/ocean-emulator-physicsnemo:25.11-arm64-latest
```

Useful overrides:

```bash
IMAGE_TAG=ghcr.io/open-athena/ocean-emulator-physicsnemo:25.11-arm64-<sha> \
PYTEST_ARGS="-x -k test_trainer" \
scripts/container/scrub_arm64_gpu_container_tests.sh
```

## Expected Result

- Docker pulls the latest ARM64 image published from `main`.
- `scripts/container/run_cuda_tests_in_image.sh` runs the CUDA-marked tests inside the container, including the FOMO flash-attention coverage.
- A successful scrub exits `0` and ends with `HARNESS_SCRUB_LOOP {"needs_followup_at":null}`.

## Failure Handling

- If pull fails, check whether `main` has published `25.11-arm64-latest` and whether GHCR auth is available.
- If Docker/NVIDIA runtime is missing, capture the host setup blocker explicitly.
- If CUDA tests fail, inspect the failing test and either fix the regression in a PR or file/update a concrete issue with the image tag, commit, GPU model, and failing traceback.
- Use scrub follow-up only when waiting on an external image publish or another long-running validation; otherwise complete the run in one turn.
