# PhysicsNeMo 25.11 Container Flow

Build a project image on top of NVIDIA PhysicsNeMo `25.11`, verify compatibility of
image-provided heavy packages, and run a smoke test:

```bash
scripts/container/build_physicsnemo_25_11.sh
```

Run CUDA tests in the built image:

```bash
scripts/container/run_cuda_tests_in_image.sh
```

On CI, GitHub Actions builds and validates this image via:

```text
.github/workflows/container-physicsnemo.yml
```

The CI workflow smoke-tests the x86_64 Dockerfile on every PR. On `main` pushes and
manual dispatches, it publishes the x86_64 image to GHCR, runs CPU/GPU tests from
that published image on an x86_64 EC2 GPU runner, and separately launches a non-GPU
ARM64 EC2 runner to build, publish, pull, and CPU-test an ARM64 image. ARM64 GPU
testing is intentionally not attempted on cloud runners because the available small
ARM GPU instances expose older NVIDIA T4G GPUs that are unsupported by the current
PyTorch/CUDA stack.

The scheduled ARM GPU scrub is defined as a repo-local harness scrub skill, not a
GitHub Actions workflow:

```text
.agents/skills/scrub-physicsnemo-arm64-gpu-container/SKILL.md
```

That scrub runs on the local ARM GPU host, pulls
`ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-arm64-latest` from `main`, and
runs the CUDA-marked tests with:

```bash
scripts/container/scrub_arm64_gpu_container_tests.sh
```

The scrub host must allow the harness user to run Docker directly or through
non-interactive `sudo docker`.

Published image tags:

```text
25.11-<sha>                  # existing x86_64 compatibility tag
25.11-x86_64-<sha>
25.11-arm64-<sha>
25.11-latest                 # existing x86_64 compatibility tag from main
25.11-x86_64-latest
25.11-arm64-latest
25.11-manual-<ref>           # existing x86_64 compatibility tag from workflow_dispatch
25.11-x86_64-manual-<ref>
25.11-arm64-manual-<ref>
```

Useful environment variables:

```bash
IMAGE_TAG=ocean-emulator:physicsnemo-25.11
DOCKERFILE=containers/Dockerfile.physicsnemo-25.11
BUILD_APPTAINER=1
SIF_PATH=dist/ocean-emulator_physicsnemo-25.11.sif
PYTEST_MARK_EXPR="cuda and not manual"
PYTEST_ARGS="-k test_trainer"
```

Notes:

- The Docker build keeps `torch`, `torchvision`, and `flash-attn` from the base image.
- The build creates `.venv` with `--system-site-packages`, then uses `uv sync` for
  the rest of the dependencies (including dev dependencies; this is a dev container).
- The build script refuses to run from a dirty git checkout (including untracked files);
  commit and push first so baked git metadata matches image contents.
- If `apptainer` (or `singularity`) is installed locally, the wrapper script also
  exports a SIF from the built Docker image.
- CUDA tests are run with Docker GPU runtime flags plus `--ipc=host` and raised
  ulimits to avoid DataLoader shared-memory failures.
- `zarr` is installed from this repo's lockfile rather than inherited from the
  base image.
