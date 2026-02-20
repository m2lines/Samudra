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

The CI workflow publishes images on `main` pushes and manual dispatches, then runs
CPU/GPU tests from the published image on an EC2 GPU runner.

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
- If `apptainer` (or `singularity`) is installed locally, the wrapper script also
  exports a SIF from the built Docker image.
- CUDA tests are run with Docker GPU runtime flags plus `--ipc=host` and raised
  ulimits to avoid DataLoader shared-memory failures.
- `zarr` is installed from this repo's lockfile rather than inherited from the
  base image.
