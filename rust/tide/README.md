# tide

`tide` is the experimental Rust sidecar loader for Ocean Emulator training. It is
installed as an optional Python extension through `uv` using the path dependency
in the repository root `pyproject.toml`.

## Current State

What works:

- Builds as a `maturin`/PyO3 extension named `tide`.
- Can be instantiated through the Python training config via
  `data.loader_version: om4-rust-v0` for raw Tide/JAX experiments.
- Supports raw batch construction for the standard single-scale Samudra
  train/validation schedule.
- Uses the existing Python sampler and batch ordering; Rust receives concrete
  per-batch example indices.
- Reads local filesystem Zarr stores and Zarr metadata directly from Rust.
- Builds a semantic SSA graph per batch and materializes requested step roots on
  demand.
- Caches materialized SSA values and deduplicates concurrent in-flight requests
  for the same value.
- Exposes raw packed step-0 prognostic/boundary/label and later-step
  boundary/label tensors for the JAX frontend.
- Keeps the Rust SSA graph raw-only: no mask, normalization, or input-concat ops
  are currently encoded as Tide IR nodes.
- Keeps normalization metadata, masks, and fill values out of Tide batch objects;
  the Samudra-specific `samudrax` helper receives the existing Python train
  dataset/source and builds that metadata on the Python/JAX side.
- Returns regular CPU NumPy arrays across the FFI boundary; the Python shim can
  keep raw torch tensors on CPU or move them to the configured torch device
  before JAX conversion.
- Has Python parity coverage for raw Tide plus `samudrax` mask/normalization
  against the existing torch loader on mock OM4 data, plus a regression test for
  mapping sliced train/validation times back to the full backing Zarr store
  indices.
- Has an experimental dev-only JAX frontend that traces small JAXPR builder
  functions, materializes raw Tide leaves through the Python API, and evaluates
  `samudrax` mask/normalization plus ordinary JAX tensor ops on CPU or CUDA when a
  compatible JAX wheel is installed.
- The JAX frontend treats non-Tide JAXPR regions as opaque blobs and can choose
  whether to run each blob on CPU or on the selected JAX device. This lets plain
  JAX code such as slicing stay in the user expression without adding a
  Tide-specific crop operator yet.

What does not work yet:

- No prepared Tide-via-torch batch API; Tide exposes raw leaves for the JAX
  frontend only.
- No GPU decode, GDS, KvikIO, or direct-to-device reads.
- No pinned-memory allocation.
- No GPU memory budget, no pinned-memory budget, and no meaningful CPU memory
  budget enforcement yet.
- No cross-batch, cross-phase, or cross-epoch prefetch.
- No object-store/S3 support.
- No FOMO, multiscale `match`/`mix`, compact datasets, or inference support.
- The frontend does not yet introspect opaque blobs to recognize slices or push
  spatial windows into Rust chunk enumeration automatically.
- No integration into the main package wheel; this remains a sidecar extension.
- No read/decode concurrency controls or memory accounting are implemented yet.

## Recent Local Results

On `/home/jder/data/om4_onedeg`, an earlier real-data parity check matched the
existing torch loader for step 0 input/label and step 1 input/label after fixing
time-index mapping for train windows that start after the backing Zarr store
start. The comparable path now is raw Tide plus `samudrax` mask/normalization; the
prepared Tide-via-torch API has been removed.

On the same local one-degree dataset, a 20-batch CUDA benchmark with batch size
4, two rollout steps, no tide prefetch, and explicit `torch.cuda.synchronize()`
around each timed batch produced:

| Case | Mean batch s | Mean excluding first s | Median s | Batch/s |
| --- | ---: | ---: | ---: | ---: |
| torch workers=0 | 0.1592 | 0.1539 | 0.1515 | 6.28 |
| torch workers=4 | 0.0496 | 0.0359 | 0.0290 | 20.15 |
| tide v0 | 0.1241 | 0.1231 | 0.1234 | 8.06 |

This means `tide` currently beats the single-process torch loader on this small
local case, but the existing multiprocessing torch loader is still substantially
faster. That is expected for v0 because `tide` has not yet implemented real
read/decode concurrency, pinned memory, GPU decode, GPU residency, or cross-batch
prefetch.

On the same one-degree dataset, the experimental JAX frontend also runs on the
local GB10 GPU with a manually installed CUDA JAX wheel:

```bash
uv pip install --python .venv/bin/python 'jax[cuda13]==0.9.2'
uv run --no-sync python scripts/bench_tide_jax_frontend.py \
  --data-root /home/jder/data/om4_onedeg \
  --backend cuda \
  --batches 5 \
  --batch-size 2 \
  --jax-crop 45,135,90,270
```

With one warmup batch outside the timed loop, the current opaque-blob CUDA
benchmark produced these local results:

| Case | Setup s | Mean batch s | Mean excluding first s | Median s | Batch/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| torch workers=0 | 0.1593 | 0.1013 | 0.0755 | 0.0755 | 9.87 |
| tide jax cpu-blob->device | 4.6652 | 0.0633 | 0.0612 | 0.0622 | 15.79 |
| tide jax device-blob | 0.6430 | 0.0582 | 0.0601 | 0.0597 | 17.19 |

The current benchmark compares opaque blob placement: the CPU-blob case
materializes Tide leaves and runs the JAX blob on CPU before moving outputs to
the selected JAX device; the device-blob case moves Tide leaves to the device
before running the same opaque JAX blob. The CUDA JAX install is not encoded in
the root lockfile yet because the available CUDA13/aarch64 JAX wheel currently
pulls in NumPy 2.x, while the project still pins NumPy below 2.

## Development

The root project includes:

```toml
[tool.uv.sources]
tide = { path = "rust/tide" }
```

Useful checks:

```bash
uv sync --dev --extra cuda
uv run --no-sync pytest tests/test_rust_loader.py -q
uv run --no-sync pytest tests/test_tide_jax.py -q
PYO3_PYTHON=/home/jder/Ocean_Emulator/.venv/bin/python cargo test --manifest-path rust/tide/Cargo.toml --lib
uv lock --check
```

## Next TODOs

- Use the repeatable `scripts/bench_tide_jax_frontend.py` harness to collect
  comparable Tide/PyTorch/JAX frontend measurements on larger local datasets.
- Add half-degree and quarter-degree local benchmarks once local fixture data is
  available.
- Implement real read/decode concurrency and add config once it is enforced.
- Add explicit memory accounting and enforce CPU/pinned/GPU budgets.
- Move the FFI boundary toward torch tensors, pinned buffers, or a GPU JAX/DLPack
  path to avoid avoidable host-side copies.
- Add GPU-side decode and explicit direct-I/O behavior with no silent CPU
  fallback.
- Implement within-batch prefetch in the Rust/JAX scheduler rather than
  reintroducing a Python thread shim.
- Add coarse opaque-blob characterization, starting with static slices, so Tide
  can decide whether to push a crop into Rust loading or run the blob on device.
- Extend the semantic IR/interpreter for repeated cross-step data reuse.
- Extend support beyond standard Samudra train/validation once the narrow path is
  stable and measurably faster.
- Implement important optimizations like batch-aware eviction and auto tuning
  between CPU-blob and device-blob strategies.
