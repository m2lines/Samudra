# tide

`tide` is the experimental Rust sidecar loader for Ocean Emulator training. It is
installed as an optional Python extension through `uv` using the path dependency
in the repository root `pyproject.toml`.

## Current State

What works:

- Builds as a `maturin`/PyO3 extension named `tide`.
- Integrates with the Python training config via `data.loader_version:
  om4-rust-v0`.
- Supports the standard single-scale Samudra train/validation path.
- Uses the existing Python sampler and batch ordering; Rust receives concrete
  per-batch example indices.
- Reads local filesystem Zarr stores and Zarr metadata directly from Rust.
- Builds a semantic SSA graph per batch and materializes requested step roots on
  demand.
- Caches materialized SSA values and deduplicates concurrent in-flight requests
  for the same value.
- Provides step-demand access from Python:
  `step0()`, `get_label(step)`, and
  `merge_prognostic_and_boundary(prev_prediction, step)`.
- Loads, packs, normalizes, masks, and concatenates step-0 input in Rust.
- For later autoregressive steps, Rust returns boundary plus label; Python
  merges the prior model prediction with the boundary tensor.
- Returns regular CPU NumPy arrays across the FFI boundary; the Python shim moves
  them to the configured torch device.
- Has Python parity coverage against the existing torch loader on mock OM4 data,
  plus a regression test for mapping sliced train/validation times back to the
  full backing Zarr store indices.

What does not work yet:

- No GPU decode, GDS, KvikIO, or direct-to-device reads.
- No pinned-memory allocation.
- No GPU memory budget, no pinned-memory budget, and no meaningful CPU memory
  budget enforcement yet.
- No cross-batch, cross-phase, or cross-epoch prefetch.
- No object-store/S3 support.
- No FOMO, multiscale `match`/`mix`, compact datasets, inference, or spatial
  subset support.
- No integration into the main package wheel; this remains a sidecar extension.
- The concurrency settings exist in config, but v0 still needs real tuning and
  enforcement work around read/decode concurrency and memory accounting.

## Recent Local Results

On `/home/jder/data/om4_onedeg`, a direct real-data parity check matched the
existing torch loader for step 0 input/label and step 1 input/label after fixing
time-index mapping for train windows that start after the backing Zarr store
start.

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
PYO3_PYTHON=/home/jder/Ocean_Emulator/.venv/bin/python cargo test --manifest-path rust/tide/Cargo.toml --lib
uv lock --check
```

## Next TODOs

- Add a repeatable benchmark harness instead of ad hoc inline benchmark scripts.
- Add half-degree and quarter-degree local benchmarks once local fixture data is
  available.
- Implement real read/decode concurrency and connect it to the existing config
  knobs.
- Add explicit memory accounting and enforce CPU/pinned/GPU budgets.
- Move the FFI boundary toward torch tensors or pinned buffers to avoid avoidable
  host-side copies.
- Add GPU-side decode and explicit direct-I/O behavior with no silent CPU
  fallback.
- Implement within-batch prefetch in Rust rather than relying on the current
  Python thread shim.
- Extend the semantic IR/interpreter for spatial subsetting and repeated
  cross-step data reuse.
- Extend support beyond standard Samudra train/validation once the narrow path is
  stable and measurably faster.
