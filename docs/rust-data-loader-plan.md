<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: Apache-2.0
-->

# Rust data loader shipping plan

## Goal

Ship an opt-in, local-only `loading.type: rust` training path that:

- reads flat OM4 Zarr stores with Rust;
- uses the existing Python batch samplers unchanged;
- uses no PyTorch data-loading worker processes;
- loads complete batches matching the `RawTrainData` contract;
- bounds and overlaps host-side batch prefetch with model execution; and
- on CUDA, overlaps pinned-memory transfer and batch preparation with the current
  model step.

The first shipped path is deliberately narrow. It proves end-to-end correctness and
performance on flat OM4 before adding more physical dataset layouts.

## Design boundaries

### Python remains responsible for

- sampling, shuffle order, epoch seeding, batching, and DDP rank partitioning;
- translating a global `ConcatDataset` index into a dataset and local index;
- constructing `RawTrainData` and `TrainData` objects;
- normalization, masking, channel flattening, and PyTorch device/stream semantics;
- coordinating loader lifetime with the training and validation loops.

### Rust is responsible for

- keeping local Zarr stores and arrays open across batches;
- translating logical flat-OM4 variables and time indices to physical Zarr reads;
- issuing bounded concurrent reads;
- writing directly into complete batch-shaped output buffers; and
- returning contextual read, shape, dtype, and index errors without panics.

OM4-compact and LLC will each get direct translation code when they are added. We
will not introduce a canonical-manifest abstraction.

### Explicit non-goals

- S3 or another remote object store;
- changing `dataset_id`, heterogeneous-batch semantics, or sampler grouping;
- replacing the existing CPU or GPU loading paths;
- moving sampling or DDP scheduling into Rust;
- linking the Rust extension to libtorch or CUDA;
- making the Rust path the default before parity and production measurements pass.

The existing `RawTrainData.dataset_id` invariant remains in force: every emitted
batch must contain samples with the same dataset ID. The Rust loader will check this
and fail with a useful error. Generalizing that contract is separate work.

## Target pipeline

```text
existing Python batch sampler
              |
              v
     batch of global indices
              |
              v
 bounded Rust host prefetch -> pinned RawTrainData batch
                                      |
                                      v
                       PyTorch CUDA prefetch stream
                                      |
                                      v
                  normalized and masked TrainData
                                      |
                                      v
                            training stream
```

The prefetch queue preserves sampler order. Each DDP rank owns an independent Rust
reader and queue; the normal training process is the only process on that rank.

## Stage 0: lock down the baseline

Record correctness fixtures and end-to-end timings for the current CPU loader on a
small local flat-OM4 store and a representative production-sized local store.

### Exit criteria

- Fixture cases cover `hist` 0 and 1, `steps` 1 and 2, multiple strides, prognostic
  and boundary variables, and both normalization/masking orders.
- Tests capture sampler output and the final processed `TrainData`, not only raw
  Zarr values.
- Benchmarks separately report loader wait time, loader work time, device preparation
  time where available, and batches per second.
- The benchmark command and environment are documented and reproducible.

Run the local warm-cache raw-batch comparison with:

```shell
uv run --extra rust python scripts/benchmark_rust_loader.py /path/to/om4-fixture
```

Reference warm-cache measurement on 2026-07-15, using a 983 MB local flat-OM4
fixture on Linux aarch64 with `batch_size=4`, `hist=1`, `steps=2`,
`thermo_dynamic_5`, and `tau_hfds`:

| loader | median raw-batch time | p95 raw-batch time |
| --- | ---: | ---: |
| Python/xarray | 210.900 ms | 218.970 ms |
| Rust | 15.836 ms | 19.913 ms |

This is a 13.318x median raw-read speedup. It qualifies the batch reader but does
not replace the Stage 5 end-to-end training and exposed-wait measurements.

## Stage 1: crate integration and opt-in configuration

Bring the useful `crab-load` code into this repository as an Apache-2.0-compatible
PyO3 crate. Add a discriminated `RustDataLoadingConfig` selected by
`loading.type: rust`. It reports zero PyTorch workers and no persistent PyTorch
workers.

Initial configuration should stay small:

- `prefetch_batches`, defaulting to 2;
- a process/rank-wide bounded Rust I/O concurrency setting shared by every store;
- CUDA device prefetch enabled by default when CUDA and pinned memory are available.

The extension must be importable from a Rust-enabled Samudra development install
(`uv sync --extra rust`) and from the supported paired wheel build. CI builds and
install-tests the Samudra and platform-native wheels together. A release must publish
both artifacts before advertising `pip install 'samudra[rust]'`; the default install
does not require a Rust toolchain.

### Exit criteria

- Existing `cpu` and `gpu` configurations parse unchanged.
- `loading.type: rust` parses, selects zero PyTorch workers, and rejects unsupported
  locations or settings with contextual errors.
- The extension builds and imports on supported Linux x86_64 and aarch64 targets.
- CI performs an actual extension import rather than only building a wheel.
- Rust formatting, clippy, and unit tests run in CI.

## Stage 2: persistent, batched flat-OM4 parity

Implement a persistent local flat-OM4 reader. The Python loader consumes each batch
of indices emitted by the existing sampler, resolves them to local dataset indices,
and asks Rust to fill one complete raw batch.

For every autoregressive step, the result matches collated `RawTrainData` shapes:

```text
input    [batch, hist + 1, prognostic_variable, lat, lon]
boundary [batch, hist + 1, boundary_variable,   lat, lon]
label    [batch, hist + 1, prognostic_variable, lat, lon]
```

Rust should deduplicate repeated time/variable reads within a batch where practical,
but correctness does not depend on that optimization. Python continues to construct
`RawTrainData` and run the existing `to_train_data` preparation.

### Exit criteria

- The existing Python samplers produce the index schedule; Rust contains no sampling
  or DDP logic.
- Existing `dataset_id` behavior is unchanged, and mixed-ID batches fail explicitly.
- Processed Rust and CPU batches are equal for every Stage 0 fixture, including NaNs,
  channel order, masks, normalization, history, steps, and stride.
- Stores and array metadata are not reopened per sample or array plane.
- Invalid variables, dtypes, ranks, shapes, and indices return contextual Python
  exceptions and do not panic.
- The Rust path uses zero PyTorch worker processes.

## Stage 3: bounded host prefetch

Add a loader iterator that snapshots the sampler's batch schedule at iterator
creation, then submits batches to a bounded producer. Rust releases the GIL while it
does I/O and decoding. Completed batches are yielded in sampler order even if reads
complete out of order.

The queue owns a strict memory bound. Iterator teardown must cancel pending work and
release readers and buffers after normal exhaustion, exceptions, early loop exits,
and loader reconstruction at a training-step transition.

### Exit criteria

- Sampler schedules are byte-for-byte identical with prefetch disabled and enabled,
  including across `set_epoch` calls and DDP ranks.
- Queue depth never exceeds `prefetch_batches` and is observable in debug metrics.
- Producer exceptions surface on the consumer with the original contextual error.
- Early exit and repeated iterator creation leave no live threads, tasks, or retained
  batch buffers.
- A slow-reader test demonstrates that I/O for batch N+1 overlaps consumption of
  batch N.
- End-to-end exposed data-wait time improves over Stage 2 without unbounded memory
  growth.

## Stage 4: pinned buffers and CUDA prefetch

Use a small reusable pool of pinned CPU Torch buffers. Rust fills writable NumPy
views of those buffers, avoiding a separate collation allocation. PyTorch then copies
and prepares the next batch on a dedicated CUDA stream.

Normalization, masking, and channel flattening run on the prefetch stream. Static
masks, means, standard deviations, and grid context are cached on the target device.
Before yielding a batch, the training stream waits on the prefetch event. Tensor and
host-buffer lifetimes are protected with PyTorch stream recording and CUDA events so
no buffer is reused while a copy is in flight.

CPU training bypasses this stage and consumes the host-prefetched batch directly.

### Exit criteria

- The loader yields the same `TrainData` contract and values as Stage 2.
- Tests cover stream synchronization and buffer reuse with deliberately delayed
  copies; no batch can observe data from a later buffer fill.
- H2D copies are non-blocking from pinned memory and occur on the configured prefetch
  stream.
- A CUDA trace shows batch N+1 transfer/preparation overlapping batch N model work.
- Peak pinned-host and device-prefetch memory stays within the configured queue and
  buffer-pool bounds.
- CPU-only environments import and run without requiring CUDA libraries.

## Stage 5: production qualification and opt-in release

Run the completed flat-OM4 loader through short single-GPU and DDP training jobs,
including validation, checkpoint transitions, and clean shutdown. Keep the Rust path
opt-in for the first release.

### Exit criteria

- A short run matches the current loader's sampled batches and loss trajectory within
  the repository's accepted reproducibility tolerance.
- Single-GPU and multi-rank DDP complete without hangs or rank schedule divergence.
- Median end-to-end loader throughput is at least 1.5 times the current warm-cache
  CPU loader on the agreed production benchmark, or profiling demonstrates that data
  wait is no longer on the critical path.
- p50 and p95 exposed data-wait times, CPU usage, host memory, pinned memory, and GPU
  utilization are recorded.
- User-facing configuration and local-data limitations are documented.
- Existing CPU/GPU loader tests remain green and the Rust path can be disabled without
  affecting them.

## Follow-on formats

After the flat-OM4 release passes production qualification:

1. Add direct OM4-compact variable/level translation and run the same parity gates.
2. Add direct LLC face selection, cropping, staggered-grid translation, dimension
   renaming, time conversion, statistics-name translation, and level-to-channel
   flattening equivalent to `DatasetConfig.canonicalize_datasets`.
3. Qualify inference separately before using the Rust loader outside training and
   validation.

Each format reuses the reader, batch, prefetch, and device-transfer machinery. Only
the physical-to-logical translation is format-specific.
