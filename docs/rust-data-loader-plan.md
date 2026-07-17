<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: Apache-2.0
-->

# Rust data loader shipping plan

## Goal

Ship an opt-in, local-only `loading.type: rust` training path that:

- reads flat and compact OM4 Zarr stores with Rust;
- keeps sampling, shuffle order, batching, and DDP partitioning in Python;
- uses no PyTorch data-loading worker processes;
- loads complete model-facing `TrainData` batches;
- bounds and overlaps host-side batch prefetch with model execution; and
- on CUDA, overlaps pinned-memory transfer and batch preparation with the current
  model step.

The initial prototype path was deliberately narrow: it proved end-to-end correctness
and performance on flat OM4. It subsequently added compact OM4 without changing the
batch, prefetch, or device-transfer contracts.

## Current plan of record: abstraction-first merge

The prototype established the desired behavior and performance, but it also made
several Rust-specific requirements visible by extending existing abstractions in
place. We will not merge the branch as one feature-sized change. Instead, we will
first establish CPU-tested storage, dataset-planning, and loader boundaries, then
rebase the Rust implementation onto them.

This section is the plan of record for how the work will be split and merged. The
functional stages later in this document remain the correctness and performance
requirements for the resulting Rust path.

The refactor will preserve the current `dataset_id` semantics. It will not add S3
support or introduce a canonical-manifest object. OM4-flat, OM4-compact, and LLC
will continue to use direct, format-specific translation code.

### Target abstractions

#### Canonical datasets hide physical storage

Opening and canonicalization form a hard boundary. They return a live, read-capable
`CanonicalDataset` whose public behavior does not reveal whether its data is backed
by xarray, a native Rust reader, flat OM4, compact OM4, or eventually LLC:

```python
@final
class CanonicalDataset:
    name: str
    channels: tuple[str, ...]
    time: TimeAxis
    resolution: Resolution
    statistics: ChannelStatistics
    masks: ChannelMasks
    metadata: Mapping[str, object]

    def select_channels(self, names: Sequence[str]) -> CanonicalDataset: ...
    def slice_time(self, time: TimeConfig) -> CanonicalDataset: ...
    def read(self, plan: BatchReadPlan) -> LoadedPlanes: ...
```

The dataset is structurally immutable: channel selection and time slicing return new
views, while its private reader may mutate file handles, caches, and buffer pools.
Tensor-valued masks are shared read-only metadata for efficiency; callers must not
mutate them in place. Resolution coordinates are returned defensively.
There is no general post-canonicalization `map()` or `map_data()` escape hatch.
Named derived or resampling behavior, if needed later, must be implemented as an
explicit canonicalizer or reader rather than an arbitrary transformation.

Canonical channel names and order come from `DatasetSpec`. Every read exposes
independent canonical planes such as `thetao_4`, regardless of whether the physical
store contains `thetao_4(time, y, x)` or `thetao(time, lev, y, x)`. Statistics and
masks use that same channel order. Depth semantics remain visible; only the physical
representation of a depth channel is hidden.

The Python OM4 canonicalizer inspects the raw store once and constructs lazy xarray
views for those canonical planes. Flat and compact selection, level indexing,
coordinate normalization, statistics translation, and mask construction are private
to that boundary. Downstream Python data-loading code must not inspect `lev`, parse
physical variable names, branch on `is_compact`, or access the raw xarray layout.

The Rust OM4 canonicalizer implements the same operational contract with native
flat and compact readers. Python and Rust canonicalizers remain direct,
format-specific code rather than producing a serialized canonical manifest. Backend
conformance tests compare canonical channels, time, coordinates, statistics, masks,
and reads so the two implementations cannot silently drift.

This choice changes source construction, `DataContainer`, training and inference
datasets, normalization setup, static-data extraction, metadata logging, and related
fixtures. That caller churn is intentional: retaining a public `.data` xarray escape
hatch would allow the physical distinction to continue leaking. Tools that genuinely
need the raw scientific xarray layout, such as conversion or exploratory
visualization, should open it outside the training-source abstraction.

#### Training semantics, reads, and batch preparation

The Rust path will not wrap `TorchTrainDataset` or call its private methods. Common
training semantics will be represented independently of either reader:

```text
TrainingShard
├── canonical datasets and ordered channel selections
├── history, rollout steps, and stride
├── masks, normalization policy, and grid context
├── batch compatibility key
└── window_plan(indices) -> BatchReadPlan

TorchSampleDataset = TrainingShard + Python canonical reads
RustBatchReader    = TrainingShard + native canonical reads

TrainBatchPreparer
├── normalization and masking
├── device-static tensor caches
├── channel shaping and gathering
└── TrainData construction
```

`TorchTrainDataset` will return to being a CPU/sample-fetch adapter. Rank-local CUDA
state and batch-level preprocessing will live in `TrainBatchPreparer`, which both
the Torch and Rust loaders use. Rust's unique-plane representation and deduplicated
device transfer can remain an internal implementation detail; the initial refactor
does not require replacing `RawTrainData` with a universal host-batch format.

This changes dataset construction in `Trainer` and dataset-focused tests, but the
stepper and model continue to receive the existing `TrainData` contract.

#### Loader and runtime ownership

Training-loop consumers will depend on a behavior-oriented loader protocol instead
of a union of concrete Torch and Rust loader classes:

```python
class TrainBatchLoader(Protocol):
    def __iter__(self) -> Iterator[TrainData]: ...
    def __len__(self) -> int: ...
    def set_epoch(self, epoch: int) -> None: ...
    def close(self) -> None: ...
```

The loader owns its schedule and lifecycle. A selected backend factory constructs
the train and validation loaders, owns process-lifetime Rust resources, and performs
backend capability validation. Generic `DataConfig.build()` and `Trainer` will not
know how to construct a Rust read pool or validate Rust-only storage restrictions.

Training-backend settings will not control the still-Python inference loader. In
particular, selecting Rust for training must not force inference to use zero PyTorch
workers. Inference execution settings will be independent of the training backend.

#### Batch compatibility

The current homogeneous-`dataset_id` requirement remains unchanged, but it will be
represented explicitly. Each `TrainingShard` exposes a stable
`batch_compatibility_key`; samplers consume dataset spans and these public keys
instead of a Trainer callback that disguises dataset identity as part of a grid
shape. Initially the key includes shard identity, because the existing collate and
`TrainData` contracts cannot legally combine dataset IDs.

Changing `dataset_id`, allowing heterogeneous batches, or revisiting cross-source
mixing remains separate future work. The compatibility-key change may alter batch
ordering or `drop_last` behavior relative to resolution-only grouping, so its tests
must make those effects explicit.

#### Rust-internal ownership

The Rust feature will use typed physical channel selections and a concrete native
runtime/executor rather than raw tuples and an empty read-pool protocol. Pinned host
buffers will be represented by leases that retain their release obligation until
the associated CUDA event completes. Iterator cleanup will use an explicit
`close()`/`try-finally` path rather than depending on destructors, and `TrainData`
will provide a narrow stream-recording operation so prefetch code does not traverse
its internals.

Host versus CUDA prefetch will be an explicit runtime policy. CUDA prefetch implies
pinned host storage and a CUDA device, avoiding invalid combinations of independent
`pin_memory` and `prefetch_to_device` flags.

### Execution strategy

The first refactor is implemented on a fresh branch from the latest `main`, without
the Rust prototype present. Once its Python CPU and inference behavior passes, the
entire Rust work is rebased onto that refactor before the remaining changes are split
further. This exposes false assumptions in the new boundary while the prototype is
still available as an integration test, instead of designing a sequence of smaller
changes that later proves impossible to reassemble.

After that integration checkpoint, separable behavior and abstraction changes are
extracted from the rebased branch in the order below. A stage may be revised or
rejected if implementation shows that its abstraction makes invariants less clear,
forces a slow common denominator, or cannot support both the Python and native read
patterns. Such a decision must be recorded here with the observed evidence and the
replacement design.

LLC-specific canonicalization is explicitly out of scope for this stack. The first
refactor must leave a dataset-family seam for LLC, but it implements and qualifies
only OM4-flat and OM4-compact.

### Merge sequence and exit criteria

#### P1: immutable, format-blind Python canonical datasets

Starting from the latest `main`, remove the general `DataSource.map()` and
`map_data()` API, introduce `CanonicalDataset` and the Python canonical-reader
boundary, and migrate CPU training and inference. Canonicalize OM4-flat and
OM4-compact into the same ordered channel contract. Replace inference's temporary
mapped sources with explicit read requests. Provide an in-memory canonical reader or
canonicalizer for focused tests instead of preserving arbitrary source transforms.

Exit when:

- production data-loading code contains no `DataSource.map()` or `map_data()` calls;
- no public source property exposes compactness, and code after canonicalization does
  not branch on `is_compact`, inspect `lev` to choose a loading path, or parse physical
  variable names;
- OM4-flat and OM4-compact produce identical canonical channel order, time,
  coordinates, statistics, masks, and CPU training/inference reads;
- channel selection and time slicing return new semantic views without mutating the
  source;
- downstream training code does not access the raw xarray dataset;
- existing CPU correctness tests pass without the Rust extension; and
- no LLC-specific canonicalization is added.

After P1 passes, rebase the complete Rust prototype onto it and run focused flat and
compact parity tests before extracting P2-P6.

#### P2: deterministic schedules and worker seeding

Extract the random-seed, per-epoch sampler, and PyTorch worker-generator changes
from the rebased Rust feature.

Exit when CPU-only tests demonstrate deterministic schedules across runs and epochs,
DDP rank partitioning remains correct, existing sampler behavior is otherwise
unchanged, and no Rust package or configuration is required.

#### P3: common loader protocol and execution-policy separation

Introduce `TrainBatchLoader`, loader-owned epoch/lifecycle operations, and separate
training and inference worker policies. Move backend construction and capability
validation behind a factory boundary.

Exit when the existing Torch path implements the protocol, Trainer and logging code
do not mention concrete loader unions, early iterator exit deterministically closes
resources, and selecting a non-Torch training backend cannot change inference worker
configuration.

#### P4: training shard, read plan, and batch preparer

Extract semantic indexing and device batch preparation from `TorchTrainDataset`.
Migrate the existing CPU/Torch loader to `TrainingShard`, a batched canonical read
plan, and `TrainBatchPreparer` before the Rust loader consumes them. The read API must
support vectorized time/channel requests and caller-owned output where useful; it
must not force the native reader back into per-sample reads.

Exit when the Torch path matches existing raw and processed correctness fixtures,
the dataset owns no rank-local CUDA caches, window planning has a public typed API,
and batch preparation can be invoked without reaching into a concrete dataset's
private state.

#### P5: explicit batch compatibility

Move homogeneous-batch grouping to `TrainingShard.batch_compatibility_key` and make
the sampler consume public shard spans/keys. Keep `dataset_id` behavior unchanged.

Exit when CPU and distributed sampler tests cover grouping, ordering, epoch changes,
and `drop_last`; mixed-ID batches remain impossible or fail explicitly; and neither
Trainer nor Rust has to reconstruct dataset identity from flattened indices merely
to validate compatibility.

#### P6: Rust backend on the established boundaries

Rebase the persistent readers, rollout-wide read deduplication, bounded host
prefetch, pinned-buffer reuse, CUDA prefetch, and device-side batch preparation onto
P1-P5. Add native canonical OM4 readers, explicit runtime policies, buffer leases,
and deterministic lifecycle handling within this feature change.

Exit when Rust no longer exposes physical representation through the canonical
dataset, wraps `TorchTrainDataset`, calls its private methods, or requires concrete
Rust types in generic Trainer/logging code; flat and compact native readers satisfy
the same canonical conformance suite as Python; the parity gates in Stages 0-4 pass;
and the realistic two-GPU performance and profiling results remain representative of
the prototype.

LLC-specific work begins only after this stack has been reviewed and the OM4 Rust
backend is expressed through these boundaries.

### Refactor implementation status (2026-07-17)

- **P1 is the first independently tested commit:** it introduces the structurally
  immutable, format-blind Python `CanonicalDataset` on current `main` and migrates
  CPU training and inference.
- **P2 is the second independently tested commit:** it isolates sampler and
  DataLoader worker RNGs and makes epoch schedules deterministic without consuming
  process-global randomness.
- **P4 is the third independently tested commit:** it introduces
  `TrainingShard.window_plan()` and `TrainBatchPreparer` while retaining the Torch
  loader and requiring no Rust package.
- **P5 is the fourth independently tested commit:** it makes batch compatibility a
  public shard property. Group schedules preserve first-seen dataset order, so the
  current process-local identity semantics cannot reorder equivalence groups across
  DDP ranks.
- **The native prototype, canonical rebase, P3, and P6 are one atomic feature
  commit:** the generic loader factory and native loader constructor change together.
  Training now uses `TrainBatchLoader`, independent inference worker policy, typed
  native selectors,
  a concrete process-local Rust I/O runtime, explicit host/CUDA prefetch policies,
  pinned-buffer leases, weak iterator ownership for deterministic early-exit
  cleanup, and explicit training, validation, and inference-worker teardown. Rust
  consumes `TrainingShard` directly and calls no `TorchTrainDataset` private methods.
- **Correctness qualified:** every abstraction commit passes pre-commit and its
  focused CPU tests. The final focused native/canonical/sampler suite passes 85
  tests; CI covers the full CPU, GPU, data, container, native-extension, architecture,
  and supported-Python matrix. Realistic two-GPU performance is recorded below.

One planned detail changed based on integration evidence. The native OM4 canonical
reader is composed: xarray remains its semantic/static/reference delegate, while a
persistent native capability serves optimized batch plane reads. This keeps
CPU-vs-Rust parity tests independent, avoids duplicating metadata logic, and still
keeps paths, physical selectors, physical time mapping, and flat/compact translation
private. A serialized canonical manifest remains unnecessary.

P3 and P6 are not split across the native-loader constructor change. The generic
factory must begin passing `TrainingShard` and an explicit prefetch policy in the
same commit that Rust stops accepting `TorchTrainDataset` and boolean policy flags.
An intermediate commit would require a compatibility adapter that imports
`TorchTrainDataset` back into `rust_data.py`, preserves two constructor contracts,
and is immediately deleted. That would weaken the invariant under review without
adding independently shippable behavior, so the constructor/factory swap is
intentionally atomic. The CPU-side P4 and P5 abstractions remain independently
reviewable and buildable before it.

### Quarter-degree performance evidence (2026-07-17)

Environment: Torch `gr101`, 2x RTX6000, full v2 high-resolution Samudra model
with 84,002,154 parameters, `/scratch/jr7309/data/om4_quarterdeg_v2`, one year
of training samples, batch size 1, gradient accumulation 4, rollout steps `[4]`,
and boundary variables `tauuo,tauvo,hfds`. Both runs used the same container
image, `ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-ca4d907eb936d37641511f5adb40c3270dd5e6ee`,
with `NCCL_P2P_DISABLE=1`.

The attempted two-epoch CPU/Rust matched job (`14115987`) was stopped after CPU
epoch 1 and part of CPU epoch 2 because CPU validation and epoch-2 first-batch
loads would not leave enough time for the Rust half in the one-hour allocation.
The table therefore compares the completed CPU epoch 1 from that job with a
completed Rust one-epoch run (`14117615`) submitted immediately afterward on the
same node.

| Loader | Job / run | Train epoch total | Mean iter excluding step 0 | Mean iter excluding step 0 and data waits >=1s | Median data wait excluding step 0 | Validation total | Peak CPU / GPU memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | `14115987` / `qdeg-2gpu-ca4d907e-cpu-20260717-154900` | 22.31 s/it | 14.59 s | 4.12 s | 0.003 s | 81.61 s/it | 41.7 GB / 49.1 GB |
| Rust | `14117615` / `qdeg-2gpu-ca4d907e-rust-only-20260717-163700` | 4.13 s/it | 3.24 s | 3.21 s | 0.002 s | 0.59 s/it | 16.3 GB / 53.9 GB |

This is a 5.4x train-epoch improvement and a 139x validation improvement for
this specific one-epoch quarter-degree run. The narrower non-stall training
number is only 1.3x faster, which matches the profile evidence: the Rust win is
mostly eliminating CPU-loader cold loads and periodic stalls while keeping steady
GPU compute roughly the same.

## Design boundaries

### Python remains responsible for

- defining the backend-independent canonical dataset and read-plan contracts;
- canonicalizing xarray-backed OM4 sources into ordered logical channels;
- sampling, shuffle order, epoch seeding, batching, and DDP rank partitioning;
- translating a global `ConcatDataset` index into a dataset and local index;
- constructing `TrainData` objects;
- normalization, masking, channel flattening, and PyTorch device/stream semantics;
- coordinating loader lifetime with the training and validation loops.

### Rust is responsible for

- canonicalizing supported native OM4 stores behind the same public dataset
  contract, including private flat/compact channel translation;
- keeping local Zarr stores and arrays open across batches;
- reading the canonical channels and time indices described by Python read plans;
- issuing bounded concurrent reads;
- writing directly into caller-owned unique-plane output buffers; and
- returning contextual read, shape, dtype, and index errors without panics.

Each supported physical format uses direct translation code private to its
canonicalizer. LLC receives its own implementation later; this stack does not add
it. We will not introduce a canonical-manifest abstraction.

### Explicit non-goals

- S3 or another remote object store;
- changing `dataset_id` or enabling heterogeneous-batch semantics;
- removing the existing CPU/GPU loader or making Rust the default;
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
 bounded Rust host prefetch -> pinned unique physical planes
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

Record correctness fixtures for the current CPU loader on a small local flat-OM4
store.

### Exit criteria

- Fixture cases cover `hist` 0 and 1, `steps` 1 and 2, multiple strides, prognostic
  and boundary variables, and both normalization/masking orders.
- Tests capture sampler output and the final processed `TrainData`, not only raw
  Zarr values.

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

Implement a persistent local flat-OM4 reader. Python resolves sampler batches to
dataset-local indices, deduplicates repeated time/variable planes across the full
rollout, and asks Rust to fill caller-owned buffers. Python then normalizes, masks,
and gathers those unique planes into the existing `TrainData` layout.

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
does I/O and decoding. The single producer overlaps batch N+1 loading with batch N
consumption while preserving sampler order.

The queue owns a strict memory bound. Iterator teardown must cancel pending work and
release readers and buffers after normal exhaustion, exceptions, early loop exits,
and loader reconstruction at a training-step transition.

### Exit criteria

- Sampler schedules are byte-for-byte identical with prefetch disabled and enabled,
  including across `set_epoch` calls and DDP ranks.
- Queue depth never exceeds `prefetch_batches`.
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

## Follow-on formats

After the flat-OM4 implementation:

1. Add direct OM4-compact variable/level translation and run the same parity gates.
   Completed on 2026-07-15.
2. Add direct LLC face selection, cropping, staggered-grid translation, dimension
   renaming, time conversion, statistics-name translation, and level-to-channel
   flattening equivalent to `DatasetConfig.canonicalize_datasets`.

Each format reuses the reader, batch, prefetch, and device-transfer machinery. Only
the physical-to-logical translation is format-specific.

### OM4-compact milestone

The first follow-on milestone adds local OM4-compact training and validation to the
same opt-in `loading.type: rust` path. The native OM4 canonicalizer privately
translates canonical channels such as `thetao_4` to typed physical selectors such as
`("thetao", 4)`. Rust reads those physical array planes without leaking compactness
or selectors through `CanonicalDataset` or its read plans. Surface channels such as
`zos` and `hfds` use selectors without a level. This remains direct format-specific
translation rather than a canonical-manifest object.

#### C0: native translation and fixture parity

Status: passed on 2026-07-15.

Exit when native reads validate rank, dtype, dimension order, spatial/time shape,
level bounds, and missing variables; direct channel ordering matches the CPU compact
path; and raw plus processed batches match for histories 0/1, steps 1/2, strides
1/2, mixed depth/surface variables, masks, nontrivial statistics, NaNs, and both
normalization/masking orders.

#### C1: prefetched loader integration and regressions

Status: passed on 2026-07-15.

Exit when compact sources run through the existing opt-in trainer selection and
bounded host-prefetch path with zero PyTorch workers; lifecycle and producer-error
tests remain green; and the full non-CUDA suite plus focused CUDA pinned-buffer and
stream tests show no regression in flat OM4 or the existing CPU loader.

## Deferred and future work

These items are intentionally outside the current OM4 hardening slice:

- Publish the Samudra and native loader wheels as an atomic, tested pair. Install
  both artifacts into a clean environment for every supported Python version and
  Linux architecture, then exercise the Rust trainer import and a fixture batch.
- Add native seasonal-climatology derivation for boundary variables such as
  `hfds_anomalies`. Until then, Rust configuration rejects derived channels early.
- Make CUDA-prefetch batches safe to retain beyond the next iterator advance, or
  explicitly narrow and document that lifetime contract.
- Investigate phase-aware CUDA prefetch scheduling so loader-side normalization,
  masking, and gathers fill model communication gaps instead of competing with
  forward kernels. CUDA does not provide an idle-only stream: both the default and
  loader streams have priority `0`, which is the least-priority tier on the local
  GB10 and Torch RTX PRO 6000 hosts (the CUDA runtime reports `0` through `-5`,
  while PyTorch stream creation clamps to `0` through `-3`). Prefer evaluating an
  early H2D copy followed by event-gated transforms; alternatively, evaluate moving
  model computation to a higher-priority non-default stream while leaving loader
  work at `0`.
- Decide whether iterator shutdown should be non-blocking while an active native
  read finishes; current shutdown waits so it can deterministically reclaim buffers.
- Map native validation and lookup failures to more specific Python exception types.
