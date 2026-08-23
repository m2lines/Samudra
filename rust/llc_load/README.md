# `ocean_llc_loader` — native Zarr reader for LLC4320

An **opt-in** Rust data loader. It is off unless a config sets
`data.loader_backend: rust`, and the CPU (xarray/zarr) path is untouched by its
presence — including if the extension was never built.

Ported from the OM4 Rust loader in
[m2lines/Samudra#800](https://github.com/m2lines/Samudra/pull/800)
(`rust/crab_load`), retargeted at the raw LLC4320 store.

## Why it exists

Caching 3D prognostics for the globe costs roughly **500 TB**, which is not
available. Reading them straight out of the raw store costs nothing extra to
store — and for a **chunk-aligned 720×720 tile** it reads exactly one chunk per
variable per timestamp, with no read amplification at all.

What does not work is reading the 2D boundary fields the same way. So the
supported configuration is: **prognostics from the raw store, boundaries from a
small packed cache** (`data.boundary_data_location`).

## Build

```bash
scripts/build_rust_loader.sh          # installs ocean_llc_loader.so into .venv
```

Installs a minimal Rust toolchain into `$CARGO_HOME` first if none is on PATH.

`cargo test` links a normal binary, so it needs `libpython3.11.so`, which this
host has only as `libpython3.11.so.1.0`. The Python tests in
`tests/test_rust_data.py` cover the reader end to end; to run the Rust unit
tests too, point the linker at a symlink:

```bash
mkdir -p /tmp/pylib && ln -sf /usr/lib64/libpython3.11.so.1.0 /tmp/pylib/libpython3.11.so
RUSTFLAGS="-L /tmp/pylib" cargo test --release --manifest-path rust/llc_load/Cargo.toml
```

## The chunking, which dictates everything

| family | shape | chunks | one chunk inflated | on disk |
| --- | --- | --- | --- | --- |
| 3D (`U`, `V`, `W`, `Theta`, `Salt`) | `(10311, 51, 13, 4320, 4320)` | `(1, 51, 1, 720, 720)` | 106 MB | 50–83 MB |
| 2D (`Eta`, `oceTAUX`, `oceTAUY`, `oceQnet`) | `(10311, 13, 4320, 4320)` | `(1, 13, 4320, 4320)` | 970 MB | 370–410 MB |

**Keep tiles on the 720 grid.** An aligned 720×720 tile is exactly one 3D chunk
per variable and zarrs takes its whole-chunk fast path. Offset that same tile off
the grid and it spans four chunks, measured at ~1.7× slower (2.53 s → 4.39 s for
205 channels). `NativeStoreSpec.is_chunk_aligned` warns when a window is off-grid.

**The 2D chunk is the whole globe, not one face** — one file per timestamp
covering all 13 faces. There is no way to read one tile's surface field without
inflating all 970 MB, so four boundary variables cost ~1.5 GB per sample to
deliver 8 MB. That is what `data.boundary_data_location` avoids. The flip side:
because one inflation serves every face, a *chunk-first* bulk builder pays it
once for all tiles — see `scripts/build_multiple_llc_patch_cache_compressed.py`.

## Measured

Single process, one chunk-aligned 720×720 face-1 tile, 205 prognostic + 4
boundary channels, median replay transition, 8 read threads:

| loader | source | median |
| --- | --- | --- |
| CPU (xarray/zarr) | full packed 3D cache | **0.9 s** |
| Rust | raw-store 3D + packed boundary cache | **2.7 s** |
| Rust | everything from the raw store | 8.2 s |
| CPU (xarray/zarr) | everything from the raw store | 13.4 s |

Rust is ~1.6× the CPU loader on the same raw store; the split configuration is
~3× a full cache but needs **no 3D storage at all**. Note the CPU loader is
*faster* than Rust on a packed cache (0.89 s vs 1.03 s) — a cache read is one
chunk, so there is nothing to parallelise and Rust's f16→f32 buffer is pure
overhead. Rust only wins where there are many chunks to overlap and partial
chunks to decode, i.e. on the raw store.

## What Python keeps

Python owns every canonical semantic: which channels exist and in what order,
which timestamps a sample reads, normalisation, masking, DDP schedules. Rust owns
only persistent Zarr handles and chunk reads.

`rust_data.channel_selectors` reproduces the channel order that
`DataSource.filter` plus `_dataset_to_numpy` produce on the xarray path, so a
Rust batch is element-for-element what the CPU loader would have built.
`tests/test_rust_data.py` asserts that against synthetic stores; it was also
verified element-wise against the production store.

## Environment knobs

| variable | default | meaning |
| --- | --- | --- |
| `OCEAN_RUST_LOADER_THREADS` | `min(8, cores)` | Rayon threads per process (`data.rust_read_threads` takes precedence) |
| `OCEAN_RUST_LOADER_FULL_ROWS` | `1` | whole-`j`-row reads, so a partial chunk decodes as one contiguous blosc range instead of one per row (~970 MB → ~12 MB); `0` reads the exact `i` window |
| `OCEAN_RUST_LOADER_CACHE_MB` | `0` (off) | per-process inflated-plane cache; only helps when one process re-reads a timestamp, i.e. grouped replay over overlapping tiles |

## Scope

Local filesystem only. Training and validation reads only — inference still goes
through `InferenceDataset` and xarray. Float32 and float16 arrays.

## Removing it

Nothing else depends on Rust:

```bash
rm -rf rust/ src/ocean_emulators/rust_data.py tests/test_rust_data.py
rm -f  scripts/build_rust_loader.sh .venv/lib/python3.11/site-packages/ocean_llc_loader.so
```

then revert the guarded blocks — each is short and `if native/rust`-gated:

| file | what to revert |
| --- | --- |
| `src/ocean_emulators/config.py` | the `loader_backend` / `rust_read_threads` / `boundary_data_location` / `llc_tiles` fields and the `native_store=` argument in `DataConfig.build` |
| `src/ocean_emulators/utils/data.py` | the `native_store` and `boundary_source` fields, `from_boundary_only_cache`, the `store_time_index` coordinate, the `filter` boundary hand-off, and the `native_store is None` early return in `from_locations` |
| `src/ocean_emulators/datasets.py` | the `self._native_*` block in `TorchTrainDataset.__init__`, `_read_prognostic`/`_read_boundary`/`_positions`/`_require_store_time_index`, and the module-level `_align_times`/`_contiguous_indexer` |
| `src/ocean_emulators/tiling.py` | `tile_catalog_from_windows` |
| `src/ocean_emulators/train.py` | the `replay_windows` branch in `_build_tile_catalog` |

`data.llc_tiles`, `data.boundary_data_location` and `DataContainer.replay_windows`
are backend-agnostic — they work with the CPU loader too — so they can stay.
