//! Native Zarr reader for raw LLC4320 stores and packed LLC caches.
//!
//! Ported from the OM4 Rust data loader in m2lines/Samudra#800
//! (`rust/crab_load`). The division of responsibility is unchanged: Python keeps
//! every canonical semantic -- sampling, DDP schedules, normalisation, masking,
//! channel order -- and Rust owns only persistent Zarr handles, chunk reads and
//! buffer lifetime.
//!
//! What is new here is the LLC geometry. OM4 arrays are `(time, lat, lon)` or
//! `(time, lev, lat, lon)` and every read is a whole plane. Raw LLC4320 arrays
//! are `(time, k, face, j, i)` / `(time, face, j, i)` over the whole globe, and
//! training reads one small tile out of them, so this reader takes a
//! face/j/i window and reads a *subset* of each chunk.
//!
//! Why the window matters, measured on the production store:
//!
//! * 3D arrays are chunked `(1, 51, 1, 720, 720)`. A **chunk-aligned** 720x720
//!   tile is exactly one chunk per variable -- every byte read is a byte wanted,
//!   and zarrs takes its whole-chunk fast path. Offsetting that same tile off
//!   the chunk grid makes it four chunks and costs ~1.7x (2.53 s -> 4.39 s for
//!   205 channels). Keep tiles on the grid.
//! * 2D arrays are chunked `(1, 13, 4320, 4320)`: one chunk is the *entire
//!   globe* at one timestamp, ~970 MiB inflated, of which a 720x720 tile is
//!   0.1%. Reading four boundary variables that way costs ~1.5 GiB per sample
//!   to deliver 8 MiB. Prefer a packed boundary cache (this reader handles them
//!   too, including float16); see `read_into` and the README.
//!
//! For the unavoidable cases, `full_row_reads` keeps the partial-chunk cost
//! down: zarrs decodes a partial chunk through the blosc partial decoder
//! (`blosc_getitem`), which inflates only the blosc blocks covering the
//! requested byte ranges. Asking for the exact `i` window produces one byte
//! range per row; asking for whole `j` rows and cropping `i` here produces one
//! contiguous range, turning ~970 MiB of inflation into ~12 MiB.

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{bail, Context};
use numpy::{PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use zarrs::{
    array::{codec::CodecOptions, Array, DataType, ElementOwned},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits},
};

type OpenArray = Array<dyn ReadableWritableListableStorageTraits>;
const XARRAY_DIMENSIONS_ATTRIBUTE: &str = "_ARRAY_DIMENSIONS";

/// LLC vertical axes. `k_p1`/`k_l`/`k_u` are the staggered variants (`W` lives
/// on `k_p1`); `lev` shows up in already-flattened stores.
const LEVEL_DIMS: &[&str] = &["k", "k_p1", "k_l", "k_u", "lev", "level"];
/// Row axis. `j_g` is the meridionally staggered variant (`V`, `oceTAUY`).
const ROW_DIMS: &[&str] = &["j", "j_g", "y", "lat"];
/// Column axis. `i_g` is the zonally staggered variant (`U`, `oceTAUX`).
const COL_DIMS: &[&str] = &["i", "i_g", "x", "lon"];

/// A packed cache's `boundary_channel` / `prognostic_channel` axis is the same
/// kind of thing as a vertical axis: an axis a channel selects one index along.
/// Treating them alike is what lets one reader serve both a raw LLC store and a
/// packed cache.
fn is_level_dim(name: &str) -> bool {
    LEVEL_DIMS.contains(&name) || name.ends_with("_channel")
}

/// Hands disjoint pieces of one output buffer to several Rayon tasks at once.
///
/// Disjointness is a property of the work list, not of the type, so every
/// `piece` call carries its own safety argument. See `read_into_impl`.
struct OutputCells {
    ptr: *mut f32,
    len: usize,
}

unsafe impl Send for OutputCells {}
unsafe impl Sync for OutputCells {}

impl OutputCells {
    /// # Safety
    /// The caller must ensure no two live pieces overlap.
    unsafe fn piece(&self, start: usize, len: usize) -> &mut [f32] {
        debug_assert!(start + len <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(start), len)
    }
}

/// Read one subset as `T`. Split out so the float16 and float32 arms share a
/// single call site.
fn retrieve_widened<T: ElementOwned>(
    array: &OpenArray,
    subset: &ArraySubset,
    options: &CodecOptions,
) -> anyhow::Result<Vec<T>> {
    Ok(array.retrieve_array_subset_elements_opt::<T>(subset, options)?)
}

fn python_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}

fn validate_index(index: i64, time_len: u64) -> anyhow::Result<u64> {
    if index < 0 {
        bail!("time index must be non-negative, got {index}");
    }
    let index = index as u64;
    if index >= time_len {
        bail!("time index {index} is out of bounds for length {time_len}");
    }
    Ok(index)
}

fn dimension_names(array: &OpenArray) -> anyhow::Result<Vec<String>> {
    if let Some(names) = array.dimension_names() {
        if names.iter().all(Option::is_some) {
            return Ok(names
                .iter()
                .map(|name| name.clone().expect("checked as Some"))
                .collect());
        }
    }

    // Zarr V2 stores written by xarray carry the axis names in
    // `_ARRAY_DIMENSIONS` rather than in the array metadata itself.
    let names = array
        .attributes()
        .get(XARRAY_DIMENSIONS_ATTRIBUTE)
        .with_context(|| {
            format!(
                "array {} has no dimension metadata; expected named LLC axes",
                array.path()
            )
        })?;
    serde_json::from_value::<Vec<String>>(names.clone()).with_context(|| {
        format!(
            "array {} has invalid {XARRAY_DIMENSIONS_ATTRIBUTE} metadata",
            array.path()
        )
    })
}

/// Where each LLC axis sits in one array's dimension order.
#[derive(Clone, Copy, Debug)]
struct AxisMap {
    ndim: usize,
    time: usize,
    level: Option<usize>,
    face: Option<usize>,
    row: usize,
    col: usize,
}

impl AxisMap {
    fn resolve(array: &OpenArray) -> anyhow::Result<Self> {
        let names = dimension_names(array)?;
        let find = |candidates: &[&str]| {
            names
                .iter()
                .position(|name| candidates.contains(&name.as_str()))
        };
        let time = find(&["time"]).with_context(|| {
            format!(
                "array {} has dimensions {names:?}; expected a time axis",
                array.path()
            )
        })?;
        let row = find(ROW_DIMS).with_context(|| {
            format!(
                "array {} has dimensions {names:?}; expected a j/y axis",
                array.path()
            )
        })?;
        let col = find(COL_DIMS).with_context(|| {
            format!(
                "array {} has dimensions {names:?}; expected an i/x axis",
                array.path()
            )
        })?;
        Ok(Self {
            ndim: names.len(),
            time,
            level: names.iter().position(|name| is_level_dim(name)),
            face: find(&["face"]),
            row,
            col,
        })
    }
}

/// One logical training channel: a physical array plus an optional level.
///
/// `Theta_7` is `("Theta", Some(7))`; `Eta` is `("Eta", None)`; a packed cache's
/// third boundary channel is `("boundary", Some(2))`.
#[derive(Clone)]
struct Channel {
    array: Arc<OpenArray>,
    axes: AxisMap,
    level: Option<u64>,
}

/// Channels backed by the same physical array, read together.
///
/// An LLC 3D chunk spans every level for one timestamp, so `Theta_0..Theta_50`
/// must inflate that chunk once, not 51 times.
struct ReadGroup {
    array: Arc<OpenArray>,
    axes: AxisMap,
    /// `(output channel index, level)`, in output order.
    channels: Vec<(usize, Option<u64>)>,
    /// Cache key prefix; the array path is unique within a store.
    key: String,
}

/// The tile this reader serves, in native LLC index space.
#[derive(Clone, Copy, Debug)]
struct Window {
    face: Option<u64>,
    row_start: u64,
    row_stop: u64,
    col_start: u64,
    col_stop: u64,
}

impl Window {
    fn height(&self) -> u64 {
        self.row_stop - self.row_start
    }
    fn width(&self) -> u64 {
        self.col_stop - self.col_start
    }
}

/// Bounded inflated-plane cache, keyed by `(array, first level, time)`.
///
/// Off by default (`OCEAN_RUST_LOADER_CACHE_MB=0`). It only pays off when one
/// process reads the same timestamp repeatedly -- grouped replay over
/// overlapping tiles -- which is exactly the case where a full-globe 2D chunk
/// would otherwise be re-read per tile.
struct PlaneCache {
    budget_bytes: usize,
    used_bytes: usize,
    order: VecDeque<(String, u64)>,
    entries: HashMap<(String, u64), Arc<Vec<f32>>>,
}

impl PlaneCache {
    fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn get(&self, key: &(String, u64)) -> Option<Arc<Vec<f32>>> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: (String, u64), value: Arc<Vec<f32>>) {
        let bytes = value.len() * std::mem::size_of::<f32>();
        if bytes > self.budget_bytes || self.entries.contains_key(&key) {
            return;
        }
        while self.used_bytes + bytes > self.budget_bytes {
            let Some(evicted) = self.order.pop_front() else {
                break;
            };
            if let Some(value) = self.entries.remove(&evicted) {
                self.used_bytes -= value.len() * std::mem::size_of::<f32>();
            }
        }
        self.used_bytes += bytes;
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Shared bounded Rayon pool for every reader in one training process.
#[pyclass]
struct LlcReadPool {
    thread_pool: Arc<ThreadPool>,
}

#[pymethods]
impl LlcReadPool {
    #[new]
    fn new(max_concurrent_reads: usize) -> PyResult<Self> {
        if max_concurrent_reads == 0 {
            return Err(python_error(anyhow::anyhow!(
                "max_concurrent_reads must be positive"
            )));
        }
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(max_concurrent_reads)
            .thread_name(|index| format!("ocean-llc-zarr-{index}"))
            .build()
            .context("creating the shared Rust Zarr read pool")
            .map_err(python_error)?;
        Ok(Self {
            thread_pool: Arc::new(thread_pool),
        })
    }
}

/// Persistent reader for one tile of a raw LLC4320 store or a packed cache.
#[pyclass]
struct LlcPatchReader {
    path: PathBuf,
    channels: Vec<Channel>,
    window: Window,
    time_len: u64,
    /// Read whole `j` rows and crop `i` here, so a partial chunk decodes as one
    /// contiguous byte range instead of one per row.
    full_row_reads: bool,
    cache: Option<Mutex<PlaneCache>>,
    thread_pool: Arc<ThreadPool>,
}

impl LlcPatchReader {
    fn open(
        path: PathBuf,
        channel_selectors: Vec<(String, Option<u64>)>,
        window: Window,
        thread_pool: Arc<ThreadPool>,
    ) -> anyhow::Result<Self> {
        if channel_selectors.is_empty() {
            bail!("at least one channel is required");
        }
        if window.row_stop <= window.row_start || window.col_stop <= window.col_start {
            bail!(
                "empty tile window j=[{}:{}) i=[{}:{})",
                window.row_start,
                window.row_stop,
                window.col_start,
                window.col_stop
            );
        }

        let store: ReadableWritableListableStorage = Arc::new(
            FilesystemStore::new(&path)
                .with_context(|| format!("opening local Zarr store {}", path.display()))?,
        );

        let mut opened: HashMap<String, (Arc<OpenArray>, AxisMap)> = HashMap::new();
        let mut channels = Vec::with_capacity(channel_selectors.len());
        let mut time_len: Option<u64> = None;

        for (name, level) in channel_selectors {
            if name.is_empty() {
                bail!("channel names must not be empty");
            }
            let (array, axes) = if let Some(entry) = opened.get(&name) {
                entry.clone()
            } else {
                let array_path = if name.starts_with('/') {
                    name.clone()
                } else {
                    format!("/{name}")
                };
                let array = Arc::new(Array::open(store.clone(), &array_path).with_context(
                    || format!("opening variable {name:?} in {}", path.display()),
                )?);
                // Raw LLC stores are float32; a packed cache is float16. Both
                // widen to the float32 the model consumes.
                if !matches!(array.data_type(), DataType::Float32 | DataType::Float16) {
                    bail!(
                        "variable {name:?} in {} has dtype {:?}; the Rust LLC loader \
                         reads float32 and float16",
                        path.display(),
                        array.data_type()
                    );
                }
                let axes = AxisMap::resolve(&array).with_context(|| {
                    format!("resolving LLC axes for {name:?} in {}", path.display())
                })?;
                let shape = array.shape();
                if shape.len() != axes.ndim {
                    bail!(
                        "variable {name:?} in {} has {} dimensions but {} dimension names",
                        path.display(),
                        shape.len(),
                        axes.ndim
                    );
                }
                if let Some(face_axis) = axes.face {
                    let face = window.face.with_context(|| {
                        format!(
                            "variable {name:?} in {} has a face dimension but no face was configured",
                            path.display()
                        )
                    })?;
                    if face >= shape[face_axis] {
                        bail!(
                            "face {face} is out of bounds for {name:?} with {} faces",
                            shape[face_axis]
                        );
                    }
                }
                if window.row_stop > shape[axes.row] || window.col_stop > shape[axes.col] {
                    bail!(
                        "tile window j=[{}:{}) i=[{}:{}) does not fit {name:?} with spatial shape {}x{}",
                        window.row_start,
                        window.row_stop,
                        window.col_start,
                        window.col_stop,
                        shape[axes.row],
                        shape[axes.col]
                    );
                }
                let array_time_len = shape[axes.time];
                match time_len {
                    Some(expected) if expected != array_time_len => bail!(
                        "variable {name:?} in {} has {array_time_len} timestamps; expected {expected}",
                        path.display()
                    ),
                    _ => time_len = Some(array_time_len),
                }
                let entry = (array, axes);
                opened.insert(name.clone(), entry.clone());
                entry
            };

            match (level, axes.level) {
                (Some(level), Some(level_axis)) => {
                    let levels = array.shape()[level_axis];
                    if level >= levels {
                        bail!(
                            "channel {name:?} selects level {level}, but the array has {levels} levels"
                        );
                    }
                }
                (Some(level), None) => bail!(
                    "channel {name:?} selects level {level}, but the array has no vertical axis"
                ),
                (None, Some(_)) => {
                    bail!("channel {name:?} has a vertical axis; a level must be selected")
                }
                (None, None) => {}
            }

            channels.push(Channel { array, axes, level });
        }

        let cache_mb = env_usize("OCEAN_RUST_LOADER_CACHE_MB", 0);
        Ok(Self {
            path,
            channels,
            window,
            time_len: time_len.expect("channel_selectors was checked as non-empty"),
            full_row_reads: env_flag("OCEAN_RUST_LOADER_FULL_ROWS", true),
            cache: (cache_mb > 0).then(|| Mutex::new(PlaneCache::new(cache_mb * 1024 * 1024))),
            thread_pool,
        })
    }

    /// Group the requested channels by physical array, preserving output order.
    fn read_groups(&self, selection: &[usize]) -> Vec<ReadGroup> {
        let mut index_of: HashMap<String, usize> = HashMap::new();
        let mut groups: Vec<ReadGroup> = Vec::new();
        for (output_index, channel_index) in selection.iter().enumerate() {
            let channel = &self.channels[*channel_index];
            let key = channel.array.path().to_string();
            if let Some(group_index) = index_of.get(&key) {
                groups[*group_index]
                    .channels
                    .push((output_index, channel.level));
            } else {
                index_of.insert(key.clone(), groups.len());
                groups.push(ReadGroup {
                    array: channel.array.clone(),
                    axes: channel.axes,
                    channels: vec![(output_index, channel.level)],
                    key,
                });
            }
        }
        groups
    }

    fn read_into_impl(
        &self,
        indexes: &[u64],
        selection: &[usize],
        output: &mut [f32],
    ) -> anyhow::Result<()> {
        let height = self.window.height() as usize;
        let width = self.window.width() as usize;
        let plane_len = height
            .checked_mul(width)
            .context("LLC tile shape overflows usize")?;
        let time_slice_len = selection
            .len()
            .checked_mul(plane_len)
            .context("LLC time-slice shape overflows usize")?;
        let expected = indexes
            .len()
            .checked_mul(time_slice_len)
            .context("LLC output shape overflows usize")?;
        if output.len() != expected {
            bail!(
                "LLC output has {} elements; expected {expected}",
                output.len()
            );
        }

        let groups = self.read_groups(selection);
        let options = CodecOptions::default();

        // Parallelise over every (timestamp, physical array) pair. Replay reads
        // one timestamp at a time, so parallelising over timestamps alone would
        // leave a single thread walking the arrays in series -- and one LLC 2D
        // array is ~390 MiB of compressed bytes off NFS, so that serialisation
        // is the whole cost. Peak scratch stays at one inflated chunk per busy
        // thread, which the pool size bounds.
        let reads: Vec<(usize, usize)> = (0..indexes.len())
            .flat_map(|time| (0..groups.len()).map(move |group| (time, group)))
            .collect();
        let cells = OutputCells {
            ptr: output.as_mut_ptr(),
            len: output.len(),
        };

        self.thread_pool.install(|| {
            reads
                .par_iter()
                .try_for_each(|(time, group)| -> anyhow::Result<()> {
                    // SAFETY: `reads` visits each (time, group) pair once and a
                    // group writes only the channel planes it owns, so no two
                    // tasks touch the same bytes.
                    let target =
                        unsafe { cells.piece(time * time_slice_len, time_slice_len) };
                    self.fill_group(
                        &groups[*group],
                        indexes[*time],
                        target,
                        plane_len,
                        &options,
                    )
                })
        })
    }

    fn fill_group(
        &self,
        group: &ReadGroup,
        index: u64,
        target: &mut [f32],
        plane_len: usize,
        options: &CodecOptions,
    ) -> anyhow::Result<()> {
        let first_level = group
            .channels
            .iter()
            .filter_map(|(_, level)| *level)
            .min()
            .unwrap_or(0);

        let cache_key = self
            .cache
            .as_ref()
            .map(|_| (format!("{}#{first_level}", group.key), index));
        if let (Some(cache), Some(key)) = (self.cache.as_ref(), cache_key.as_ref()) {
            if let Some(planes) = cache.lock().expect("plane cache mutex").get(key) {
                Self::scatter(group, first_level, &planes, target, plane_len);
                return Ok(());
            }
        }

        let planes = Arc::new(self.retrieve(group, index, first_level, plane_len, options)?);
        Self::scatter(group, first_level, &planes, target, plane_len);
        if let (Some(cache), Some(key)) = (self.cache.as_ref(), cache_key) {
            cache.lock().expect("plane cache mutex").insert(key, planes);
        }
        Ok(())
    }

    /// Read `[levels, height, width]` for one timestamp of one physical array.
    fn retrieve(
        &self,
        group: &ReadGroup,
        index: u64,
        first_level: u64,
        plane_len: usize,
        options: &CodecOptions,
    ) -> anyhow::Result<Vec<f32>> {
        let axes = group.axes;
        let shape = group.array.shape();
        let level_count = match axes.level {
            Some(_) => {
                let last_level = group
                    .channels
                    .iter()
                    .filter_map(|(_, level)| *level)
                    .max()
                    .expect("levelled group has selected levels");
                last_level - first_level + 1
            }
            None => 1,
        };

        // Whole-row reads keep a partial chunk to one contiguous byte range.
        // Widen only to this array's OWN column chunking -- widening further
        // would pull in neighbouring chunks, and the chunk width differs
        // between LLC's 3D arrays (720) and its 2D ones (4320).
        let (col_start, col_stop) = if self.full_row_reads {
            let chunk_shape = group
                .array
                .chunk_shape(&vec![0; shape.len()])
                .context("reading the chunk shape")?;
            let chunk_cols = chunk_shape[axes.col].get();
            let start = (self.window.col_start / chunk_cols) * chunk_cols;
            let stop = self
                .window
                .col_stop
                .div_ceil(chunk_cols)
                .saturating_mul(chunk_cols)
                .min(shape[axes.col]);
            (start, stop)
        } else {
            (self.window.col_start, self.window.col_stop)
        };

        let mut start = vec![0u64; axes.ndim];
        let mut extent = vec![1u64; axes.ndim];
        start[axes.time] = index;
        if let Some(level_axis) = axes.level {
            start[level_axis] = first_level;
            extent[level_axis] = level_count;
        }
        if let Some(face_axis) = axes.face {
            start[face_axis] = self.window.face.expect("validated at open");
        }
        start[axes.row] = self.window.row_start;
        extent[axes.row] = self.window.height();
        start[axes.col] = col_start;
        extent[axes.col] = col_stop - col_start;

        let read_width = (col_stop - col_start) as usize;
        let subset = ArraySubset::new_with_start_shape(start, extent)
            .context("constructing an LLC array subset")?;
        let elements = match group.array.data_type() {
            DataType::Float16 => retrieve_widened::<half::f16>(&group.array, &subset, options)
                .map(|values| values.into_iter().map(f32::from).collect::<Vec<_>>()),
            _ => retrieve_widened::<f32>(&group.array, &subset, options),
        }
        .with_context(|| {
            format!(
                "reading {} at time index {index} from {}",
                group.array.path(),
                self.path.display()
            )
        })?;

        let height = self.window.height() as usize;
        let expected = level_count as usize * height * read_width;
        if elements.len() != expected {
            bail!(
                "read {} elements from {} at time index {index}; expected {expected}",
                elements.len(),
                group.array.path()
            );
        }
        if read_width == self.window.width() as usize {
            return Ok(elements);
        }

        // Crop the widened columns back to the tile.
        let offset = (self.window.col_start - col_start) as usize;
        let width = self.window.width() as usize;
        let mut cropped = Vec::with_capacity(level_count as usize * plane_len);
        for level in 0..level_count as usize {
            for row in 0..height {
                let start = (level * height + row) * read_width + offset;
                cropped.extend_from_slice(&elements[start..start + width]);
            }
        }
        Ok(cropped)
    }

    fn scatter(
        group: &ReadGroup,
        first_level: u64,
        planes: &[f32],
        target: &mut [f32],
        plane_len: usize,
    ) {
        for (output_index, level) in &group.channels {
            let source = level.map_or(0, |level| level - first_level) as usize * plane_len;
            let destination = output_index * plane_len;
            target[destination..destination + plane_len]
                .copy_from_slice(&planes[source..source + plane_len]);
        }
    }
}

#[pymethods]
impl LlcPatchReader {
    #[new]
    #[pyo3(signature = (path, channels, face, j_start, j_stop, i_start, i_stop, read_pool))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        channels: Vec<(String, Option<u64>)>,
        face: Option<u64>,
        j_start: u64,
        j_stop: u64,
        i_start: u64,
        i_stop: u64,
        read_pool: PyRef<'_, LlcReadPool>,
    ) -> PyResult<Self> {
        let thread_pool = read_pool.thread_pool.clone();
        let window = Window {
            face,
            row_start: j_start,
            row_stop: j_stop,
            col_start: i_start,
            col_stop: i_stop,
        };
        py.allow_threads(|| Self::open(path, channels, window, thread_pool))
            .map_err(python_error)
    }

    /// Fill a writable C-contiguous `[time, channel, j, i]` float32 array.
    ///
    /// `channel_indexes` selects and orders channels from the set this reader
    /// was opened with, so one reader can serve both a full training read and a
    /// boundary-only replay read.
    fn read_into(
        &self,
        py: Python<'_>,
        indexes: Vec<i64>,
        channel_indexes: Vec<usize>,
        target: Bound<'_, PyArray4<f32>>,
    ) -> PyResult<()> {
        if channel_indexes.is_empty() {
            return Err(python_error(anyhow::anyhow!(
                "at least one channel is required for a read"
            )));
        }
        if let Some(bad) = channel_indexes
            .iter()
            .find(|index| **index >= self.channels.len())
        {
            return Err(python_error(anyhow::anyhow!(
                "channel index {bad} is out of bounds for {} opened channels",
                self.channels.len()
            )));
        }
        let indexes = indexes
            .iter()
            .map(|index| validate_index(*index, self.time_len))
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(python_error)?;

        let mut target = target.try_readwrite().map_err(|error| {
            python_error(anyhow::anyhow!(
                "LLC output could not be borrowed for writing: {error}"
            ))
        })?;
        let expected_shape = [
            indexes.len(),
            channel_indexes.len(),
            self.window.height() as usize,
            self.window.width() as usize,
        ];
        if target.shape() != expected_shape {
            return Err(python_error(anyhow::anyhow!(
                "LLC output has shape {:?}; expected {:?}",
                target.shape(),
                expected_shape
            )));
        }
        let target = target
            .as_slice_mut()
            .context("LLC output must be C-contiguous")
            .map_err(python_error)?;
        py.allow_threads(|| self.read_into_impl(&indexes, &channel_indexes, target))
            .map_err(python_error)
    }

    /// `(time, j, i)` -- the store's time length and this reader's tile size.
    #[getter]
    fn shape(&self) -> (u64, u64, u64) {
        (self.time_len, self.window.height(), self.window.width())
    }

    #[getter]
    fn full_row_reads(&self) -> bool {
        self.full_row_reads
    }
}

/// Read a static `[j, i]` field (`XC`, `rA`, `mask_c`, ...) for the tile.
///
/// Static fields are read once at startup, so this opens and drops the store
/// rather than holding a handle.
#[pyfunction]
#[pyo3(signature = (path, name, face, j_start, j_stop, i_start, i_stop, level=None))]
#[allow(clippy::too_many_arguments)]
fn read_static<'py>(
    py: Python<'py>,
    path: PathBuf,
    name: String,
    face: Option<u64>,
    j_start: u64,
    j_stop: u64,
    i_start: u64,
    i_stop: u64,
    level: Option<u64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let values = py
        .allow_threads(|| -> anyhow::Result<Vec<f32>> {
            let store: ReadableWritableListableStorage = Arc::new(
                FilesystemStore::new(&path)
                    .with_context(|| format!("opening local Zarr store {}", path.display()))?,
            );
            let array_path = if name.starts_with('/') {
                name.clone()
            } else {
                format!("/{name}")
            };
            let array = Array::open(store, &array_path)
                .with_context(|| format!("opening static field {name:?}"))?;
            let names = dimension_names(&array)?;
            let mut start = vec![0u64; names.len()];
            let mut extent = vec![1u64; names.len()];
            for (axis, dim) in names.iter().enumerate() {
                if ROW_DIMS.contains(&dim.as_str()) {
                    start[axis] = j_start;
                    extent[axis] = j_stop - j_start;
                } else if COL_DIMS.contains(&dim.as_str()) {
                    start[axis] = i_start;
                    extent[axis] = i_stop - i_start;
                } else if dim == "face" {
                    start[axis] = face.context("static field has a face axis but no face given")?;
                } else if is_level_dim(dim) {
                    start[axis] = level.unwrap_or(0);
                }
            }
            let subset = ArraySubset::new_with_start_shape(start, extent)
                .context("constructing a static LLC subset")?;
            let options = CodecOptions::default();
            match array.data_type() {
                DataType::Float16 => retrieve_widened::<half::f16>(&array, &subset, &options)
                    .map(|values| values.into_iter().map(f32::from).collect()),
                _ => retrieve_widened::<f32>(&array, &subset, &options),
            }
        })
        .map_err(python_error)?;
    let height = (j_stop - j_start) as usize;
    let width = (i_stop - i_start) as usize;
    numpy::PyArray1::from_vec(py, values)
        .reshape([height, width])
        .map_err(|error| python_error(anyhow::anyhow!("reshaping static field: {error}")))
}

#[pymodule]
fn ocean_llc_loader(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LlcReadPool>()?;
    module.add_class::<LlcPatchReader>()?;
    module.add_function(wrap_pyfunction!(read_static, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{env_flag, env_usize, is_level_dim, validate_index, PlaneCache};
    use std::sync::Arc;

    #[test]
    fn rejects_negative_indexes() {
        assert!(validate_index(-1, 4)
            .unwrap_err()
            .to_string()
            .contains("non-negative"));
    }

    #[test]
    fn rejects_indexes_past_the_end() {
        assert!(validate_index(4, 4)
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
    }

    #[test]
    fn accepts_indexes_inside_the_array() {
        assert_eq!(validate_index(3, 4).unwrap(), 3);
    }

    #[test]
    fn packed_channel_axes_count_as_level_axes() {
        assert!(is_level_dim("k"));
        assert!(is_level_dim("k_p1"));
        assert!(is_level_dim("boundary_channel"));
        assert!(is_level_dim("prognostic_channel"));
        assert!(!is_level_dim("j"));
        assert!(!is_level_dim("time"));
    }

    #[test]
    fn cache_evicts_in_insertion_order() {
        let mut cache = PlaneCache::new(8 * std::mem::size_of::<f32>());
        cache.insert(("a".into(), 0), Arc::new(vec![0.0; 4]));
        cache.insert(("b".into(), 0), Arc::new(vec![0.0; 4]));
        cache.insert(("c".into(), 0), Arc::new(vec![0.0; 4]));
        assert!(cache.get(&("a".into(), 0)).is_none());
        assert!(cache.get(&("c".into(), 0)).is_some());
    }

    #[test]
    fn cache_rejects_oversized_entries() {
        let mut cache = PlaneCache::new(std::mem::size_of::<f32>());
        cache.insert(("a".into(), 0), Arc::new(vec![0.0; 4]));
        assert!(cache.get(&("a".into(), 0)).is_none());
    }

    #[test]
    fn env_helpers_fall_back_to_defaults() {
        assert!(env_flag("OCEAN_RUST_LOADER_UNSET_FLAG", true));
        assert_eq!(env_usize("OCEAN_RUST_LOADER_UNSET_SIZE", 7), 7);
    }
}
