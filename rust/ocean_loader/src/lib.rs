use anyhow::{anyhow, bail, ensure, Context, Result};
use ndarray::{s, Array2, Array4};
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

type ValueId = usize;
type Store = ReadableWritableListableStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TensorKind {
    Prognostic,
    Boundary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SpatialWindow {
    lat_start: usize,
    lat_end: usize,
    lon_start: usize,
    lon_end: usize,
}

impl SpatialWindow {
    fn full(lat: usize, lon: usize) -> Self {
        Self {
            lat_start: 0,
            lat_end: lat,
            lon_start: 0,
            lon_end: lon,
        }
    }

    fn height(&self) -> usize {
        self.lat_end - self.lat_start
    }

    fn width(&self) -> usize {
        self.lon_end - self.lon_start
    }
}

#[derive(Clone, Debug)]
struct ExampleSpec {
    input_time_indices: Vec<Vec<usize>>,
    label_time_indices: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct BatchSpec {
    examples: Vec<ExampleSpec>,
    spatial_window: SpatialWindow,
}

#[derive(Clone, Debug)]
struct StepRoot {
    boundary: ValueId,
    label: ValueId,
}

#[derive(Clone, Debug)]
struct TrainBatchPlan {
    step0_input: ValueId,
    step0_label: ValueId,
    later_steps: Vec<StepRoot>,
}

#[derive(Clone, Debug)]
struct Graph {
    ops: Vec<Op>,
}

impl Graph {
    fn op(&self, value: ValueId) -> &Op {
        &self.ops[value]
    }
}

#[derive(Clone, Debug)]
enum Op {
    LoadPlane {
        out: ValueId,
        var: String,
        time: usize,
    },
    Crop {
        out: ValueId,
        input: ValueId,
        window: SpatialWindow,
    },
    PackPrognostic {
        out: ValueId,
        inputs: Vec<ValueId>,
        batch: usize,
        time: usize,
    },
    PackBoundary {
        out: ValueId,
        inputs: Vec<ValueId>,
        batch: usize,
        time: usize,
    },
    PackLabel {
        out: ValueId,
        inputs: Vec<ValueId>,
        batch: usize,
        time: usize,
    },
    Normalize {
        out: ValueId,
        input: ValueId,
        kind: TensorKind,
    },
    ApplyMask {
        out: ValueId,
        input: ValueId,
        kind: TensorKind,
        fill_value_bits: u32,
    },
    ConcatInput {
        out: ValueId,
        prognostic: ValueId,
        boundary: ValueId,
    },
}

impl Op {
    fn out(&self) -> ValueId {
        match self {
            Self::LoadPlane { out, .. }
            | Self::Crop { out, .. }
            | Self::PackPrognostic { out, .. }
            | Self::PackBoundary { out, .. }
            | Self::PackLabel { out, .. }
            | Self::Normalize { out, .. }
            | Self::ApplyMask { out, .. }
            | Self::ConcatInput { out, .. } => *out,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum OpKey {
    LoadPlane { var: String, time: usize },
    Crop { input: ValueId, window: SpatialWindow },
    PackPrognostic {
        inputs: Vec<ValueId>,
        batch: usize,
        time: usize,
    },
    PackBoundary {
        inputs: Vec<ValueId>,
        batch: usize,
        time: usize,
    },
    PackLabel {
        inputs: Vec<ValueId>,
        batch: usize,
        time: usize,
    },
    Normalize { input: ValueId, kind: TensorKind },
    ApplyMask {
        input: ValueId,
        kind: TensorKind,
        fill_value_bits: u32,
    },
    ConcatInput {
        prognostic: ValueId,
        boundary: ValueId,
    },
}

struct Builder {
    ops: Vec<Op>,
    dedupe: HashMap<OpKey, ValueId>,
}

impl Builder {
    fn new() -> Self {
        Self {
            ops: Vec::new(),
            dedupe: HashMap::new(),
        }
    }

    fn fresh(&self) -> ValueId {
        self.ops.len()
    }

    fn intern(&mut self, key: OpKey, make: impl FnOnce(ValueId) -> Op) -> ValueId {
        if let Some(value) = self.dedupe.get(&key) {
            return *value;
        }

        let out = self.fresh();
        let op = make(out);
        self.ops.push(op);
        self.dedupe.insert(key, out);
        out
    }

    fn load_plane(&mut self, var: &str, time: usize) -> ValueId {
        self.intern(
            OpKey::LoadPlane {
                var: var.to_owned(),
                time,
            },
            |out| Op::LoadPlane {
                out,
                var: var.to_owned(),
                time,
            },
        )
    }

    fn crop(&mut self, input: ValueId, window: SpatialWindow, is_full: bool) -> ValueId {
        if is_full {
            input
        } else {
            self.intern(OpKey::Crop { input, window }, |out| Op::Crop {
                out,
                input,
                window,
            })
        }
    }

    fn pack_prognostic(&mut self, inputs: Vec<ValueId>, batch: usize, time: usize) -> ValueId {
        self.intern(
            OpKey::PackPrognostic {
                inputs: inputs.clone(),
                batch,
                time,
            },
            |out| Op::PackPrognostic {
                out,
                inputs,
                batch,
                time,
            },
        )
    }

    fn pack_boundary(&mut self, inputs: Vec<ValueId>, batch: usize, time: usize) -> ValueId {
        self.intern(
            OpKey::PackBoundary {
                inputs: inputs.clone(),
                batch,
                time,
            },
            |out| Op::PackBoundary {
                out,
                inputs,
                batch,
                time,
            },
        )
    }

    fn pack_label(&mut self, inputs: Vec<ValueId>, batch: usize, time: usize) -> ValueId {
        self.intern(
            OpKey::PackLabel {
                inputs: inputs.clone(),
                batch,
                time,
            },
            |out| Op::PackLabel {
                out,
                inputs,
                batch,
                time,
            },
        )
    }

    fn normalize(&mut self, input: ValueId, kind: TensorKind) -> ValueId {
        self.intern(OpKey::Normalize { input, kind }, |out| Op::Normalize {
            out,
            input,
            kind,
        })
    }

    fn apply_mask(&mut self, input: ValueId, kind: TensorKind, fill_value: f32) -> ValueId {
        self.intern(
            OpKey::ApplyMask {
                input,
                kind,
                fill_value_bits: fill_value.to_bits(),
            },
            |out| Op::ApplyMask {
                out,
                input,
                kind,
                fill_value_bits: fill_value.to_bits(),
            },
        )
    }

    fn concat_input(&mut self, prognostic: ValueId, boundary: ValueId) -> ValueId {
        self.intern(
            OpKey::ConcatInput {
                prognostic,
                boundary,
            },
            |out| Op::ConcatInput {
                out,
                prognostic,
                boundary,
            },
        )
    }
}

trait PlaneBackend: Send + Sync {
    fn load_plane(&self, var: &str, time: usize) -> Result<Array2<f32>>;
}

#[derive(Clone)]
struct ZarrBackend {
    store: Store,
}

impl ZarrBackend {
    fn new(data_path: &str) -> Result<Self> {
        let store: Store = Arc::new(
            FilesystemStore::new(data_path)
                .with_context(|| format!("opening zarr store at {data_path}"))?,
        );
        Ok(Self { store })
    }
}

impl PlaneBackend for ZarrBackend {
    fn load_plane(&self, var: &str, time: usize) -> Result<Array2<f32>> {
        let array = Array::open(self.store.clone(), var)
            .or_else(|_| Array::open(self.store.clone(), &format!("/{var}")))
            .with_context(|| format!("opening array {var}"))?;
        let shape = array.shape().to_vec();
        ensure!(shape.len() == 3, "expected 3-D array for {var}, got {shape:?}");
        let lat = shape[1] as usize;
        let lon = shape[2] as usize;

        let subset = ArraySubset::new_with_start_shape(
            vec![time as u64, 0, 0],
            vec![1, lat as u64, lon as u64],
        )?;

        let values_f32 = array
            .retrieve_array_subset_elements::<f32>(&subset)
            .ok()
            .or_else(|| {
                array
                    .retrieve_array_subset_elements::<f64>(&subset)
                    .ok()
                    .map(|vals| vals.into_iter().map(|v| v as f32).collect())
            })
            .with_context(|| format!("reading array {var} at time index {time}"))?;

        Array2::from_shape_vec((lat, lon), values_f32).context("reshaping plane")
    }
}

#[derive(Clone)]
struct DatasetInner {
    backend: Arc<dyn PlaneBackend>,
    prognostic_vars: Vec<String>,
    boundary_vars: Vec<String>,
    prognostic_mean: Vec<f32>,
    prognostic_std: Vec<f32>,
    boundary_mean: Vec<f32>,
    boundary_std: Vec<f32>,
    prognostic_mask: Array3Bool,
    boundary_mask: Array2Bool,
    normalize_before_mask: bool,
    masked_fill_value: f32,
    lat: usize,
    lon: usize,
    cpu_budget_bytes: usize,
    chunk_read_concurrency: usize,
    decode_concurrency: usize,
}

type Array3Bool = ndarray::Array3<bool>;
type Array2Bool = ndarray::Array2<bool>;

#[derive(Clone, Debug)]
enum ValueData {
    Plane(Array2<f32>),
    Tensor(Array4<f32>),
}

#[derive(Clone, Debug)]
enum ValueStatus {
    Running,
    Ready(Arc<ValueData>),
    Failed(String),
}

struct RuntimeState {
    status: HashMap<ValueId, ValueStatus>,
    cache_bytes: usize,
}

impl RuntimeState {
    fn new() -> Self {
        Self {
            status: HashMap::new(),
            cache_bytes: 0,
        }
    }
}

struct BatchRuntime {
    dataset: Arc<DatasetInner>,
    graph: Arc<Graph>,
    plan: Arc<TrainBatchPlan>,
    state: Mutex<RuntimeState>,
    condvar: Condvar,
}

impl BatchRuntime {
    fn materialize_step0(&self) -> Result<(Array4<f32>, Array4<f32>)> {
        let input = self.materialize_tensor(self.plan.step0_input)?;
        let label = self.materialize_tensor(self.plan.step0_label)?;
        Ok((input, label))
    }

    fn materialize_step_parts(&self, step: usize) -> Result<(Array4<f32>, Array4<f32>)> {
        let root = self
            .plan
            .later_steps
            .get(step.saturating_sub(1))
            .with_context(|| format!("step {step} is out of range"))?;
        let boundary = self.materialize_tensor(root.boundary)?;
        let label = self.materialize_tensor(root.label)?;
        Ok((boundary, label))
    }

    fn prefetch_step(&self, step: usize) -> Result<()> {
        if step == 0 {
            let _ = self.materialize_step0()?;
        } else {
            let _ = self.materialize_step_parts(step)?;
        }
        Ok(())
    }

    fn materialize_tensor(&self, value: ValueId) -> Result<Array4<f32>> {
        match &*self.materialize_value(value)? {
            ValueData::Tensor(tensor) => Ok(tensor.clone()),
            ValueData::Plane(_) => bail!("value {value} is a plane, not a tensor"),
        }
    }

    fn materialize_value(&self, value: ValueId) -> Result<Arc<ValueData>> {
        loop {
            let should_compute = {
                let mut state = self.state.lock().unwrap();
                match state.status.get(&value) {
                    Some(ValueStatus::Ready(data)) => return Ok(data.clone()),
                    Some(ValueStatus::Failed(message)) => {
                        return Err(anyhow!("materialization failed for value {value}: {message}"));
                    }
                    Some(ValueStatus::Running) => {
                        state = self.condvar.wait(state).unwrap();
                        drop(state);
                        false
                    }
                    None => {
                        state.status.insert(value, ValueStatus::Running);
                        true
                    }
                }
            };

            if !should_compute {
                continue;
            }

            let result = self.compute_value(value);
            let mut state = self.state.lock().unwrap();
            match &result {
                Ok(data) => {
                    state.cache_bytes += estimate_bytes(data.as_ref());
                    let _ = self.dataset.cpu_budget_bytes;
                    let _ = self.dataset.chunk_read_concurrency;
                    let _ = self.dataset.decode_concurrency;
                    state.status.insert(value, ValueStatus::Ready(data.clone()));
                }
                Err(err) => {
                    state
                        .status
                        .insert(value, ValueStatus::Failed(format!("{err:#}")));
                }
            }
            self.condvar.notify_all();
            return result;
        }
    }

    fn compute_value(&self, value: ValueId) -> Result<Arc<ValueData>> {
        match self.graph.op(value) {
            Op::LoadPlane { var, time, .. } => Ok(Arc::new(ValueData::Plane(
                self.dataset.backend.load_plane(var, *time)?,
            ))),
            Op::Crop { input, window, .. } => {
                let plane = self.materialize_plane(*input)?;
                Ok(Arc::new(ValueData::Plane(
                    plane
                        .slice(s![
                            window.lat_start..window.lat_end,
                            window.lon_start..window.lon_end
                        ])
                        .to_owned(),
                )))
            }
            Op::PackPrognostic {
                inputs,
                batch,
                time,
                ..
            } => Ok(Arc::new(ValueData::Tensor(self.pack_tensor(
                inputs,
                *batch,
                *time,
                self.dataset.prognostic_vars.len(),
            )?))),
            Op::PackBoundary {
                inputs,
                batch,
                time,
                ..
            } => Ok(Arc::new(ValueData::Tensor(self.pack_tensor(
                inputs,
                *batch,
                *time,
                self.dataset.boundary_vars.len(),
            )?))),
            Op::PackLabel {
                inputs,
                batch,
                time,
                ..
            } => Ok(Arc::new(ValueData::Tensor(self.pack_tensor(
                inputs,
                *batch,
                *time,
                self.dataset.prognostic_vars.len(),
            )?))),
            Op::Normalize { input, kind, .. } => {
                let tensor = self.materialize_tensor(*input)?;
                Ok(Arc::new(ValueData::Tensor(self.normalize_tensor(tensor, *kind)?)))
            }
            Op::ApplyMask {
                input,
                kind,
                fill_value_bits,
                ..
            } => {
                let tensor = self.materialize_tensor(*input)?;
                Ok(Arc::new(ValueData::Tensor(self.apply_mask(
                    tensor,
                    *kind,
                    f32::from_bits(*fill_value_bits),
                )?)))
            }
            Op::ConcatInput {
                prognostic,
                boundary,
                ..
            } => {
                let (prog, bound) = rayon::join(
                    || self.materialize_tensor(*prognostic),
                    || self.materialize_tensor(*boundary),
                );
                let prog = prog?;
                let bound = bound?;
                ensure!(
                    prog.shape()[0] == bound.shape()[0]
                        && prog.shape()[2] == bound.shape()[2]
                        && prog.shape()[3] == bound.shape()[3],
                    "concat inputs must agree on batch/height/width"
                );
                let batch = prog.shape()[0];
                let channels = prog.shape()[1] + bound.shape()[1];
                let lat = prog.shape()[2];
                let lon = prog.shape()[3];
                let mut out = Array4::<f32>::zeros((batch, channels, lat, lon));
                let prog_channels = prog.shape()[1];
                out.slice_mut(s![.., 0..prog_channels, .., ..]).assign(&prog);
                out.slice_mut(s![.., prog_channels.., .., ..]).assign(&bound);
                Ok(Arc::new(ValueData::Tensor(out)))
            }
        }
    }

    fn materialize_plane(&self, value: ValueId) -> Result<Array2<f32>> {
        match &*self.materialize_value(value)? {
            ValueData::Plane(plane) => Ok(plane.clone()),
            ValueData::Tensor(_) => bail!("value {value} is a tensor, not a plane"),
        }
    }

    fn pack_tensor(
        &self,
        inputs: &[ValueId],
        batch: usize,
        time: usize,
        vars: usize,
    ) -> Result<Array4<f32>> {
        let planes = inputs
            .par_iter()
            .map(|value| self.materialize_plane(*value))
            .collect::<Vec<_>>();
        let planes = planes.into_iter().collect::<Result<Vec<_>>>()?;
        ensure!(
            planes.len() == batch * time * vars,
            "expected {} planes, got {}",
            batch * time * vars,
            planes.len()
        );

        let lat = planes[0].shape()[0];
        let lon = planes[0].shape()[1];
        let channels = time * vars;
        let mut out = Array4::<f32>::zeros((batch, channels, lat, lon));

        for batch_idx in 0..batch {
            for time_idx in 0..time {
                for var_idx in 0..vars {
                    let plane_index = ((batch_idx * time) + time_idx) * vars + var_idx;
                    let channel = time_idx * vars + var_idx;
                    out.slice_mut(s![batch_idx, channel, .., ..])
                        .assign(&planes[plane_index]);
                }
            }
        }

        Ok(out)
    }

    fn normalize_tensor(&self, tensor: Array4<f32>, kind: TensorKind) -> Result<Array4<f32>> {
        let (means, stds) = match kind {
            TensorKind::Prognostic => (
                &self.dataset.prognostic_mean,
                &self.dataset.prognostic_std,
            ),
            TensorKind::Boundary => (&self.dataset.boundary_mean, &self.dataset.boundary_std),
        };
        let base_channels = means.len();
        ensure!(base_channels > 0, "normalization statistics must not be empty");
        ensure!(
            tensor.shape()[1] % base_channels == 0,
            "channel count {} is not divisible by statistics length {}",
            tensor.shape()[1],
            base_channels
        );

        let mut out = tensor;
        for batch_idx in 0..out.shape()[0] {
            for channel in 0..out.shape()[1] {
                let stat_idx = channel % base_channels;
                let mean = means[stat_idx];
                let std = stds[stat_idx];
                out.slice_mut(s![batch_idx, channel, .., ..])
                    .mapv_inplace(|v| {
                        let norm = (v - mean) / std;
                        if norm.is_nan() { 0.0 } else { norm }
                    });
            }
        }
        Ok(out)
    }

    fn apply_mask(
        &self,
        tensor: Array4<f32>,
        kind: TensorKind,
        fill_value: f32,
    ) -> Result<Array4<f32>> {
        let mut out = tensor;
        match kind {
            TensorKind::Prognostic => {
                let mask = &self.dataset.prognostic_mask;
                ensure!(
                    out.shape()[1] % mask.shape()[0] == 0,
                    "prognostic channels {} must be divisible by mask channels {}",
                    out.shape()[1],
                    mask.shape()[0]
                );
                for batch_idx in 0..out.shape()[0] {
                    for channel in 0..out.shape()[1] {
                        let mask_idx = channel % mask.shape()[0];
                        for lat in 0..out.shape()[2] {
                            for lon in 0..out.shape()[3] {
                                if !mask[(mask_idx, lat, lon)] {
                                    out[(batch_idx, channel, lat, lon)] = fill_value;
                                }
                            }
                        }
                    }
                }
            }
            TensorKind::Boundary => {
                let mask = &self.dataset.boundary_mask;
                for batch_idx in 0..out.shape()[0] {
                    for channel in 0..out.shape()[1] {
                        for lat in 0..out.shape()[2] {
                            for lon in 0..out.shape()[3] {
                                if !mask[(lat, lon)] {
                                    out[(batch_idx, channel, lat, lon)] = fill_value;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

fn estimate_bytes(data: &ValueData) -> usize {
    match data {
        ValueData::Plane(plane) => plane.len() * std::mem::size_of::<f32>(),
        ValueData::Tensor(tensor) => tensor.len() * std::mem::size_of::<f32>(),
    }
}

fn build_batch_plan(dataset: &DatasetInner, spec: &BatchSpec) -> Result<(Graph, TrainBatchPlan)> {
    ensure!(!spec.examples.is_empty(), "batch must contain at least one example");
    let rollout_steps = spec.examples[0].input_time_indices.len();
    ensure!(rollout_steps > 0, "batch must contain at least one rollout step");

    for example in &spec.examples {
        ensure!(
            example.input_time_indices.len() == rollout_steps
                && example.label_time_indices.len() == rollout_steps,
            "all examples must agree on rollout length"
        );
    }

    let full_window = SpatialWindow::full(dataset.lat, dataset.lon);
    let is_full = spec.spatial_window == full_window;

    let mut builder = Builder::new();

    let step0_prog = load_packed_tensor(
        &mut builder,
        &dataset.prognostic_vars,
        &spec.examples,
        0,
        TimeSet::Input,
        spec.spatial_window,
        is_full,
        PackKind::Prognostic,
    )?;
    let step0_prog = normalize_and_mask(
        &mut builder,
        step0_prog,
        TensorKind::Prognostic,
        dataset.normalize_before_mask,
        dataset.masked_fill_value,
    );

    let step0_boundary = load_packed_tensor(
        &mut builder,
        &dataset.boundary_vars,
        &spec.examples,
        0,
        TimeSet::Input,
        spec.spatial_window,
        is_full,
        PackKind::Boundary,
    )?;
    let step0_boundary = normalize_and_mask(
        &mut builder,
        step0_boundary,
        TensorKind::Boundary,
        dataset.normalize_before_mask,
        dataset.masked_fill_value,
    );

    let step0_input = builder.concat_input(step0_prog, step0_boundary);

    let step0_label = load_packed_tensor(
        &mut builder,
        &dataset.prognostic_vars,
        &spec.examples,
        0,
        TimeSet::Label,
        spec.spatial_window,
        is_full,
        PackKind::Label,
    )?;
    let step0_label = normalize_and_mask(
        &mut builder,
        step0_label,
        TensorKind::Prognostic,
        dataset.normalize_before_mask,
        dataset.masked_fill_value,
    );

    let mut later_steps = Vec::new();
    for step in 1..rollout_steps {
        let boundary = load_packed_tensor(
            &mut builder,
            &dataset.boundary_vars,
            &spec.examples,
            step,
            TimeSet::Input,
            spec.spatial_window,
            is_full,
            PackKind::Boundary,
        )?;
        let boundary = normalize_and_mask(
            &mut builder,
            boundary,
            TensorKind::Boundary,
            dataset.normalize_before_mask,
            dataset.masked_fill_value,
        );

        let label = load_packed_tensor(
            &mut builder,
            &dataset.prognostic_vars,
            &spec.examples,
            step,
            TimeSet::Label,
            spec.spatial_window,
            is_full,
            PackKind::Label,
        )?;
        let label = normalize_and_mask(
            &mut builder,
            label,
            TensorKind::Prognostic,
            dataset.normalize_before_mask,
            dataset.masked_fill_value,
        );
        later_steps.push(StepRoot { boundary, label });
    }

    Ok((
        Graph { ops: builder.ops },
        TrainBatchPlan {
            step0_input,
            step0_label,
            later_steps,
        },
    ))
}

#[derive(Clone, Copy)]
enum TimeSet {
    Input,
    Label,
}

#[derive(Clone, Copy)]
enum PackKind {
    Prognostic,
    Boundary,
    Label,
}

fn load_packed_tensor(
    builder: &mut Builder,
    vars: &[String],
    examples: &[ExampleSpec],
    step: usize,
    time_set: TimeSet,
    window: SpatialWindow,
    is_full_window: bool,
    pack_kind: PackKind,
) -> Result<ValueId> {
    let time_count = match time_set {
        TimeSet::Input => examples[0].input_time_indices[step].len(),
        TimeSet::Label => examples[0].label_time_indices[step].len(),
    };
    let mut planes = Vec::with_capacity(examples.len() * time_count * vars.len());
    for example in examples {
        let times = match time_set {
            TimeSet::Input => &example.input_time_indices[step],
            TimeSet::Label => &example.label_time_indices[step],
        };
        ensure!(
            times.len() == time_count,
            "all examples must agree on time count for step {step}"
        );

        for &time in times {
            for var in vars {
                let plane = builder.load_plane(var, time);
                let plane = builder.crop(plane, window, is_full_window);
                planes.push(plane);
            }
        }
    }

    let batch = examples.len();
    let value = match pack_kind {
        PackKind::Prognostic => builder.pack_prognostic(planes, batch, time_count),
        PackKind::Boundary => builder.pack_boundary(planes, batch, time_count),
        PackKind::Label => builder.pack_label(planes, batch, time_count),
    };
    Ok(value)
}

fn normalize_and_mask(
    builder: &mut Builder,
    input: ValueId,
    kind: TensorKind,
    normalize_before_mask: bool,
    masked_fill_value: f32,
) -> ValueId {
    if normalize_before_mask {
        let normalized = builder.normalize(input, kind);
        builder.apply_mask(normalized, kind, masked_fill_value)
    } else {
        let masked = builder.apply_mask(input, kind, masked_fill_value);
        builder.normalize(masked, kind)
    }
}

#[pyclass]
struct Dataset {
    inner: Arc<DatasetInner>,
}

#[pymethods]
impl Dataset {
    #[new]
    #[pyo3(
        signature = (
            data_path,
            prognostic_vars,
            boundary_vars,
            prognostic_mean,
            prognostic_std,
            boundary_mean,
            boundary_std,
            prognostic_mask,
            boundary_mask,
            normalize_before_mask,
            masked_fill_value,
            cpu_budget_bytes=1 << 30,
            chunk_read_concurrency=4,
            decode_concurrency=4
        )
    )]
    fn new(
        data_path: String,
        prognostic_vars: Vec<String>,
        boundary_vars: Vec<String>,
        prognostic_mean: PyReadonlyArray1<'_, f32>,
        prognostic_std: PyReadonlyArray1<'_, f32>,
        boundary_mean: PyReadonlyArray1<'_, f32>,
        boundary_std: PyReadonlyArray1<'_, f32>,
        prognostic_mask: PyReadonlyArray3<'_, bool>,
        boundary_mask: PyReadonlyArray2<'_, bool>,
        normalize_before_mask: bool,
        masked_fill_value: f32,
        cpu_budget_bytes: usize,
        chunk_read_concurrency: usize,
        decode_concurrency: usize,
    ) -> PyResult<Self> {
        let path = std::path::Path::new(&data_path);
        if !path.is_absolute() {
            return Err(PyValueError::new_err(format!(
                "data_path must be absolute, got {data_path}"
            )));
        }

        let backend = ZarrBackend::new(&data_path)
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;

        let prognostic_mask = prognostic_mask.as_array().to_owned();
        let boundary_mask = boundary_mask.as_array().to_owned();
        let lat = boundary_mask.shape()[0];
        let lon = boundary_mask.shape()[1];

        if prognostic_mask.shape()[1] != lat || prognostic_mask.shape()[2] != lon {
            return Err(PyValueError::new_err(
                "prognostic_mask spatial shape must match boundary_mask",
            ));
        }

        Ok(Self {
            inner: Arc::new(DatasetInner {
                backend: Arc::new(backend),
                prognostic_vars,
                boundary_vars,
                prognostic_mean: prognostic_mean.as_slice()?.to_vec(),
                prognostic_std: prognostic_std.as_slice()?.to_vec(),
                boundary_mean: boundary_mean.as_slice()?.to_vec(),
                boundary_std: boundary_std.as_slice()?.to_vec(),
                prognostic_mask,
                boundary_mask,
                normalize_before_mask,
                masked_fill_value,
                lat,
                lon,
                cpu_budget_bytes,
                chunk_read_concurrency,
                decode_concurrency,
            }),
        })
    }

    fn open_batch(
        &self,
        input_time_indices: Vec<Vec<Vec<usize>>>,
        label_time_indices: Vec<Vec<Vec<usize>>>,
        spatial_window: Option<(usize, usize, usize, usize)>,
    ) -> PyResult<RustTrainBatch> {
        if input_time_indices.is_empty() || input_time_indices.len() != label_time_indices.len() {
            return Err(PyValueError::new_err(
                "input_time_indices and label_time_indices must be non-empty and match in length",
            ));
        }

        let examples = input_time_indices
            .into_iter()
            .zip(label_time_indices)
            .map(|(input_time_indices, label_time_indices)| ExampleSpec {
                input_time_indices,
                label_time_indices,
            })
            .collect::<Vec<_>>();

        let window = match spatial_window {
            Some((lat_start, lat_end, lon_start, lon_end)) => SpatialWindow {
                lat_start,
                lat_end,
                lon_start,
                lon_end,
            },
            None => SpatialWindow::full(self.inner.lat, self.inner.lon),
        };

        let (graph, plan) = build_batch_plan(
            &self.inner,
            &BatchSpec {
                examples,
                spatial_window: window,
            },
        )
        .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;

        Ok(RustTrainBatch {
            inner: Arc::new(BatchRuntime {
                dataset: self.inner.clone(),
                graph: Arc::new(graph),
                plan: Arc::new(plan),
                state: Mutex::new(RuntimeState::new()),
                condvar: Condvar::new(),
            }),
        })
    }
}

#[pyclass]
struct RustTrainBatch {
    inner: Arc<BatchRuntime>,
}

#[pymethods]
impl RustTrainBatch {
    fn num_steps(&self) -> usize {
        self.inner.plan.later_steps.len() + 1
    }

    fn step0<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray4<f32>>)> {
        let (input, label) = py
            .allow_threads(|| self.inner.materialize_step0())
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;
        Ok((input.into_pyarray(py), label.into_pyarray(py)))
    }

    fn step_parts<'py>(
        &self,
        py: Python<'py>,
        step: usize,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray4<f32>>)> {
        if step == 0 {
            return Err(PyValueError::new_err(
                "step_parts is only valid for rollout steps > 0",
            ));
        }
        let (boundary, label) = py
            .allow_threads(|| self.inner.materialize_step_parts(step))
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;
        Ok((boundary.into_pyarray(py), label.into_pyarray(py)))
    }

    fn prefetch_step(&self, py: Python<'_>, step: usize) -> PyResult<()> {
        py.allow_threads(|| self.inner.prefetch_step(step))
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;
        Ok(())
    }
}

#[pymodule]
fn tide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<RustTrainBatch>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Clone)]
    struct FakeBackend {
        loads: Arc<AtomicUsize>,
        sleep_ms: u64,
    }

    impl PlaneBackend for FakeBackend {
        fn load_plane(&self, var: &str, time: usize) -> Result<Array2<f32>> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            if self.sleep_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.sleep_ms));
            }
            Ok(Array2::from_elem((2, 2), (time as f32) + (var.len() as f32)))
        }
    }

    fn test_dataset(loads: Arc<AtomicUsize>, sleep_ms: u64) -> Arc<DatasetInner> {
        Arc::new(DatasetInner {
            backend: Arc::new(FakeBackend { loads, sleep_ms }),
            prognostic_vars: vec!["thetao_0".to_owned(), "so_0".to_owned()],
            boundary_vars: vec!["hfds".to_owned()],
            prognostic_mean: vec![0.0, 0.0],
            prognostic_std: vec![1.0, 1.0],
            boundary_mean: vec![0.0],
            boundary_std: vec![1.0],
            prognostic_mask: ndarray::Array3::from_elem((2, 2, 2), true),
            boundary_mask: ndarray::Array2::from_elem((2, 2), true),
            normalize_before_mask: true,
            masked_fill_value: 0.0,
            lat: 2,
            lon: 2,
            cpu_budget_bytes: 1 << 20,
            chunk_read_concurrency: 2,
            decode_concurrency: 2,
        })
    }

    fn build_runtime(dataset: Arc<DatasetInner>, spec: BatchSpec) -> BatchRuntime {
        let (graph, plan) = build_batch_plan(&dataset, &spec).unwrap();
        BatchRuntime {
            dataset,
            graph: Arc::new(graph),
            plan: Arc::new(plan),
            state: Mutex::new(RuntimeState::new()),
            condvar: Condvar::new(),
        }
    }

    #[test]
    fn builder_excludes_later_prognostic_input() {
        let loads = Arc::new(AtomicUsize::new(0));
        let dataset = test_dataset(loads, 0);
        let runtime = build_runtime(
            dataset,
            BatchSpec {
                examples: vec![ExampleSpec {
                    input_time_indices: vec![vec![0], vec![1]],
                    label_time_indices: vec![vec![1], vec![2]],
                }],
                spatial_window: SpatialWindow::full(2, 2),
            },
        );

        let prognostic_at_time1 = runtime
            .graph
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::LoadPlane { var, time, .. } if *time == 1 && var.starts_with("thetao") => {
                    Some(())
                }
                _ => None,
            })
            .count();

        assert_eq!(prognostic_at_time1, 1);
    }

    #[test]
    fn builder_reuses_repeated_loads() {
        let loads = Arc::new(AtomicUsize::new(0));
        let dataset = test_dataset(loads, 0);
        let runtime = build_runtime(
            dataset,
            BatchSpec {
                examples: vec![
                    ExampleSpec {
                        input_time_indices: vec![vec![0]],
                        label_time_indices: vec![vec![1]],
                    },
                    ExampleSpec {
                        input_time_indices: vec![vec![0]],
                        label_time_indices: vec![vec![1]],
                    },
                ],
                spatial_window: SpatialWindow::full(2, 2),
            },
        );

        let load_ops = runtime
            .graph
            .ops
            .iter()
            .filter(|op| matches!(op, Op::LoadPlane { .. }))
            .count();

        assert_eq!(load_ops, 5);
    }

    #[test]
    fn materializing_step_does_not_force_other_steps() {
        let loads = Arc::new(AtomicUsize::new(0));
        let dataset = test_dataset(loads.clone(), 0);
        let runtime = build_runtime(
            dataset,
            BatchSpec {
                examples: vec![ExampleSpec {
                    input_time_indices: vec![vec![0], vec![1], vec![2]],
                    label_time_indices: vec![vec![1], vec![2], vec![3]],
                }],
                spatial_window: SpatialWindow::full(2, 2),
            },
        );

        let (_boundary, _label) = runtime.materialize_step_parts(1).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn concurrent_materialization_shares_inflight_work() {
        let loads = Arc::new(AtomicUsize::new(0));
        let dataset = test_dataset(loads.clone(), 25);
        let runtime = Arc::new(build_runtime(
            dataset,
            BatchSpec {
                examples: vec![ExampleSpec {
                    input_time_indices: vec![vec![0]],
                    label_time_indices: vec![vec![1]],
                }],
                spatial_window: SpatialWindow::full(2, 2),
            },
        ));

        let left = runtime.clone();
        let right = runtime.clone();
        let t1 = std::thread::spawn(move || left.materialize_step0().unwrap());
        let t2 = std::thread::spawn(move || right.materialize_step0().unwrap());

        let _ = t1.join().unwrap();
        let _ = t2.join().unwrap();

        assert_eq!(loads.load(Ordering::SeqCst), 5);
    }
}
