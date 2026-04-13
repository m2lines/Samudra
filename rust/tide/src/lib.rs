use anyhow::{anyhow, bail, ensure, Context, Result};
use ndarray::{s, Array2, Array4};
use numpy::{IntoPyArray, PyArray4};
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

#[derive(Clone, Debug)]
struct ExampleSpec {
    input_time_indices: Vec<Vec<usize>>,
    label_time_indices: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct BatchSpec {
    examples: Vec<ExampleSpec>,
}

#[derive(Clone, Debug)]
struct StepRoot {
    boundary: ValueId,
    label: ValueId,
}

#[derive(Clone, Debug)]
struct Step0Root {
    prognostic: ValueId,
    boundary: ValueId,
    label: ValueId,
}

#[derive(Clone, Debug)]
struct TrainBatchPlan {
    raw_step0: Step0Root,
    raw_later_steps: Vec<StepRoot>,
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
        var: String,
        time: usize,
    },
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
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum OpKey {
    LoadPlane {
        var: String,
        time: usize,
    },
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

    fn intern(&mut self, key: OpKey, make: impl FnOnce() -> Op) -> ValueId {
        if let Some(value) = self.dedupe.get(&key) {
            return *value;
        }

        let out = self.fresh();
        let op = make();
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
            || Op::LoadPlane {
                var: var.to_owned(),
                time,
            },
        )
    }

    fn pack_prognostic(&mut self, inputs: Vec<ValueId>, batch: usize, time: usize) -> ValueId {
        self.intern(
            OpKey::PackPrognostic {
                inputs: inputs.clone(),
                batch,
                time,
            },
            || Op::PackPrognostic {
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
            || Op::PackBoundary {
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
            || Op::PackLabel {
                inputs,
                batch,
                time,
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
        ensure!(
            shape.len() == 3,
            "expected 3-D array for {var}, got {shape:?}"
        );
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
}

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
}

impl RuntimeState {
    fn new() -> Self {
        Self {
            status: HashMap::new(),
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
    fn materialize_raw_step0_parts(&self) -> Result<(Array4<f32>, Array4<f32>, Array4<f32>)> {
        let root = &self.plan.raw_step0;
        let prognostic = self.materialize_tensor(root.prognostic)?;
        let boundary = self.materialize_tensor(root.boundary)?;
        let label = self.materialize_tensor(root.label)?;
        Ok((prognostic, boundary, label))
    }

    fn materialize_raw_step_parts(&self, step: usize) -> Result<(Array4<f32>, Array4<f32>)> {
        let root = self
            .plan
            .raw_later_steps
            .get(step.saturating_sub(1))
            .with_context(|| format!("step {step} is out of range"))?;
        let boundary = self.materialize_tensor(root.boundary)?;
        let label = self.materialize_tensor(root.label)?;
        Ok((boundary, label))
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
                        return Err(anyhow!(
                            "materialization failed for value {value}: {message}"
                        ));
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
}

fn build_batch_plan(dataset: &DatasetInner, spec: &BatchSpec) -> Result<(Graph, TrainBatchPlan)> {
    ensure!(
        !spec.examples.is_empty(),
        "batch must contain at least one example"
    );
    let rollout_steps = spec.examples[0].input_time_indices.len();
    ensure!(
        rollout_steps > 0,
        "batch must contain at least one rollout step"
    );

    for example in &spec.examples {
        ensure!(
            example.input_time_indices.len() == rollout_steps
                && example.label_time_indices.len() == rollout_steps,
            "all examples must agree on rollout length"
        );
    }

    let mut builder = Builder::new();

    let step0_prog_raw = load_packed_tensor(
        &mut builder,
        &dataset.prognostic_vars,
        &spec.examples,
        0,
        TimeSet::Input,
        PackKind::Prognostic,
    )?;

    let step0_boundary_raw = load_packed_tensor(
        &mut builder,
        &dataset.boundary_vars,
        &spec.examples,
        0,
        TimeSet::Input,
        PackKind::Boundary,
    )?;

    let step0_label_raw = load_packed_tensor(
        &mut builder,
        &dataset.prognostic_vars,
        &spec.examples,
        0,
        TimeSet::Label,
        PackKind::Label,
    )?;

    let mut raw_later_steps = Vec::new();
    for step in 1..rollout_steps {
        let boundary_raw = load_packed_tensor(
            &mut builder,
            &dataset.boundary_vars,
            &spec.examples,
            step,
            TimeSet::Input,
            PackKind::Boundary,
        )?;

        let label_raw = load_packed_tensor(
            &mut builder,
            &dataset.prognostic_vars,
            &spec.examples,
            step,
            TimeSet::Label,
            PackKind::Label,
        )?;
        raw_later_steps.push(StepRoot {
            boundary: boundary_raw,
            label: label_raw,
        });
    }

    Ok((
        Graph { ops: builder.ops },
        TrainBatchPlan {
            raw_step0: Step0Root {
                prognostic: step0_prog_raw,
                boundary: step0_boundary_raw,
                label: step0_label_raw,
            },
            raw_later_steps,
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

#[pyclass]
struct Dataset {
    inner: Arc<DatasetInner>,
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(
        data_path: String,
        prognostic_vars: Vec<String>,
        boundary_vars: Vec<String>,
    ) -> PyResult<Self> {
        let path = std::path::Path::new(&data_path);
        if !path.is_absolute() {
            return Err(PyValueError::new_err(format!(
                "data_path must be absolute, got {data_path}"
            )));
        }

        let backend = ZarrBackend::new(&data_path)
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;
        if prognostic_vars.is_empty() && boundary_vars.is_empty() {
            return Err(PyValueError::new_err(
                "at least one zarr variable is required",
            ));
        }

        Ok(Self {
            inner: Arc::new(DatasetInner {
                backend: Arc::new(backend),
                prognostic_vars,
                boundary_vars,
            }),
        })
    }

    fn open_batch(
        &self,
        input_time_indices: Vec<Vec<Vec<usize>>>,
        label_time_indices: Vec<Vec<Vec<usize>>>,
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

        let (graph, plan) = build_batch_plan(&self.inner, &BatchSpec { examples })
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
        self.inner.plan.raw_later_steps.len() + 1
    }

    fn raw_step0_parts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray4<f32>>,
    )> {
        let (prognostic, boundary, label) = py
            .allow_threads(|| self.inner.materialize_raw_step0_parts())
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;
        Ok((
            prognostic.into_pyarray(py),
            boundary.into_pyarray(py),
            label.into_pyarray(py),
        ))
    }

    fn raw_step_parts<'py>(
        &self,
        py: Python<'py>,
        step: usize,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray4<f32>>)> {
        if step == 0 {
            return Err(PyValueError::new_err(
                "raw_step_parts is only valid for rollout steps > 0",
            ));
        }
        let (boundary, label) = py
            .allow_threads(|| self.inner.materialize_raw_step_parts(step))
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))?;
        Ok((boundary.into_pyarray(py), label.into_pyarray(py)))
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
            Ok(Array2::from_elem(
                (2, 2),
                (time as f32) + (var.len() as f32),
            ))
        }
    }

    fn test_dataset(loads: Arc<AtomicUsize>, sleep_ms: u64) -> Arc<DatasetInner> {
        Arc::new(DatasetInner {
            backend: Arc::new(FakeBackend { loads, sleep_ms }),
            prognostic_vars: vec!["thetao_0".to_owned(), "so_0".to_owned()],
            boundary_vars: vec!["hfds".to_owned()],
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
            },
        );

        let (_boundary, _label) = runtime.materialize_raw_step_parts(1).unwrap();
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
            },
        ));

        let left = runtime.clone();
        let right = runtime.clone();
        let t1 = std::thread::spawn(move || left.materialize_raw_step0_parts().unwrap());
        let t2 = std::thread::spawn(move || right.materialize_raw_step0_parts().unwrap());

        let _ = t1.join().unwrap();
        let _ = t2.join().unwrap();

        assert_eq!(loads.load(Ordering::SeqCst), 5);
    }
}
