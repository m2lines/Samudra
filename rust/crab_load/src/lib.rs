// SPDX-FileCopyrightText: 2026 Samudra Authors
//
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{bail, Context};
use numpy::{IntoPyArray, PyArray4, PyReadwriteArray4, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use zarrs::{
    array::{Array, DataType},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits},
};

type OpenArray = Array<dyn ReadableWritableListableStorageTraits>;
type ReadShape = (usize, usize, usize, usize);
type ReadOutput = (ReadShape, Vec<f32>);

const XARRAY_DIMENSIONS_ATTRIBUTE: &str = "_ARRAY_DIMENSIONS";

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
        return names
            .iter()
            .map(|name| {
                name.clone()
                    .context(format!("array {} has an unnamed dimension", array.path()))
            })
            .collect();
    }

    let names = array
        .attributes()
        .get(XARRAY_DIMENSIONS_ATTRIBUTE)
        .with_context(|| {
            format!(
                "array {} has no dimension metadata; expected (time, lat/y, lon/x)",
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

fn validate_flat_om4_dimensions(array: &OpenArray) -> anyhow::Result<()> {
    let names = dimension_names(array)?;
    let valid = names.len() == 3
        && names[0] == "time"
        && matches!(names[1].as_str(), "lat" | "y")
        && matches!(names[2].as_str(), "lon" | "x");
    if !valid {
        bail!(
            "array {} has dimensions {names:?}; expected (time, lat/y, lon/x)",
            array.path()
        );
    }
    Ok(())
}

fn validate_compact_om4_dimensions(
    array: &OpenArray,
    depth_resolved: bool,
) -> anyhow::Result<(usize, Option<usize>)> {
    let names = dimension_names(array)?;
    let axes = if depth_resolved {
        let leading_axes = match names.as_slice() {
            [time, lev, ..] if time == "time" && lev == "lev" => Some((0, Some(1))),
            [lev, time, ..] if lev == "lev" && time == "time" => Some((1, Some(0))),
            _ => None,
        };
        leading_axes.filter(|_| {
            names.len() == 4
                && matches!(names[2].as_str(), "lat" | "y")
                && matches!(names[3].as_str(), "lon" | "x")
        })
    } else {
        (names.len() == 3
            && names[0] == "time"
            && matches!(names[1].as_str(), "lat" | "y")
            && matches!(names[2].as_str(), "lon" | "x"))
        .then_some((0, None))
    };
    if axes.is_none() {
        let expected = if depth_resolved {
            "(time, lev, lat/y, lon/x) or (lev, time, lat/y, lon/x)"
        } else {
            "(time, lat/y, lon/x)"
        };
        bail!(
            "array {} has dimensions {names:?}; expected {expected}",
            array.path()
        );
    }
    Ok(axes.expect("axes was checked above"))
}

fn compact_variable_selector(logical_name: &str) -> anyhow::Result<(&str, Option<u64>)> {
    if logical_name.is_empty() {
        bail!("variable names must not be empty");
    }
    if let Some((base_name, suffix)) = logical_name.rsplit_once('_') {
        if !base_name.is_empty()
            && !suffix.is_empty()
            && suffix.bytes().all(|byte| byte.is_ascii_digit())
        {
            let level = suffix.parse::<u64>().with_context(|| {
                format!("parsing level from compact OM4 variable {logical_name:?}")
            })?;
            return Ok((base_name, Some(level)));
        }
    }
    Ok((logical_name, None))
}

/// Shared bounded Rayon pool for every flat-OM4 reader in one training process.
#[pyclass]
struct FlatOm4ReadPool {
    thread_pool: Arc<ThreadPool>,
    max_concurrent_reads: usize,
}

#[pymethods]
impl FlatOm4ReadPool {
    #[new]
    fn new(max_concurrent_reads: usize) -> PyResult<Self> {
        if max_concurrent_reads == 0 {
            return Err(python_error(anyhow::anyhow!(
                "max_concurrent_reads must be positive"
            )));
        }
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(max_concurrent_reads)
            .thread_name(|index| format!("samudra-zarr-{index}"))
            .build()
            .context("creating the shared Rust Zarr read pool")
            .map_err(python_error)?;
        Ok(Self {
            thread_pool: Arc::new(thread_pool),
            max_concurrent_reads,
        })
    }

    #[getter]
    fn max_concurrent_reads(&self) -> usize {
        self.max_concurrent_reads
    }
}

/// Persistent reader for local, flat OM4 arrays shaped `(time, lat, lon)`.
#[pyclass]
struct FlatOm4Reader {
    path: PathBuf,
    arrays: HashMap<String, Arc<OpenArray>>,
    shape: [u64; 3],
    thread_pool: Arc<ThreadPool>,
}

impl FlatOm4Reader {
    fn open(
        path: PathBuf,
        variables: Vec<String>,
        thread_pool: Arc<ThreadPool>,
    ) -> anyhow::Result<Self> {
        if variables.is_empty() {
            bail!("at least one variable is required");
        }

        let store: ReadableWritableListableStorage = Arc::new(
            FilesystemStore::new(&path)
                .with_context(|| format!("opening local Zarr store {}", path.display()))?,
        );
        let mut arrays = HashMap::with_capacity(variables.len());
        let mut common_shape = None;

        for variable in variables {
            if arrays.contains_key(&variable) {
                continue;
            }
            if variable.is_empty() {
                bail!("variable names must not be empty");
            }
            let array_path = if variable.starts_with('/') {
                variable.clone()
            } else {
                format!("/{variable}")
            };
            let array = Array::open(store.clone(), &array_path)
                .with_context(|| format!("opening variable {variable:?} in {}", path.display()))?;
            if array.data_type() != &DataType::Float32 {
                bail!(
                    "variable {variable:?} in {} has dtype {:?}; flat OM4 Rust loading requires float32",
                    path.display(),
                    array.data_type()
                );
            }
            let shape: [u64; 3] = array.shape().try_into().with_context(|| {
                format!(
                    "variable {variable:?} in {} has shape {:?}; expected (time, lat, lon)",
                    path.display(),
                    array.shape()
                )
            })?;
            if shape.contains(&0) {
                bail!(
                    "variable {variable:?} in {} has a zero-sized extent in shape {shape:?}",
                    path.display()
                );
            }
            validate_flat_om4_dimensions(&array).with_context(|| {
                format!(
                    "validating dimensions for variable {variable:?} in {}",
                    path.display()
                )
            })?;
            if let Some(expected) = common_shape {
                if shape != expected {
                    bail!(
                        "variable {variable:?} in {} has shape {shape:?}; expected {expected:?}",
                        path.display()
                    );
                }
            } else {
                common_shape = Some(shape);
            }
            arrays.insert(variable, Arc::new(array));
        }

        Ok(Self {
            path,
            arrays,
            shape: common_shape.expect("variables was checked as non-empty"),
            thread_pool,
        })
    }

    fn read_into_impl(
        &self,
        indexes: &[i64],
        variables: &[String],
        output: &mut [f32],
    ) -> anyhow::Result<ReadShape> {
        if variables.is_empty() {
            bail!("at least one variable is required for a read");
        }

        let indexes = indexes
            .iter()
            .map(|index| validate_index(*index, self.shape[0]))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let arrays = variables
            .iter()
            .map(|variable| {
                self.arrays.get(variable).cloned().with_context(|| {
                    format!(
                        "variable {variable:?} was not opened by the reader for {}",
                        self.path.display()
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let lat = self.shape[1] as usize;
        let lon = self.shape[2] as usize;
        let plane_len = lat
            .checked_mul(lon)
            .context("flat OM4 spatial shape overflows usize")?;
        let plane_count = indexes
            .len()
            .checked_mul(arrays.len())
            .context("flat OM4 read plane count overflows usize")?;
        let element_count = plane_count
            .checked_mul(plane_len)
            .context("flat OM4 output shape overflows usize")?;
        if output.len() != element_count {
            bail!(
                "flat OM4 output has {} elements; expected {element_count}",
                output.len()
            );
        }
        let reads = indexes
            .iter()
            .flat_map(|index| arrays.iter().map(move |array| (*index, array)))
            .collect::<Vec<_>>();

        self.thread_pool.install(|| {
            output
                .par_chunks_mut(plane_len)
                .zip(reads.par_iter())
                .try_for_each(|(target, (index, array))| -> anyhow::Result<()> {
                    let subset = ArraySubset::new_with_start_shape(
                        vec![*index, 0, 0],
                        vec![1, self.shape[1], self.shape[2]],
                    )
                    .context("constructing a flat OM4 array subset")?;
                    let elements = array
                        .retrieve_array_subset_elements::<f32>(&subset)
                        .with_context(|| {
                            format!(
                                "reading {} at time index {index} from {}",
                                array.path(),
                                self.path.display()
                            )
                        })?;
                    if elements.len() != plane_len {
                        bail!(
                            "read {} elements from {} at time index {index}; expected {plane_len}",
                            elements.len(),
                            array.path()
                        );
                    }
                    target.copy_from_slice(&elements);
                    Ok(())
                })
        })?;

        Ok((indexes.len(), arrays.len(), lat, lon))
    }

    fn read_impl(&self, indexes: &[i64], variables: &[String]) -> anyhow::Result<ReadOutput> {
        let lat = self.shape[1] as usize;
        let lon = self.shape[2] as usize;
        let element_count = indexes
            .len()
            .checked_mul(variables.len())
            .and_then(|count| count.checked_mul(lat))
            .and_then(|count| count.checked_mul(lon))
            .context("flat OM4 output shape overflows usize")?;
        let mut output = vec![0.0_f32; element_count];
        let shape = self.read_into_impl(indexes, variables, &mut output)?;
        Ok((shape, output))
    }
}

#[derive(Clone)]
struct CompactVariable {
    array: Arc<OpenArray>,
    level: Option<u64>,
    time_axis: usize,
    level_axis: Option<usize>,
}

/// Persistent reader translating canonical OM4 channels into compact arrays.
#[pyclass]
struct CompactOm4Reader {
    path: PathBuf,
    variables: HashMap<String, CompactVariable>,
    shape: [u64; 3],
    thread_pool: Arc<ThreadPool>,
}

impl CompactOm4Reader {
    fn open(
        path: PathBuf,
        logical_variables: Vec<String>,
        thread_pool: Arc<ThreadPool>,
    ) -> anyhow::Result<Self> {
        if logical_variables.is_empty() {
            bail!("at least one variable is required");
        }

        let store: ReadableWritableListableStorage = Arc::new(
            FilesystemStore::new(&path)
                .with_context(|| format!("opening local Zarr store {}", path.display()))?,
        );
        let mut physical_arrays: HashMap<String, Arc<OpenArray>> = HashMap::new();
        let mut variables = HashMap::with_capacity(logical_variables.len());
        let mut common_shape = None;

        for logical_name in logical_variables {
            if variables.contains_key(&logical_name) {
                continue;
            }
            let (physical_name, level) = compact_variable_selector(&logical_name)?;
            let array = if let Some(array) = physical_arrays.get(physical_name) {
                array.clone()
            } else {
                let array_path = if physical_name.starts_with('/') {
                    physical_name.to_string()
                } else {
                    format!("/{physical_name}")
                };
                let array = Arc::new(
                    Array::open(store.clone(), &array_path).with_context(|| {
                        format!(
                            "opening compact OM4 variable {physical_name:?} for canonical channel {logical_name:?} in {}",
                            path.display()
                        )
                    })?,
                );
                physical_arrays.insert(physical_name.to_string(), array.clone());
                array
            };

            if array.data_type() != &DataType::Float32 {
                bail!(
                    "variable {physical_name:?} in {} has dtype {:?}; compact OM4 Rust loading requires float32",
                    path.display(),
                    array.data_type()
                );
            }

            let (shape, time_axis, level_axis) = if let Some(level) = level {
                let shape: [u64; 4] = array.shape().try_into().with_context(|| {
                    format!(
                        "variable {physical_name:?} for canonical channel {logical_name:?} in {} has shape {:?}; expected (time, lev, lat, lon)",
                        path.display(),
                        array.shape()
                    )
                })?;
                if shape.contains(&0) {
                    bail!(
                        "variable {physical_name:?} in {} has a zero-sized extent in shape {shape:?}",
                        path.display()
                    );
                }
                let (time_axis, level_axis) = validate_compact_om4_dimensions(&array, true)
                    .with_context(|| {
                        format!(
                            "validating dimensions for variable {physical_name:?} in {}",
                            path.display()
                        )
                    })?;
                let level_axis = level_axis.expect("depth-resolved arrays have a level axis");
                if level >= shape[level_axis] {
                    bail!(
                        "canonical channel {logical_name:?} selects level {level}, but compact OM4 variable {physical_name:?} in {} has {} levels",
                        path.display(),
                        shape[level_axis]
                    );
                }
                (
                    [shape[time_axis], shape[2], shape[3]],
                    time_axis,
                    Some(level_axis),
                )
            } else {
                let shape: [u64; 3] = array.shape().try_into().with_context(|| {
                    format!(
                        "surface variable {physical_name:?} in {} has shape {:?}; expected (time, lat, lon)",
                        path.display(),
                        array.shape()
                    )
                })?;
                if shape.contains(&0) {
                    bail!(
                        "variable {physical_name:?} in {} has a zero-sized extent in shape {shape:?}",
                        path.display()
                    );
                }
                let (time_axis, level_axis) = validate_compact_om4_dimensions(&array, false)
                    .with_context(|| {
                        format!(
                            "validating dimensions for variable {physical_name:?} in {}",
                            path.display()
                        )
                    })?;
                (shape, time_axis, level_axis)
            };

            if let Some(expected) = common_shape {
                if shape != expected {
                    bail!(
                        "variable {physical_name:?} in {} has time/spatial shape {shape:?}; expected {expected:?}",
                        path.display()
                    );
                }
            } else {
                common_shape = Some(shape);
            }
            variables.insert(
                logical_name,
                CompactVariable {
                    array,
                    level,
                    time_axis,
                    level_axis,
                },
            );
        }

        Ok(Self {
            path,
            variables,
            shape: common_shape.expect("variables was checked as non-empty"),
            thread_pool,
        })
    }

    fn read_into_impl(
        &self,
        indexes: &[i64],
        logical_variables: &[String],
        output: &mut [f32],
    ) -> anyhow::Result<ReadShape> {
        if logical_variables.is_empty() {
            bail!("at least one variable is required for a read");
        }
        let indexes = indexes
            .iter()
            .map(|index| validate_index(*index, self.shape[0]))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let variables = logical_variables
            .iter()
            .map(|logical_name| {
                self.variables.get(logical_name).cloned().with_context(|| {
                    format!(
                        "canonical channel {logical_name:?} was not opened by the compact OM4 reader for {}",
                        self.path.display()
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let lat = self.shape[1] as usize;
        let lon = self.shape[2] as usize;
        let plane_len = lat
            .checked_mul(lon)
            .context("compact OM4 spatial shape overflows usize")?;
        let element_count = indexes
            .len()
            .checked_mul(variables.len())
            .and_then(|count| count.checked_mul(plane_len))
            .context("compact OM4 output shape overflows usize")?;
        if output.len() != element_count {
            bail!(
                "compact OM4 output has {} elements; expected {element_count}",
                output.len()
            );
        }
        let reads = indexes
            .iter()
            .flat_map(|index| variables.iter().map(move |variable| (*index, variable)))
            .collect::<Vec<_>>();

        self.thread_pool.install(|| {
            output
                .par_chunks_mut(plane_len)
                .zip(reads.par_iter())
                .try_for_each(|(target, (index, variable))| -> anyhow::Result<()> {
                    let subset = if let Some(level) = variable.level {
                        let mut start = vec![0, 0, 0, 0];
                        start[variable.time_axis] = *index;
                        start[variable.level_axis.expect("depth variable has level axis")] = level;
                        ArraySubset::new_with_start_shape(
                            start,
                            vec![1, 1, self.shape[1], self.shape[2]],
                        )
                    } else {
                        ArraySubset::new_with_start_shape(
                            vec![*index, 0, 0],
                            vec![1, self.shape[1], self.shape[2]],
                        )
                    }
                    .context("constructing a compact OM4 array subset")?;
                    let elements = variable
                        .array
                        .retrieve_array_subset_elements::<f32>(&subset)
                        .with_context(|| {
                            format!(
                                "reading {} at time index {index} from {}",
                                variable.array.path(),
                                self.path.display()
                            )
                        })?;
                    if elements.len() != plane_len {
                        bail!(
                            "read {} elements from {} at time index {index}; expected {plane_len}",
                            elements.len(),
                            variable.array.path()
                        );
                    }
                    target.copy_from_slice(&elements);
                    Ok(())
                })
        })?;

        Ok((indexes.len(), variables.len(), lat, lon))
    }

    fn read_impl(
        &self,
        indexes: &[i64],
        logical_variables: &[String],
    ) -> anyhow::Result<ReadOutput> {
        let lat = self.shape[1] as usize;
        let lon = self.shape[2] as usize;
        let element_count = indexes
            .len()
            .checked_mul(logical_variables.len())
            .and_then(|count| count.checked_mul(lat))
            .and_then(|count| count.checked_mul(lon))
            .context("compact OM4 output shape overflows usize")?;
        let mut output = vec![0.0_f32; element_count];
        let shape = self.read_into_impl(indexes, logical_variables, &mut output)?;
        Ok((shape, output))
    }
}

#[pymethods]
impl CompactOm4Reader {
    #[new]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        logical_variables: Vec<String>,
        read_pool: PyRef<'_, FlatOm4ReadPool>,
    ) -> PyResult<Self> {
        let thread_pool = read_pool.thread_pool.clone();
        py.allow_threads(|| Self::open(path, logical_variables, thread_pool))
            .map_err(python_error)
    }

    /// Read canonical channels in `(time_index, variable, lat, lon)` order.
    fn read<'py>(
        &self,
        py: Python<'py>,
        indexes: Vec<i64>,
        logical_variables: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let (shape, output) = py
            .allow_threads(|| self.read_impl(&indexes, &logical_variables))
            .map_err(python_error)?;
        ndarray::Array::from_shape_vec(shape, output)
            .context("constructing the compact OM4 NumPy output")
            .map_err(python_error)
            .map(|array| array.into_pyarray(py))
    }

    /// Fill a writable C-contiguous float32 NumPy array without an intermediate copy.
    fn read_into(
        &self,
        py: Python<'_>,
        indexes: Vec<i64>,
        logical_variables: Vec<String>,
        mut target: PyReadwriteArray4<'_, f32>,
    ) -> PyResult<()> {
        let expected_shape = [
            indexes.len(),
            logical_variables.len(),
            self.shape[1] as usize,
            self.shape[2] as usize,
        ];
        if target.shape() != expected_shape {
            return Err(python_error(anyhow::anyhow!(
                "compact OM4 output has shape {:?}; expected {:?}",
                target.shape(),
                expected_shape
            )));
        }
        let target = target
            .as_slice_mut()
            .context("compact OM4 output must be C-contiguous")
            .map_err(python_error)?;
        py.allow_threads(|| self.read_into_impl(&indexes, &logical_variables, target))
            .map_err(python_error)?;
        Ok(())
    }

    #[getter]
    fn shape(&self) -> (u64, u64, u64) {
        (self.shape[0], self.shape[1], self.shape[2])
    }
}

#[pymethods]
impl FlatOm4Reader {
    #[new]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        variables: Vec<String>,
        read_pool: PyRef<'_, FlatOm4ReadPool>,
    ) -> PyResult<Self> {
        let thread_pool = read_pool.thread_pool.clone();
        py.allow_threads(|| Self::open(path, variables, thread_pool))
            .map_err(python_error)
    }

    /// Read planes in `(time_index, variable, lat, lon)` order.
    fn read<'py>(
        &self,
        py: Python<'py>,
        indexes: Vec<i64>,
        variables: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let (shape, output) = py
            .allow_threads(|| self.read_impl(&indexes, &variables))
            .map_err(python_error)?;
        ndarray::Array::from_shape_vec(shape, output)
            .context("constructing the flat OM4 NumPy output")
            .map_err(python_error)
            .map(|array| array.into_pyarray(py))
    }

    /// Fill a writable C-contiguous float32 NumPy array without an intermediate copy.
    fn read_into(
        &self,
        py: Python<'_>,
        indexes: Vec<i64>,
        variables: Vec<String>,
        mut target: PyReadwriteArray4<'_, f32>,
    ) -> PyResult<()> {
        let expected_shape = [
            indexes.len(),
            variables.len(),
            self.shape[1] as usize,
            self.shape[2] as usize,
        ];
        if target.shape() != expected_shape {
            return Err(python_error(anyhow::anyhow!(
                "flat OM4 output has shape {:?}; expected {:?}",
                target.shape(),
                expected_shape
            )));
        }
        let target = target
            .as_slice_mut()
            .context("flat OM4 output must be C-contiguous")
            .map_err(python_error)?;
        py.allow_threads(|| self.read_into_impl(&indexes, &variables, target))
            .map_err(python_error)?;
        Ok(())
    }

    #[getter]
    fn shape(&self) -> (u64, u64, u64) {
        (self.shape[0], self.shape[1], self.shape[2])
    }
}

#[pymodule]
fn samudra_rust_loader(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FlatOm4ReadPool>()?;
    module.add_class::<FlatOm4Reader>()?;
    module.add_class::<CompactOm4Reader>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compact_variable_selector, validate_index};

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
    fn parses_compact_depth_channel() {
        assert_eq!(
            compact_variable_selector("thetao_18").unwrap(),
            ("thetao", Some(18))
        );
    }

    #[test]
    fn preserves_compact_surface_channel() {
        assert_eq!(
            compact_variable_selector("hfds_anomalies").unwrap(),
            ("hfds_anomalies", None)
        );
    }
}
