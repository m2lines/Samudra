// SPDX-FileCopyrightText: 2026 Samudra Authors
//
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{bail, Context};
use numpy::{PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use zarrs::{
    array::{Array, DataType},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits},
};

type OpenArray = Array<dyn ReadableWritableListableStorageTraits>;
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

/// Shared bounded Rayon pool for every flat-OM4 reader in one training process.
#[pyclass]
struct FlatOm4ReadPool {
    thread_pool: Arc<ThreadPool>,
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
        })
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
    ) -> anyhow::Result<()> {
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

        Ok(())
    }
}

#[derive(Clone)]
struct CompactVariable {
    array: Arc<OpenArray>,
    level: Option<u64>,
    time_axis: usize,
    level_axis: Option<usize>,
}

struct CompactReadGroup {
    array: Arc<OpenArray>,
    time_axis: usize,
    level_axis: Option<usize>,
    channels: Vec<(usize, Option<u64>)>,
}

impl CompactReadGroup {
    fn retrieve(
        &self,
        index: u64,
        lat: u64,
        lon: u64,
        path: &std::path::Path,
    ) -> anyhow::Result<(u64, Vec<f32>)> {
        let (first_level, subset, expected_planes) = if let Some(level_axis) = self.level_axis {
            let first_level = self
                .channels
                .iter()
                .filter_map(|(_, level)| *level)
                .min()
                .context("compact OM4 depth group has no selected levels")?;
            let last_level = self
                .channels
                .iter()
                .filter_map(|(_, level)| *level)
                .max()
                .context("compact OM4 depth group has no selected levels")?;
            let level_count = last_level - first_level + 1;
            let mut start = vec![0, 0, 0, 0];
            let mut shape = vec![1, 1, lat, lon];
            start[self.time_axis] = index;
            start[level_axis] = first_level;
            shape[level_axis] = level_count;
            (
                first_level,
                ArraySubset::new_with_start_shape(start, shape)
                    .context("constructing a grouped compact OM4 depth subset")?,
                level_count,
            )
        } else {
            (
                0,
                ArraySubset::new_with_start_shape(vec![index, 0, 0], vec![1, lat, lon])
                    .context("constructing a compact OM4 surface subset")?,
                1,
            )
        };
        let elements = self
            .array
            .retrieve_array_subset_elements::<f32>(&subset)
            .with_context(|| {
                format!(
                    "reading {} at time index {index} from {}",
                    self.array.path(),
                    path.display()
                )
            })?;
        let expected_elements = expected_planes
            .checked_mul(lat)
            .and_then(|count| count.checked_mul(lon))
            .context("compact OM4 grouped read shape overflows u64")?;
        let expected_elements = usize::try_from(expected_elements)
            .context("compact OM4 grouped read shape overflows usize")?;
        if elements.len() != expected_elements {
            bail!(
                "read {} elements from {} at time index {index}; expected {expected_elements}",
                elements.len(),
                self.array.path()
            );
        }
        Ok((first_level, elements))
    }
}

/// Persistent reader for explicit compact-OM4 physical array/level selectors.
#[pyclass]
struct CompactOm4Reader {
    path: PathBuf,
    variables: HashMap<(String, Option<u64>), CompactVariable>,
    shape: [u64; 3],
    thread_pool: Arc<ThreadPool>,
}

impl CompactOm4Reader {
    fn open(
        path: PathBuf,
        variable_selectors: Vec<(String, Option<u64>)>,
        thread_pool: Arc<ThreadPool>,
    ) -> anyhow::Result<Self> {
        if variable_selectors.is_empty() {
            bail!("at least one variable is required");
        }

        let store: ReadableWritableListableStorage = Arc::new(
            FilesystemStore::new(&path)
                .with_context(|| format!("opening local Zarr store {}", path.display()))?,
        );
        let mut physical_arrays: HashMap<String, Arc<OpenArray>> = HashMap::new();
        let mut variables = HashMap::with_capacity(variable_selectors.len());
        let mut common_shape = None;

        for (physical_name, level) in variable_selectors {
            let selector = (physical_name.clone(), level);
            if variables.contains_key(&selector) {
                continue;
            }
            if physical_name.is_empty() {
                bail!("variable names must not be empty");
            }
            let array = if let Some(array) = physical_arrays.get(&physical_name) {
                array.clone()
            } else {
                let array_path = if physical_name.starts_with('/') {
                    physical_name.clone()
                } else {
                    format!("/{physical_name}")
                };
                let array =
                    Arc::new(Array::open(store.clone(), &array_path).with_context(|| {
                        format!(
                            "opening compact OM4 variable {physical_name:?} in {}",
                            path.display()
                        )
                    })?);
                physical_arrays.insert(physical_name.clone(), array.clone());
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
                        "variable {physical_name:?} at level {level} in {} has shape {:?}; expected (time, lev, lat, lon)",
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
                        "compact OM4 variable {physical_name:?} selects level {level}, but {} has {} levels",
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
                selector,
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
        variable_selectors: &[(String, Option<u64>)],
        output: &mut [f32],
    ) -> anyhow::Result<()> {
        if variable_selectors.is_empty() {
            bail!("at least one variable is required for a read");
        }
        let indexes = indexes
            .iter()
            .map(|index| validate_index(*index, self.shape[0]))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let variables = variable_selectors
            .iter()
            .map(|selector| {
                self.variables.get(selector).cloned().with_context(|| {
                    format!(
                        "variable selector {selector:?} was not opened by the compact OM4 reader for {}",
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
        // A compact depth array is normally chunked across every level for one time
        // point. Group logical channels backed by the same physical array so that,
        // for example, thetao_0 through thetao_4 decompress that chunk once rather
        // than five times.
        let mut group_indexes: HashMap<String, usize> = HashMap::new();
        let mut groups: Vec<CompactReadGroup> = Vec::new();
        for (channel_index, variable) in variables.iter().enumerate() {
            let array_path = variable.array.path().to_string();
            if let Some(group_index) = group_indexes.get(&array_path) {
                groups[*group_index]
                    .channels
                    .push((channel_index, variable.level));
            } else {
                group_indexes.insert(array_path, groups.len());
                groups.push(CompactReadGroup {
                    array: variable.array.clone(),
                    time_axis: variable.time_axis,
                    level_axis: variable.level_axis,
                    channels: vec![(channel_index, variable.level)],
                });
            }
        }
        let time_slice_len = variables
            .len()
            .checked_mul(plane_len)
            .context("compact OM4 time-slice shape overflows usize")?;

        self.thread_pool.install(|| {
            output
                .par_chunks_mut(time_slice_len)
                .zip(indexes.par_iter())
                .try_for_each(|(target, index)| -> anyhow::Result<()> {
                    // Keep scratch bounded to one physical array per concurrent time
                    // index. Parallelism across time indices is sufficient for normal
                    // batches, while retaining every decompressed depth array at once
                    // can consume hundreds of MiB for quarter-degree stores.
                    for group in &groups {
                        let (first_level, elements) =
                            group.retrieve(*index, self.shape[1], self.shape[2], &self.path)?;
                        for (channel_index, level) in &group.channels {
                            let source_plane =
                                level.map_or(0, |level| level - first_level) as usize;
                            let source_start = source_plane * plane_len;
                            let target_start = channel_index * plane_len;
                            target[target_start..target_start + plane_len]
                                .copy_from_slice(&elements[source_start..source_start + plane_len]);
                        }
                    }
                    Ok(())
                })
        })?;

        Ok(())
    }
}

#[pymethods]
impl CompactOm4Reader {
    #[new]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        variable_selectors: Vec<(String, Option<u64>)>,
        read_pool: PyRef<'_, FlatOm4ReadPool>,
    ) -> PyResult<Self> {
        let thread_pool = read_pool.thread_pool.clone();
        py.allow_threads(|| Self::open(path, variable_selectors, thread_pool))
            .map_err(python_error)
    }

    /// Fill a writable C-contiguous float32 NumPy array without an intermediate copy.
    fn read_into(
        &self,
        py: Python<'_>,
        indexes: Vec<i64>,
        variable_selectors: Vec<(String, Option<u64>)>,
        target: Bound<'_, PyArray4<f32>>,
    ) -> PyResult<()> {
        let mut target = target.try_readwrite().map_err(|error| {
            python_error(anyhow::anyhow!(
                "compact OM4 output could not be borrowed for writing: {error}"
            ))
        })?;
        let expected_shape = [
            indexes.len(),
            variable_selectors.len(),
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
        py.allow_threads(|| self.read_into_impl(&indexes, &variable_selectors, target))
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

    /// Fill a writable C-contiguous float32 NumPy array without an intermediate copy.
    fn read_into(
        &self,
        py: Python<'_>,
        indexes: Vec<i64>,
        variables: Vec<String>,
        target: Bound<'_, PyArray4<f32>>,
    ) -> PyResult<()> {
        let mut target = target.try_readwrite().map_err(|error| {
            python_error(anyhow::anyhow!(
                "flat OM4 output could not be borrowed for writing: {error}"
            ))
        })?;
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
    use super::validate_index;

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
}
