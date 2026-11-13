<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Training on your own data

Samudra trains on sequences of gridded fields. A data source provides those
fields, their normalization statistics, and masks identifying valid cells.
The source configuration defines the variables and time ranges used for
training and validation.

There are several places to adapt the data pipeline:

| Task | Where to make the change |
| --- | --- |
| Use another dataset with an existing source's format and variables | Store locations and time ranges in the data YAML |
| Select a different set of variables within an existing source | Variable presets in `src/samudra/constants.py` and preset keys in YAML |
| Support different variable names, depths, or dataset conventions | A source configuration in `src/samudra/config.py` |
| Read from a different storage implementation | The `CanonicalReader` interface in `src/samudra/utils/data.py` |
| Change how training samples are assembled | Training windows and batch preparation in `src/samudra/datasets.py` |

## Variables and time sequences

**Prognostic variables** are the state fields the model receives and predicts.
Training samples contain an input history and future values of these fields
as targets. **Boundary variables** are supplied to the model at each step of a
rollout. They must be available for every input window used during training,
validation, and inference.

All selected fields must share aligned time samples and spatial grids. Variable
names, units, and normalization statistics should describe the quantities stored
in the dataset.

## Configure an existing source

The built-in source configurations are `om4`, `cm4`, and `llc`. Their presets are in
`src/samudra/configs/data/`. To use one, copy a training preset and its included
model/data YAML files, preserving or updating the `!include` paths.

The following data YAML selects an OM4 variable preset and specifies the three
stores and time ranges:

```yaml
sources:
  - type: om4
    data_location: my_data.zarr
    data_means_location: my_means.zarr
    data_stds_location: my_stds.zarr
    prognostic_vars_key: thermo_dynamic_all
    boundary_vars_key: tau_hfds
    train_time:
      start: "2000-01-01"
      end: "2009-12-31"
    val_time:
      start: "2010-01-01"
      end: "2010-12-31"
loading:
  type: cpu
```

Locations resolve relative to `experiment.data_root` in the training
configuration. Set the time ranges to dates in your dataset. Selections include
both endpoints, so use distinct endpoints for training and validation.
The source's time configuration
must match the dataset calendar: `Om4TimeConfig` uses Julian dates, while
`Cm4TimeConfig` uses no-leap dates and preserves CM4 interval-end timestamps.
The `cm4` source shares OM4 layouts and defaults to the `tau_hfds_wfo` boundary
preset.

`loading.type: cpu` selects Python/xarray data loading. The model's compute device
is configured separately. The `gpu` and `rust` loading paths support local OM4
stores with additional storage requirements; see [Rust data loader](rust-data-loader.md).

## Add a variable preset

`build_om4_layout()` and `build_llc_layout()` in `src/samudra/constants.py` define
named sets of prognostic and boundary variables. The YAML fields
`prognostic_vars_key` and `boundary_vars_key` select entries in these dictionaries.

For example, to train on sea surface height using an OM4-compatible dataset, add
this entry to the prognostic-variable dictionary in `build_om4_layout()`:

```python
"surface_height": ["zos"],
```

Then select `prognostic_vars_key: surface_height` in the source YAML. Keep the
existing presets so configurations using them continue to work. Install the
modified checkout using the [development setup](contributing.md).

Each selected variable needs values in the data store, a mean and standard
deviation in the statistics stores, and a matching mask. Supply variable metadata
through the dataset's `long_name` and `units` attributes or the layout's
`default_metadata` mapping.

A variable preset uses the source's existing depth, mask, grid, and calendar
conventions. For example, `build_om4_layout()` defines 19 specific depths and
thicknesses. To represent a different depth grid or dataset convention, add a
source configuration as described below. Also check the diagnostics enabled by
the training configuration: some require particular prognostic fields.

## Add a source configuration

A source configuration translates a dataset's conventions into the common
representation used by the training pipeline. In the code, this representation
is called **canonical data**: named channels with standard dimensions, ordered
variable lists, normalization statistics, and masks.

The pipeline has four main parts:

| Component | Responsibility |
| --- | --- |
| `BaseDataSourceConfig` subclasses | Open datasets, define the layout, translate names and dimensions, and select time ranges |
| `CanonicalSource` | Combine a reader, masks, and a `DataLayout` describing channel order and physical metadata |
| `XarrayCanonicalReader` | Read requested times and channels from xarray datasets |
| `TrainingWindows` and batch preparation | Assemble inputs and targets, normalize, apply masks, and transfer batches to the compute device |

To add an xarray-readable source:

1. Add a `BaseDataSourceConfig` subclass in `src/samudra/config.py`.
   Implement `build_layout()` to return a `DataLayout` with the ordered
   prognostic/boundary names, depths, thicknesses, grid type, and metadata.
2. Implement `canonicalize_datasets(data, means, stds)` to return the three
   datasets in canonical form and their layout. `Om4FamilyDataSourceConfig` implements the shared OM4/CM4
   transformations; `Cm4DataSourceConfig` adds no-leap calendar validation, and
   `LlcDataSourceConfig` illustrates a different source convention. Use a time
   configuration matching the dataset's calendar and validate that training and
   validation ranges are separate.
3. Register the type in `DataSourceType` and the discriminated `DataSourceConfig`
   union. If adding a time configuration, include it in `TimeConfig`.
   Regenerate schemas with `uv run src/samudra/config_schema.py`.
4. Select the source type in YAML with `loading.type: cpu`. The base configuration
   builds a `CanonicalSource` backed by `XarrayCanonicalReader` and passes it to
   the existing training pipeline.

### Canonical dataset format

- Each channel is a named `(time, lat, lon)` variable. Depth levels are separate
  channels, such as `thetao_0` and `thetao_1`. A final underscore followed by
  digits is interpreted as a depth index.
- `lat` and `lon` are one-dimensional model-grid axes. Geographic coordinates
  can also be preserved separately, as in the OM4 adapter's handling of
  curvilinear data. The model's grid type and padding must suit the grid topology.
- Means and standard deviations are scalars for each channel, in the same units
  as the data. Compute them from the training period and check that they are
  finite, with positive standard deviations.
- Masks normally use the names `mask_<depth_index>`, with surface channels using
  `mask_0`. `True` identifies valid cells. The mask extraction helper uses the
  first timestep if a mask has a time dimension.
- Reads return unnormalized float32 arrays in the requested time/channel order.
  Batch preparation applies normalization and masking.

`CanonicalSource.from_datasets(...)` constructs a source from canonical xarray
datasets, a `DataLayout`, and explicit prognostic/boundary names. It extracts
masks and validates channel availability. To provide masks explicitly, use
`CanonicalSource.from_canonical_datasets(...)` with a `Masks` object and adapt
the source construction method accordingly. Both factories take datasets that
have already passed through the source's canonicalization step.

## Add a storage reader

`CanonicalReader` in `src/samudra/utils/data.py` defines the interface for reading
channels and exposing statistics, coordinates, metadata, and time slices.
Implement it to integrate a different storage system. A read uses indices
relative to the current time slice and returns an array with shape
`(*time_indices.shape, channel, lat, lon)`.

When xarray supplies the canonical metadata, use the `reader_factory` argument
of `CanonicalSource.from_datasets` to install the reader. Otherwise, construct a
`CanonicalSource` with the reader, masks, and layout in the source's build method.

The accelerated batch pipeline additionally requires `BulkCanonicalReader`.
`TrainingSourceBackend` in `src/samudra/data_backend.py` handles reader
construction and backend-specific validation. `NativeOm4Reader` and
`build_om4_reader` in `src/samudra/native_reader.py` provide an example of mapping
physical storage variables to canonical channels.

## Change sample construction

`TrainingWindows` in `src/samudra/datasets.py` selects the times and channels for
inputs, boundary fields, and targets. Batch preparation applies normalization,
masks, and tensor shaping. `TrainBatchLoader` defines the interface consumed by
the training loop.

These are the extension points for a different sampling scheme or batch
structure. Keep the model, stepper, and inference dataset consistent with any
changes to the meaning or shape of inputs and targets.

## Validate the integration

Inspect a small source slice and a prepared batch to check channel order,
values, normalization, masks, input/target timestamps, and boundary inputs across
a rollout. Ensure each split contains enough samples for the input history and
forecast horizon. Multiple training sources must have equal `DataLayout` values.

Check that enabled metrics and visualizations apply to the selected fields and
grid. For example, ocean heat content requires the layout's temperature channel
and layer thicknesses. Then run a short training and validation experiment before
scaling up.
