import logging
import time
from collections.abc import Callable
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, final

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float
from torch.utils.data import Dataset
from xarray_einstats.einops import rearrange as xr_rearrange  # noqa: F401

from ocean_emulators.constants import (
    BoundaryVarNames,
    Example,
    GridMask,
    Input,
    LoaderVersion,
    Prognostic,
    PrognosticMask,
    PrognosticVarNames,
)
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import (
    DataSource,
    LoadStats,
    OceanData,
    conditional_rearrange,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.location import _zarr_gpu_decode_context
from ocean_emulators.utils.logging import elapsed

logger = logging.getLogger(__name__)


def _to_torch_float32(array_like: Any) -> torch.Tensor:
    if hasattr(array_like, "compute"):
        array_like = array_like.compute()

    if isinstance(array_like, np.ndarray):
        return torch.from_numpy(array_like.astype(np.float32, copy=False))

    if hasattr(array_like, "__dlpack__"):
        return torch.from_dlpack(array_like).to(dtype=torch.float32)

    if hasattr(array_like, "get"):
        array_like = array_like.get()
        return torch.from_numpy(np.asarray(array_like, dtype=np.float32))

    return torch.as_tensor(array_like, dtype=torch.float32)


def _dataarray_to_torch_float32(array: xr.DataArray) -> torch.Tensor:
    return _to_torch_float32(array.data)


def _array_like_type_name(array_like: Any) -> str:
    return f"{type(array_like).__module__}.{type(array_like).__qualname__}"


def _materialize_dataarray_to_torch_float32(
    build_array: Callable[[], xr.DataArray],
    *,
    use_zarr_gpu_decode: bool,
) -> tuple[torch.Tensor, str]:
    with _zarr_gpu_decode_context(use_zarr_gpu_decode):
        # Keep every xarray transformation that can touch backend data inside
        # the Zarr GPU config scope.
        array = build_array()
        array_like = array.data
    return _to_torch_float32(array_like), _array_like_type_name(array_like)


def _mask_on_tensor_device(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    if mask.device == tensor.device:
        return mask
    return mask.to(tensor.device, non_blocking=tensor.device.type == "cuda")


class InferenceDataset(Dataset):
    """This class is used for inference rollouts.

    It creates rolling indices to keep track of histories/past states.
    For example,
    Hist=0 ; 0->[0, 1]; 1->[1, 2]; 2->[2, 3]; 3->[3, 4]
    Hist=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]];
            2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
    Hist=2 ; 0->[[0, 1, 2], [3, 4, 5]];
            1->[[3, 4, 5], [6, 7, 8]];
            2->[[6, 7, 8], [9, 10, 11]];
            3->[[9, 10, 11], [12, 13, 14]]
    """

    @elapsed
    def __init__(
        self,
        src: DataSource,
        prognostic_var_names,
        boundary_var_names,
        hist,
        normalize_before_mask,
        masked_fill_value,
        long_rollout,
    ):
        super().__init__()
        # NOTE: Keep tensors on CPU during initialization. This allows the dataset
        # to be passed between DataLoader worker processes. Call to(device) before
        # using the dataset for inference.

        self.hist = hist

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        data = src.data
        self.input_res = src.resolution
        self._prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        self._boundary_src = src.filter(boundary_var_names, prefix="boundary")
        self.use_zarr_gpu_decode = src.use_zarr_gpu_decode
        self._logged_gpu_decode_materialization = False
        self._times = data.time
        self.normalize_before_mask = normalize_before_mask
        self.masked_fill_value = masked_fill_value

        time_indices = np.arange(data.time.size)
        indices = xr.DataArray(
            time_indices,
            dims=["time"],
            coords={"time": time_indices},
        )
        total_steps = 2 * self.hist + 1
        rolling_indices = indices.rolling(
            time=len(time_indices) - total_steps, center=False
        ).construct("window_dim")
        rolling_indices = rolling_indices.transpose("window_dim", "time").isel(
            time=slice(len(time_indices) - total_steps - 1, None)
        )  # Remove first few null indices
        self.rolling_indices = rolling_indices.isel(
            window_dim=slice(0, None, self.hist + 1)
        )  # Skip indices based on history
        self.rolling_indices = self.rolling_indices.astype(int)

        if long_rollout:
            logger.info(
                f"Long rollout will use input at time {data.time.values[0]} and produce"
                f" output at {data.time.values[self.hist + 1]}"
            )

        self.wet: PrognosticMask = src.masks.prognostic
        self.wet_surface: GridMask = src.masks.boundary
        self.wet_label = src.masks.prognostic_with_hist(self.hist)
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()
            self.wet_label = self.wet_label.pin_memory()

        self.ctx = GridContext(self.wet_label, self.input_res)

    def __len__(self):
        return self.size

    def to(self, device: torch.device) -> "InferenceDataset":
        """Move the dataset's context tensors to the specified device.

        Call this before using the dataset for inference to ensure tensors
        are on the correct device (GPU).
        """
        self.ctx = self.ctx.to(device)
        self.wet_label = self.wet_label.to(device, non_blocking=True)
        return self

    @property
    def initial_prognostic(self):
        x_index = self._get_x_index(0)
        data_in = self._get_prognostic(x_index)
        return data_in

    def inference_target(self, step: int | slice):
        x_index = self._get_x_index(step)
        label = self._get_label(x_index)
        return label

    def get_initial_input(self):
        data = self.__getitem__(0)[0]
        return data

    def get_target_time(self, start_step: int, num_steps: int):
        x_index = self._get_x_index(start_step)
        batch_index = x_index.values[0]
        steps_predicted = len(batch_index) // 2
        start_target_index = batch_index[steps_predicted]

        return self._times.isel(
            time=slice(
                start_target_index, start_target_index + num_steps * steps_predicted
            )
        )

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        x_index = self._get_x_index(step)
        boundary = self._get_boundary(x_index).to(prognostic.device)
        data = torch.cat((prognostic, boundary), dim=1)
        return data

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx):
        x_index = self._get_x_index(idx)
        data_in = self._get_prognostic(x_index)
        data_in_boundary = self._get_boundary(x_index)
        data_in = torch.cat((data_in, data_in_boundary), dim=1)
        label = self._get_label(x_index)
        return (data_in, label)

    def _get_x_index(self, idx):
        if isinstance(idx, slice):
            if (
                (idx.start is not None and idx.start < 0)
                or (idx.stop is not None and idx.stop < 0)
                or (idx.step is not None and idx.step < 0)
            ):
                raise IndexError("Sorry, negative indexing is not supported!")
            if idx.step is None:
                idx = slice(idx.start, idx.stop, 1)
            if idx.start is None and idx.stop is None:
                idx = slice(0, self.size, idx.step)
            elif idx.start is None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop is None:
                idx = slice(idx.start, self.size, idx.step)
        elif isinstance(idx, int):
            if idx < 0:
                raise IndexError("Sorry, negative indexing is not supported!")
            elif idx >= self.size:
                raise IndexError(f"Index {idx} out of range with size {self.size}")
            idx = slice(idx, idx + 1, 1)

        rolling_idx = self.rolling_indices.isel(window_dim=idx)
        x_index = xr.Variable(["window_dim", "time"], rolling_idx)

        return x_index

    def _get_prognostic(self, x_index):
        data_in_src = self._prognostic_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(None, self.hist + 1))
        )
        if self.normalize_before_mask:
            data_in_ds = data_in_src.normalize()
        else:
            data_in_ds = data_in_src.data

        data_in, backend_type = _materialize_dataarray_to_torch_float32(
            lambda: (
                conditional_rearrange(
                    data_in_ds,
                    "window_dim time (variable lev)=var lat lon",
                    concat_dim="var",
                ).rename({"var": "variable"})
                if "lev" in data_in_ds.dims
                else data_in_ds.to_array().transpose(
                    "window_dim", "time", "variable", "lat", "lon"
                )
            ),
            use_zarr_gpu_decode=self.use_zarr_gpu_decode,
        )
        self._maybe_log_gpu_decode_materialization(
            tensor_name="prognostic",
            backend_type=backend_type,
            tensor=data_in,
        )
        wet = _mask_on_tensor_device(self.wet, data_in)
        data_in = torch.where(wet, data_in, self.masked_fill_value)
        if not self.normalize_before_mask:
            data_in = self._prognostic_src.normalize_with(data_in, variable_axis=2)
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        return data_in

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        data_in_boundary_src = self._boundary_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(None, self.hist + 1))
        )
        if self.normalize_before_mask:
            data_in_boundary_ds = data_in_boundary_src.normalize()
        else:
            data_in_boundary_ds = data_in_boundary_src.data
        data_in_boundary, backend_type = _materialize_dataarray_to_torch_float32(
            lambda: data_in_boundary_ds.to_array().transpose(
                "window_dim", "time", "variable", "lat", "lon"
            ),
            use_zarr_gpu_decode=self.use_zarr_gpu_decode,
        )
        self._maybe_log_gpu_decode_materialization(
            tensor_name="boundary",
            backend_type=backend_type,
            tensor=data_in_boundary,
        )
        wet_surface = _mask_on_tensor_device(self.wet_surface, data_in_boundary)
        data_in_boundary = torch.where(
            wet_surface,
            data_in_boundary,
            self.masked_fill_value,
        )
        if not self.normalize_before_mask:
            data_in_boundary = self._boundary_src.normalize_with(
                data_in_boundary, variable_axis=2
            )
        data_in_boundary = rearrange(
            data_in_boundary,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        return data_in_boundary

    def _get_label(self, x_index):
        label_src = self._prognostic_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        )
        if self.normalize_before_mask:
            label_ds = label_src.normalize()
        else:
            label_ds = label_src.data
        label, backend_type = _materialize_dataarray_to_torch_float32(
            lambda: (
                conditional_rearrange(
                    label_ds,
                    "window_dim time (variable lev)=var lat lon",
                    concat_dim="var",
                ).rename({"var": "variable"})
                if "lev" in label_ds.dims
                else label_ds.to_array().transpose(
                    "window_dim", "time", "variable", "lat", "lon"
                )
            ),
            use_zarr_gpu_decode=self.use_zarr_gpu_decode,
        )
        self._maybe_log_gpu_decode_materialization(
            tensor_name="label",
            backend_type=backend_type,
            tensor=label,
        )
        wet = _mask_on_tensor_device(self.wet, label)
        label = torch.where(wet, label, self.masked_fill_value)
        if not self.normalize_before_mask:
            label = self._prognostic_src.normalize_with(label, variable_axis=2)
        label = rearrange(
            label,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        return label

    def _maybe_log_gpu_decode_materialization(
        self,
        *,
        tensor_name: str,
        backend_type: str,
        tensor: torch.Tensor,
    ) -> None:
        if not self.use_zarr_gpu_decode or self._logged_gpu_decode_materialization:
            return

        logger.info(
            "InferenceDataset GPU zarr materialization: "
            "source=%s tensor=%s backend=%s tensor_device=%s tensor_is_cuda=%s",
            self._prognostic_src.name,
            tensor_name,
            backend_type,
            tensor.device,
            tensor.is_cuda,
        )
        self._logged_gpu_decode_materialization = True

    def get_coords_dict(self):
        return {
            co: self._prognostic_src.data[co] for co in self._prognostic_src.data.coords
        }


class InferenceDatasets(Dataset):
    def __init__(self, datasets: list[InferenceDataset], lengths: list[int]):
        self.datasets = datasets
        self.lengths = lengths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return (self.datasets[idx], self.lengths[idx])


class RawTrainData:
    def __init__(self, dataset_id: "TorchTrainDataset.Id"):
        self.dataset_id: TorchTrainDataset.Id = dataset_id
        self.raw_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.load_stats: LoadStats | None = None

    def insert(
        self,
        input_: torch.Tensor,
        boundary: torch.Tensor,
        label: torch.Tensor,
    ):
        """Add a prognostic input, boundary, and prognostic label as the last step."""
        self.raw_data.append((input_, boundary, label))

    def to(self, device: torch.device):
        self.raw_data = [
            (
                input_.to(device, non_blocking=True),
                boundary.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )
            for input_, boundary, label in self.raw_data
        ]

    def pin_memory(self):
        self.raw_data = [
            (
                input_.pin_memory(),
                boundary.pin_memory(),
                label.pin_memory(),
            )
            for input_, boundary, label in self.raw_data
        ]
        return self


class TrainData:
    """A single batch of training data.

    A single batch contains multiple steps worth of `Example`s (i.e., input/output pairs). These steps are used during
    autoregressive rollout in the training and inference process.

    Constraint: The `Input` tensor is a combination of (flattened) prognostic variables (at all depth levels) and
    boundary forcings. The top `num_prognostic_channels` number of channels must be prognostic variables whereas the
    remaining bottom channels are boundary forcings.
    """

    def __init__(self, num_prognostic_channels: int, ctx: GridContext):
        self.num_prognostic_channels = num_prognostic_channels
        self.ctx = ctx
        self.example_by_step: list[Example] = []
        self.load_stats: LoadStats | None = None

    def append(self, input_: Input, label: Prognostic):
        """Add another Example as a new step."""
        self.example_by_step.append((input_, label))

    def get_initial_input(self) -> Input:
        return self.get_input(0)

    def get_input(self, step: int) -> Input:
        return self[step][0]

    def get_label(self, step: int) -> Prognostic:
        return self[step][1]

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        input_ = self.get_input(step)
        merged = input_.clone()
        merged[:, : self.num_prognostic_channels] = prognostic
        return merged

    def values(self):
        return self.example_by_step

    def __getitem__(self, step: int) -> Example:
        """Converts index (step) into (data, label) tuple."""
        return self.example_by_step[step]

    def __len__(self) -> int:
        return len(self.example_by_step)

    def __iter__(self):
        return iter(range(len(self)))

    def to(self, device: torch.device) -> None:
        for step in self:
            self.example_by_step[step] = (
                self[step][0].to(device, non_blocking=True),
                self[step][1].to(device, non_blocking=True),
            )

    def pin_memory(self):
        for step in self:
            self.example_by_step[step] = (
                self[step][0].pin_memory(),
                self[step][1].pin_memory(),
            )
        return self


@final
class TorchTrainDataset(Dataset[RawTrainData]):
    """
    This class is used for training and validation.

    It creates rolling indices to keep track of histories/past states. But different
    from InferenceDataset, as it creates rolling indices based on stride. By default,
    the sliding window / stride is 1.

    We make use of TrainData class to store a single sample.

    For example,
    Hist=0 ; TD: step=0->[0, 1]; step=1->[1, 2]; step=2->[2, 3]; step=3->[3, 4]
    Hist=1 ; TD: step=0->[[0, 1], [2, 3]]; step=1->[[2, 3], [4, 5]];
            step=2->[[4, 5], [6, 7]]; step=3->[[6, 7], [8, 9]]
    Hist=2 ; TD: step=0->[[0, 1, 2], [3, 4, 5]];
            step=1->[[3, 4, 5], [6, 7, 8]];
            step=2->[[6, 7, 8], [9, 10, 11]];
            step=3->[[9, 10, 11], [12, 13, 14]]
    """

    type Id = str

    FLAG = LoaderVersion.OM4_TORCH

    @elapsed
    def __init__(
        self,
        src: DataSource,
        dst: DataSource | None,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        temporal_stride: int = 1,
        executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__()
        self.id = f"{self.__class__.__name__}_{str(id(self))}"
        # If the src and dst DataSource are the same, we can do a lot less work.
        srcs = [src, dst] if dst else [src]

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        if temporal_stride < 1:
            raise ValueError("temporal_stride must be >= 1")
        self.temporal_stride: int = temporal_stride
        self.normalize_before_mask: bool = normalize_before_mask
        self.masked_fill_value: float = masked_fill_value
        self._executor = executor
        self.use_zarr_gpu_decode = any(src.use_zarr_gpu_decode for src in srcs)
        self._logged_gpu_decode_materialization = False

        self.num_prognostic_channels: int = (hist + 1) * len(prognostic_var_names)
        assert np.array_equal(srcs[0].data.time, srcs[-1].data.time), (
            "src and dst DataSource have different time slices!"
        )
        time_ = src.data.time
        self.prognostic_srcs = [
            src.filter(prognostic_var_names, prefix="prog") for src in srcs
        ]
        self.boundary_src = src.filter(boundary_var_names, prefix="boundary")

        # This class will be used only for training and validation
        total_steps: int = 2 * self.hist + 2

        # Calculate the number of windows
        num_windows = time_.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["window"])

        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])

        # Construct rolling indices
        self.rolling_indices: Float[xr.DataArray, "window time"] = (
            indices_da + stride * window_dim
        )

        # NB(alxmrs): Keep masks on CPU - will be moved to GPU in to_train_data()
        self.wet_prognostic: list[PrognosticMask] = [
            src.masks.prognostic for src in srcs
        ]
        self.wet_surface: GridMask = src.masks.boundary

        self.ctx = GridContext(
            self.prognostic_srcs[-1].masks.prognostic_with_hist(self.hist),
            self.prognostic_srcs[0].resolution,
        )

        base_size = (
            time_.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )
        self.size: int = max(
            0,
            (base_size + self.temporal_stride - 1) // self.temporal_stride,
        )

    def __len__(self) -> int:
        return self.size

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        TD = RawTrainData(self.id)

        for step in range(self.steps):
            x_index = self._get_x_index(idx, step)
            current_x_index = x_index.isel(time=slice(0, self.hist + 1))
            forecast_x_index = x_index.isel(time=slice(self.hist + 1, None))

            # Only materialize the time ranges we actually use to reduce memory.
            input_selected = self.prognostic_srcs[0].data.isel(time=current_x_index)
            boundary_selected = self.boundary_src.data.isel(time=current_x_index)
            label_selected = self.prognostic_srcs[-1].data.isel(
                time=forecast_x_index
            )  # forecasted data
            prognostic_selected = [input_selected, label_selected]

            if self._executor is not None:
                datasets = prognostic_selected + [boundary_selected]
                concurrent_compute(
                    *datasets,
                    executor=self._executor,
                )

            if "lev" in prognostic_selected[0].dims:
                prognostics = [
                    _materialize_dataarray_to_torch_float32(
                        lambda selected=selected: conditional_rearrange(
                            selected,
                            "time (variable lev)=var lat lon",
                            concat_dim="var",
                        ).rename({"var": "variable"}),
                        use_zarr_gpu_decode=self.use_zarr_gpu_decode,
                    )
                    for selected in prognostic_selected
                ]
            else:
                prognostics = [
                    _materialize_dataarray_to_torch_float32(
                        lambda selected=selected: selected.to_array().transpose(
                            "time", "variable", "lat", "lon"
                        ),
                        use_zarr_gpu_decode=self.use_zarr_gpu_decode,
                    )
                    for selected in prognostic_selected
                ]
            boundary, boundary_backend = _materialize_dataarray_to_torch_float32(
                lambda: boundary_selected.to_array().transpose(
                    "time", "variable", "lat", "lon"
                ),
                use_zarr_gpu_decode=self.use_zarr_gpu_decode,
            )
            input_, input_backend = prognostics[0]
            label, label_backend = prognostics[-1]
            self._maybe_log_gpu_decode_materialization(
                input_backend=input_backend,
                boundary_backend=boundary_backend,
                label_backend=label_backend,
                input_tensor=input_,
                boundary_tensor=boundary,
                label_tensor=label,
            )
            TD.insert(input_, boundary, label)
        TD.load_stats = LoadStats(time.perf_counter() - start_time)

        return TD

    def to_train_data(
        self, raw_train_data: RawTrainData, device: torch.device
    ) -> TrainData:
        """Convert RawTrainData to TrainData, moving tensors to the specified device.

        Args:
            raw_train_data: CPU data from worker process
            device: Target device (typically GPU) to move tensors to

        Returns:
            TrainData with tensors on the target device
        """
        train_data = TrainData(self.num_prognostic_channels, self.ctx.to(device))
        for input_, boundary, label in raw_train_data.raw_data:
            input_, label = self._to_example(
                OceanData.from_data_source(
                    input_,
                    self.wet_prognostic[0],
                    self.prognostic_srcs[0],
                ).to(device=device, non_blocking=True),
                OceanData.from_data_source(
                    boundary,
                    self.wet_surface,
                    self.boundary_src,
                ).to(device=device, non_blocking=True),
                OceanData.from_data_source(
                    label, self.wet_prognostic[-1], self.prognostic_srcs[-1]
                ).to(device=device, non_blocking=True),
            )
            train_data.append(input_, label)
        train_data.load_stats = raw_train_data.load_stats
        return train_data

    def _to_example(
        self, input_: OceanData, boundary: OceanData, label: OceanData
    ) -> tuple[Input, Prognostic]:
        # Input/boundary only include current steps; label only includes forecasted steps.
        total_input = self._prep_tensor_steps(input_, boundary)
        label_tensor = self._prep_tensor_steps(label)
        return total_input, label_tensor

    def _prep_tensor_steps(
        self,
        prognostic: OceanData,
        boundary: OceanData | None = None,
    ) -> Input:
        """Prepare tensor steps by normalizing, masking and flattening dimensions."""
        prognostic_steps = prognostic.normalize_and_mask(
            self.normalize_before_mask, self.masked_fill_value
        )

        # Flatten time and variable dimensions
        def flatten_dims(tensor: torch.Tensor) -> torch.Tensor:
            return rearrange(
                tensor, "batch time variable lat lon -> batch (time variable) lat lon"
            )

        prognostic_steps = flatten_dims(prognostic_steps)
        if boundary is not None:
            boundary_steps = boundary.normalize_and_mask(
                self.normalize_before_mask, self.masked_fill_value
            )
            boundary_steps = flatten_dims(boundary_steps)
            return torch.cat((prognostic_steps, boundary_steps), dim=1)

        return prognostic_steps

    def _get_x_index(self, idx: int, step: int) -> xr.DataArray:
        assert isinstance(idx, int)
        if idx < 0:
            raise IndexError("Sorry, negative indexing is not supported!")
        if idx >= len(self):
            raise IndexError("Index out of range")

        window_index = idx * self.temporal_stride + step * (self.hist + 1) * self.stride
        return self.rolling_indices.isel(window=window_index, drop=True)

    def _maybe_log_gpu_decode_materialization(
        self,
        *,
        input_backend: str,
        boundary_backend: str,
        label_backend: str,
        input_tensor: torch.Tensor,
        boundary_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
    ) -> None:
        if not self.use_zarr_gpu_decode or self._logged_gpu_decode_materialization:
            return

        logger.info(
            "TorchTrainDataset GPU zarr materialization: "
            "dataset=%s input_backend=%s boundary_backend=%s label_backend=%s "
            "input_device=%s boundary_device=%s label_device=%s "
            "input_is_cuda=%s boundary_is_cuda=%s label_is_cuda=%s",
            self.id,
            input_backend,
            boundary_backend,
            label_backend,
            input_tensor.device,
            boundary_tensor.device,
            label_tensor.device,
            input_tensor.is_cuda,
            boundary_tensor.is_cuda,
            label_tensor.is_cuda,
        )
        self._logged_gpu_decode_materialization = True


def concurrent_compute(
    *datasets: xr.Dataset,
    executor: ThreadPoolExecutor,
) -> None:
    def load_variable_data(var: xr.Variable) -> None:
        var.load()

    futures = []
    for ds in datasets:
        for var in ds.variables.values():
            futures.append(executor.submit(load_variable_data, var))

    wait(futures)


@final
class TrainDataLoader:
    """Wrapper around a torch DataLoader that handles GPU post-processing.

    This class wraps a DataLoader[RawTrainData] and converts the raw data
    to TrainData by applying GPU-based normalization and masking. This allows
    the data loading process to handle I/O while the main process handles
    GPU operations.

    Since the data samples flow from one process to the other, we want to tie
    them back to the dataset they came from which knows how to do that second
    half once they're in the main process which has GPU access set up. To do that,
    each data sample (which could come from a different dataset) has a dataset ID
    -- `datasets` maps from those IDs to the original datasets.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader[RawTrainData],
        datasets: list[TorchTrainDataset],
        device: torch.device,
    ):
        self._dataloader = dataloader
        self._datasets = {dataset.id: dataset for dataset in datasets}
        self._device = device

    def __iter__(self):
        """Iterate over the dataloader, converting RawTrainData to TrainData."""
        for raw_train_data in self._dataloader:
            dataset = self._datasets[raw_train_data.dataset_id]
            train_data = dataset.to_train_data(raw_train_data, self._device)
            yield train_data

    def __len__(self) -> int:
        return len(self._dataloader)

    def __getitem__(self, index: int) -> TrainData:
        """Access a single item by index, converting RawTrainData to TrainData.

        Note: This bypasses the DataLoader's sampling/batching and directly accesses
        the underlying dataset for test purposes.
        """
        # Access the underlying dataset directly
        raw_train_data = self._dataloader.dataset[index]
        # Apply the collate function to add batch dimension (expects a list)
        collate_fn = self._dataloader.collate_fn
        if collate_fn is not None:
            raw_train_data = collate_fn([raw_train_data])
        # Get the dataset that created this raw data
        dataset = self._datasets[raw_train_data.dataset_id]
        # Convert to TrainData
        train_data = dataset.to_train_data(raw_train_data, self._device)
        return train_data

    @property
    def dataset(self):
        return self._dataloader.dataset

    @property
    def sampler(self):
        return self._dataloader.sampler
