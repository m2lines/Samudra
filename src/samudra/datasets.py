# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import ClassVar, final

import numpy as np
import torch
import xarray as xr
from jaxtyping import Float
from torch.utils.data import Dataset

from samudra.constants import (
    Boundary,
    BoundaryVarNames,
    LoaderVersion,
    Prognostic,
    PrognosticVarNames,
    RolloutStep,
)
from samudra.utils.ctx import BatchGrid
from samudra.utils.data import BatchPreprocessor, CanonicalSource, LoadStats
from samudra.utils.device import using_gpu
from samudra.utils.logging import elapsed

logger = logging.getLogger(__name__)


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
        source: CanonicalSource,
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
        self.prognostic_var_names = tuple(prognostic_var_names)
        self.boundary_var_names = tuple(boundary_var_names)

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        self.input_resolution = source.resolution
        self._device = torch.device("cpu")
        self._source = source
        self._times = source.time
        self.preprocessor = BatchPreprocessor(
            source,
            self.prognostic_var_names,
            self.boundary_var_names,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=masked_fill_value,
        )

        time_indices = np.arange(source.time.size)
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
                f"Long rollout will use input at time {source.time.values[0]} and produce"
                f" output at {source.time.values[self.hist + 1]}"
            )

        self.wet_label = source.masks.prognostic_with_hist(self.hist)
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet_label = self.wet_label.pin_memory()

        # Inference only currently supports the same output resolution as the input
        # resolution.
        self.ctx = BatchGrid(
            self.wet_label, self.input_resolution, self.input_resolution
        )

    def __len__(self):
        return self.size

    def to(self, device: torch.device) -> "InferenceDataset":
        """Move the dataset's context tensors to the specified device.

        Call this before using the dataset for inference to ensure tensors
        are on the correct device (GPU).
        """
        self.ctx = self.ctx.to(device)
        self._device = device
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

    def get_initial_input(self) -> tuple[Prognostic, Boundary]:
        prog, boundary, _ = self.__getitem__(0)
        return prog, boundary

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

    def get_boundary(self, step: int) -> Boundary:
        """Return boundary at the requested step."""
        x_index = self._get_x_index(step)
        boundary = self._get_boundary(x_index)
        return boundary

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx):
        x_index = self._get_x_index(idx)
        data_in_prog = self._get_prognostic(x_index)
        data_in_boundary = self._get_boundary(x_index)
        label = self._get_label(x_index)
        return (data_in_prog, data_in_boundary, label)

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
        return self._read_and_prepare(
            x_index.values[:, : self.hist + 1],
            channels=self.prognostic_var_names,
            prognostic=True,
        )

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        return self._read_and_prepare(
            x_index.values[:, : self.hist + 1],
            channels=self.boundary_var_names,
            prognostic=False,
        )

    def _get_label(self, x_index):
        return self._read_and_prepare(
            x_index.values[:, self.hist + 1 :],
            channels=self.prognostic_var_names,
            prognostic=True,
        )

    def _read_and_prepare(
        self,
        time_indices: np.ndarray,
        *,
        channels: tuple[str, ...],
        prognostic: bool,
    ) -> torch.Tensor:
        data = torch.from_numpy(self._source.read(time_indices, channels))
        prepare = (
            self.preprocessor.prepare_prognostic
            if prognostic
            else self.preprocessor.prepare_boundary
        )
        return prepare(data, self._device)

    def get_coords_dict(self):
        return self._source.coordinates()


class InferenceDatasets(Dataset):
    def __init__(self, datasets: list[InferenceDataset], lengths: list[int]):
        self.datasets = datasets
        self.lengths = lengths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return (self.datasets[idx], self.lengths[idx])


class HostBatch:
    def __init__(self, dataset_id: "TorchTrainDataset.Id"):
        self.dataset_id: TorchTrainDataset.Id = dataset_id
        self.steps: list[RolloutStep] = []
        self.load_stats: LoadStats | None = None

    def append(
        self,
        input_: torch.Tensor,
        boundary: torch.Tensor,
        label: torch.Tensor,
    ):
        """Add a prognostic input, boundary, and prognostic label as the last step."""
        self.steps.append(RolloutStep(input_, boundary, label))

    def pin_memory(self):
        self.steps = [
            RolloutStep(
                input_.pin_memory(),
                boundary.pin_memory(),
                label.pin_memory(),
            )
            for input_, boundary, label in self.steps
        ]
        return self


class ModelBatch:
    """A single batch of training data.

    A single batch contains multiple steps worth of ``RolloutStep`` entries, each
    of which is a ``(prognostic_input, boundary_input, label)`` triple. The
    prognostic and boundary tensors are carried separately because the
    samudra-multi model encodes them separately (Samudra just concatenates them later).
    """

    def __init__(self, ctx: BatchGrid):
        self.ctx = ctx
        self.steps: list[RolloutStep] = []
        self.load_stats: LoadStats | None = None

    def append(
        self, prognostic_input: Prognostic, boundary_input: Boundary, label: Prognostic
    ) -> None:
        """Add another RolloutStep as a new step."""
        self.steps.append(RolloutStep(prognostic_input, boundary_input, label))

    def get_initial_input(self) -> tuple[Prognostic, Boundary]:
        return self.get_input(0)

    def get_input(self, step: int) -> tuple[Prognostic, Boundary]:
        prog, boundary, _ = self.steps[step]
        return prog, boundary

    def get_label(self, step: int) -> Prognostic:
        return self.steps[step][2]

    def __getitem__(self, step: int) -> RolloutStep:
        """Converts index (step) into (prognostic, boundary, label) triple."""
        return self.steps[step]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


@final
class TorchTrainDataset(Dataset[HostBatch]):
    """
    This class is used for training and validation.

    It creates rolling indices to keep track of histories/past states. But different
    from InferenceDataset, as it creates rolling indices based on stride. By default,
    the sliding window / stride is 1.

    We make use of ModelBatch class to store a single sample.

    For example,
    Hist=0 ; step=0->[0, 1]; step=1->[1, 2]; step=2->[2, 3]; step=3->[3, 4]
    Hist=1 ; step=0->[[0, 1], [2, 3]]; step=1->[[2, 3], [4, 5]];
            step=2->[[4, 5], [6, 7]]; step=3->[[6, 7], [8, 9]]
    Hist=2 ; step=0->[[0, 1, 2], [3, 4, 5]];
            step=1->[[3, 4, 5], [6, 7, 8]];
            step=2->[[6, 7, 8], [9, 10, 11]];
            step=3->[[9, 10, 11], [12, 13, 14]]
    """

    type Id = str

    FLAG = LoaderVersion.OM4_TORCH

    # Shared across all instances within a process. Created lazily on first
    # __getitem__ call so that each DataLoader worker gets its own clean
    # executor.
    _shared_executor: ClassVar[ThreadPoolExecutor | None] = None

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        if cls._shared_executor is None:
            cls._shared_executor = ThreadPoolExecutor(
                max_workers=None, thread_name_prefix="concurrent_compute"
            )
        return cls._shared_executor

    @elapsed
    def __init__(
        self,
        input_source: CanonicalSource,
        label_source: CanonicalSource | None,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        concurrent_compute_: bool = False,
    ):
        super().__init__()
        self.id = f"{self.__class__.__name__}_{str(id(self))}"
        sources = [input_source, label_source] if label_source else [input_source]

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        self._concurrent_compute = concurrent_compute_

        assert np.array_equal(sources[0].time, sources[-1].time), (
            "Input and label sources have different time slices!"
        )
        time_ = input_source.time
        self.sources = sources
        self.prognostic_var_names = tuple(prognostic_var_names)
        self.boundary_var_names = tuple(boundary_var_names)

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

        self.preprocessors = [
            BatchPreprocessor(
                source,
                self.prognostic_var_names,
                self.boundary_var_names,
                normalize_before_mask=normalize_before_mask,
                masked_fill_value=masked_fill_value,
            )
            for source in sources
        ]

        self.ctx = BatchGrid(
            label_mask=self.sources[-1].masks.prognostic_with_hist(self.hist),
            input_resolution_cpu=self.sources[0].resolution,
            output_resolution_cpu=self.sources[-1].resolution,
        )

        self.size: int = (
            time_.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

    def __len__(self) -> int:
        return self.size

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        host_batch = HostBatch(self.id)

        for step in range(self.steps):
            x_index = self._get_x_index(idx, step)
            current_x_index = x_index.isel(time=slice(0, self.hist + 1))
            forecast_x_index = x_index.isel(time=slice(self.hist + 1, None))

            reads = (
                (
                    self.sources[0],
                    current_x_index.values,
                    self.prognostic_var_names,
                ),
                (
                    self.sources[0],
                    current_x_index.values,
                    self.boundary_var_names,
                ),
                (
                    self.sources[-1],
                    forecast_x_index.values,
                    self.prognostic_var_names,
                ),
            )
            if self._concurrent_compute:
                executor = self._get_executor()
                futures = [
                    executor.submit(source.read, indices, channels)
                    for source, indices, channels in reads
                ]
                loaded = [future.result() for future in futures]
            else:
                loaded = [
                    source.read(indices, channels)
                    for source, indices, channels in reads
                ]
            input_, boundary, label = map(torch.from_numpy, loaded)
            host_batch.append(input_, boundary, label)
        host_batch.load_stats = LoadStats(time.perf_counter() - start_time)

        return host_batch

    def to_model_batch(self, host_batch: HostBatch, device: torch.device) -> ModelBatch:
        """Convert HostBatch to ModelBatch, moving tensors to the specified device.

        Args:
            host_batch: CPU data from worker process
            device: Target device (typically GPU) to move tensors to

        Returns:
            ModelBatch with tensors on the target device
        """
        model_batch = ModelBatch(self.ctx.to(device))
        for input_, boundary, label in host_batch.steps:
            prog_input = self.preprocessors[0].prepare_prognostic(input_, device)
            boundary_input = self.preprocessors[0].prepare_boundary(boundary, device)
            label_tensor = self.preprocessors[-1].prepare_prognostic(label, device)
            model_batch.append(prog_input, boundary_input, label_tensor)
        model_batch.load_stats = host_batch.load_stats
        return model_batch

    def _get_x_index(self, idx: int, step: int) -> xr.DataArray:
        assert isinstance(idx, int)
        if idx < 0:
            raise IndexError("Sorry, negative indexing is not supported!")
        if idx >= len(self):
            raise IndexError("Index out of range")

        window_index = idx + step * (self.hist + 1) * self.stride
        return self.rolling_indices.isel(window=window_index, drop=True)


@final
class BatchLoader:
    """Wrapper around a torch DataLoader that handles GPU post-processing.

    This class wraps a DataLoader[HostBatch] and converts the raw data
    to ModelBatch by applying GPU-based normalization and masking. This allows
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
        host_loader: torch.utils.data.DataLoader[HostBatch],
        datasets: list[TorchTrainDataset],
        device: torch.device,
    ):
        self._host_loader = host_loader
        self._datasets = {dataset.id: dataset for dataset in datasets}
        self._device = device

    def __iter__(self):
        """Iterate over the dataloader, converting HostBatch to ModelBatch."""
        for host_batch in self._host_loader:
            dataset = self._datasets[host_batch.dataset_id]
            model_batch = dataset.to_model_batch(host_batch, self._device)
            yield model_batch

    def __len__(self) -> int:
        return len(self._host_loader)

    def __getitem__(self, index: int) -> ModelBatch:
        """Access a single item by index, converting HostBatch to ModelBatch.

        Note: This bypasses the DataLoader's sampling/batching and directly accesses
        the underlying dataset for test purposes.
        """
        # Access the underlying dataset directly
        host_batch = self._host_loader.dataset[index]
        # Apply the collate function to add batch dimension (expects a list)
        collate_fn = self._host_loader.collate_fn
        if collate_fn is not None:
            host_batch = collate_fn([host_batch])
        # Get the dataset that created this raw data
        dataset = self._datasets[host_batch.dataset_id]
        # Convert to ModelBatch
        model_batch = dataset.to_model_batch(host_batch, self._device)
        return model_batch

    @property
    def dataset(self):
        return self._host_loader.dataset

    @property
    def sampler(self):
        return self._host_loader.sampler
