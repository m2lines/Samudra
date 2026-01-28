from __future__ import annotations

import queue
import threading
import time
from bisect import bisect_right
from collections.abc import Iterable
import torch
from torch.utils.data import ConcatDataset, Sampler

from ocean_emulators.datasets import RawTrainData, TorchTrainDataset, TrainData
from ocean_emulators.utils.data import LoadStats
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.distributed import get_rank
from ocean_emulators.utils.train import collate_raw_train_data


def _lazy_import_dali():
    try:
        from nvidia.dali import fn  # type: ignore[import-untyped]
        from nvidia.dali.pipeline import Pipeline  # type: ignore[import-untyped]
        from nvidia.dali.plugin.pytorch import (  # type: ignore[import-untyped]
            DALIGenericIterator,
            LastBatchPolicy,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "NVIDIA DALI is required for loader_version=om4-dali. "
            "Install with `uv add --optional cuda nvidia-dali-cuda130`."
        ) from exc
    return fn, Pipeline, DALIGenericIterator, LastBatchPolicy


def _resolve_dataset(
    dataset: TorchTrainDataset | ConcatDataset, idx: int
) -> tuple[TorchTrainDataset, int]:
    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect_right(dataset.cumulative_sizes, idx)
        if dataset_index == 0:
            sample_index = idx
        else:
            sample_index = idx - dataset.cumulative_sizes[dataset_index - 1]
        resolved = dataset.datasets[dataset_index]
        assert isinstance(resolved, TorchTrainDataset)
        return resolved, sample_index
    return dataset, idx


class _DaliBatchSource:
    def __init__(
        self,
        dataset: TorchTrainDataset | ConcatDataset,
        *,
        batch_size: int,
        drop_last: bool,
        load_stats_queue: queue.SimpleQueue[tuple[TorchTrainDataset.Id, float]],
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._load_stats_queue = load_stats_queue
        self._indices: list[int] = []
        self._offset = 0
        self._lock = threading.Lock()

    def reset(self, indices: Iterable[int]) -> None:
        with self._lock:
            self._indices = list(indices)
            self._offset = 0

    def __call__(self) -> list[torch.Tensor]:
        with self._lock:
            if self._offset >= len(self._indices):
                raise StopIteration
            batch_indices = self._indices[
                self._offset : self._offset + self._batch_size
            ]
            if len(batch_indices) < self._batch_size and self._drop_last:
                self._offset = len(self._indices)
                raise StopIteration
            self._offset += len(batch_indices)

        start_time = time.perf_counter()
        outputs, dataset_id = self._load_batch(batch_indices)
        self._load_stats_queue.put(
            (dataset_id, time.perf_counter() - start_time)
        )
        return outputs

    def _load_batch(
        self, batch_indices: list[int]
    ) -> tuple[list[torch.Tensor], TorchTrainDataset.Id]:
        dataset: TorchTrainDataset | None = None
        per_step_prognostic: list[list[torch.Tensor]] | None = None
        per_step_boundary: list[list[torch.Tensor]] | None = None

        for idx in batch_indices:
            resolved, sample_index = _resolve_dataset(self._dataset, idx)
            if dataset is None:
                dataset = resolved
                per_step_prognostic = [[] for _ in range(dataset.steps)]
                per_step_boundary = [[] for _ in range(dataset.steps)]
            elif dataset.id != resolved.id:
                raise ValueError(
                    "DALI loader does not support heterogeneous batches. "
                    f"Got {dataset.id} and {resolved.id} in the same batch."
                )

            assert per_step_prognostic is not None
            assert per_step_boundary is not None

            for step in range(dataset.steps):
                x_index = dataset._get_x_index(sample_index, step)
                prognostic_all, boundary = dataset._load_step(x_index)
                per_step_prognostic[step].append(prognostic_all)
                per_step_boundary[step].append(boundary)

        assert dataset is not None
        assert per_step_prognostic is not None
        assert per_step_boundary is not None

        outputs: list[torch.Tensor] = []
        for step in range(dataset.steps):
            outputs.append(torch.stack(per_step_prognostic[step], dim=0))
            outputs.append(torch.stack(per_step_boundary[step], dim=0))

        return outputs, dataset.id


class DaliTrainDataLoader:
    def __init__(
        self,
        *,
        dataset: TorchTrainDataset | ConcatDataset,
        datasets: list[TorchTrainDataset],
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool,
        device: torch.device | None = None,
        prefetch_queue_depth: int = 2,
        num_threads: int = 1,
        exec_pipelined: bool = True,
        exec_async: bool = True,
    ):
        self._dataset = dataset
        self._datasets = {ds.id: ds for ds in datasets}
        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last

        device = device or get_device()
        if device.type != "cuda":
            raise ValueError("DALI loader requires CUDA device.")
        self._device = device

        steps = {ds.steps for ds in datasets}
        if len(steps) != 1:
            raise ValueError("All datasets must use the same number of steps.")
        self._steps = steps.pop()

        self._output_map = [
            f"step{step}_{kind}"
            for step in range(self._steps)
            for kind in ("prognostic", "boundary")
        ]

        self._load_stats_queue: queue.SimpleQueue[
            tuple[TorchTrainDataset.Id, float]
        ] = queue.SimpleQueue()
        self._source = _DaliBatchSource(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            load_stats_queue=self._load_stats_queue,
        )

        (
            self._fn,
            self._Pipeline,
            self._DALIGenericIterator,
            self._LastBatchPolicy,
        ) = _lazy_import_dali()

        self._pipe = self._build_pipeline(
            prefetch_queue_depth=prefetch_queue_depth,
            num_threads=num_threads,
            exec_pipelined=exec_pipelined,
            exec_async=exec_async,
        )

    def _build_pipeline(
        self,
        *,
        prefetch_queue_depth: int,
        num_threads: int,
        exec_pipelined: bool,
        exec_async: bool,
    ):
        device_id = (
            self._device.index
            if self._device.index is not None
            else torch.cuda.current_device()
        )
        seed = (torch.initial_seed() + get_rank()) % (2**32)
        pipe = self._Pipeline(
            batch_size=self._batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
            exec_pipelined=exec_pipelined,
            exec_async=exec_async,
        )
        with pipe:
            outputs = self._fn.external_source(
                source=self._source,
                num_outputs=2 * self._steps,
                device="gpu",
                batch=True,
            )
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            pipe.set_outputs(*outputs)
        pipe.build()
        return pipe

    def _num_samples(self) -> int:
        total = len(self._sampler)
        if self._drop_last:
            return (total // self._batch_size) * self._batch_size
        return total

    def __len__(self) -> int:
        total = len(self._sampler)
        if self._drop_last:
            return total // self._batch_size
        return (total + self._batch_size - 1) // self._batch_size

    def _clear_queue(self) -> None:
        while True:
            try:
                self._load_stats_queue.get_nowait()
            except queue.Empty:
                break

    def __iter__(self):
        self._clear_queue()
        self._source.reset(self._sampler)
        self._pipe.reset()
        size = self._num_samples()
        iterator = self._DALIGenericIterator(
            [self._pipe],
            output_map=self._output_map,
            size=size,
            auto_reset=False,
            last_batch_policy=(
                self._LastBatchPolicy.DROP
                if self._drop_last
                else self._LastBatchPolicy.PARTIAL
            ),
        )
        for batch in iterator:
            outputs = batch[0]
            dataset_id, load_time = self._load_stats_queue.get()
            dataset = self._datasets[dataset_id]
            raw = RawTrainData(dataset_id)
            for step in range(self._steps):
                raw.insert(
                    outputs[f"step{step}_prognostic"],
                    outputs[f"step{step}_boundary"],
                )
            raw.load_stats = LoadStats(load_time)
            yield dataset.to_train_data(raw)

    def __getitem__(self, index: int) -> TrainData:
        raw_train_data = self._dataset[index]
        raw_train_data = collate_raw_train_data([raw_train_data])
        dataset = self._datasets[raw_train_data.dataset_id]
        return dataset.to_train_data(raw_train_data)

    @property
    def dataset(self):
        return self._dataset

    @property
    def sampler(self):
        return self._sampler
