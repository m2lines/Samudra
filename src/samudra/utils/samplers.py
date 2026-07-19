# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random
from collections.abc import Callable, Hashable
from typing import Protocol, Self, TypeVar

from torch import Generator
from torch.utils.data import BatchSampler, Sampler, SubsetRandomSampler

type Batch = list[int]
type DdpStepChunk = tuple[Batch, ...]


class BatchCompatibleDataset(Protocol):
    @property
    def batch_compatibility_key(self) -> Hashable: ...

    def __len__(self) -> int: ...


DatasetT = TypeVar("DatasetT", bound=BatchCompatibleDataset)


class _SimpleSubsetSampler(Sampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EquivalenceGroupBatchSampler(Sampler[Batch]):
    """Groups indices into equivalence classes, batches within groups, and optionally shuffles.

    This sampler partitions dataset indices into groups. It creates batches within each group,
    then chains them together. When shuffle=True, batches are globally shuffled each epoch
    to avoid sequential group processing.

    Args:
        groups: List of index lists, where each inner list contains indices belonging to
            the same equivalence group.
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle indices within groups and shuffle batches globally
        drop_last: Whether to drop incomplete batches at the end of each group
    """

    def __init__(
        self,
        groups: list[list[int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Keep a stable, sequential representation for length calculation and
        # for the distributed sampler's group-wise chunking. Shuffled batch
        # samplers are constructed per epoch with their own generator below.
        self._samplers = [
            BatchSampler(
                _SimpleSubsetSampler(group),
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            for group in self.groups
        ]

    @classmethod
    def from_dataset_sizes(
        cls,
        dataset_sizes: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> Self:
        """Create sampler from dataset sizes, treating each as a contiguous group.

        Args:
            dataset_sizes: List of individual dataset sizes. Groups are created based on
                cumulative boundaries, where each dataset forms its own equivalence group.
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group
            seed: Base seed for deterministic epoch shuffling
        """
        cumsum = 0
        groups = []
        for size in dataset_sizes:
            groups.append(list(range(cumsum, cumsum + size)))
            cumsum += size
        return cls(groups, batch_size, shuffle, drop_last, seed)

    @classmethod
    def from_datasets(
        cls,
        datasets: list[DatasetT],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
        group_key: Callable[[DatasetT], Hashable] | None = None,
    ) -> Self:
        """Create sampler by grouping datasets using a key function.

        This factory method allows grouping datasets by arbitrary criteria (e.g., resolution,
        regardless of other parameters like stride). Datasets with the same key are batched together.

        Args:
            datasets: List of TorchTrainDataset instances to group
            group_key: Optional legacy override for the dataset's public
                ``batch_compatibility_key``.
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group
            seed: Base seed for deterministic epoch shuffling

        Examples:
                - lambda ds: (ds._input_src.data.sizes['lat'], ds._input_src.data.sizes['lon'])  # group by resolution
                - lambda ds: ds._input_src.data.sizes['lat']  # group by latitude size only

        Returns:
            EquivalenceGroupBatchSampler configured to group by the provided key

        Example:
            >>> # Group datasets by resolution, allowing different strides to be batched together
            >>> sampler = EquivalenceGroupBatchSampler.from_datasets(
            ...     datasets=dataset_list,
            ...     group_key=lambda ds: tuple(prog.grid_size for prog in ds.prognostic_srcs),
            ...     batch_size=32,
            ...     shuffle=True,
            ...     drop_last=True,
            ... )
        """
        # Dicts preserve first-seen order, which makes the group schedule depend
        # only on the rank-stable dataset list order. Compatibility keys only
        # define equality; they need not be comparable or identical objects in
        # different processes (for example, the current per-shard identity key).
        groups: dict[Hashable, list[int]] = {}

        cumsum = 0
        for ds in datasets:
            key = group_key(ds) if group_key is not None else ds.batch_compatibility_key
            assert isinstance(key, Hashable), "`group_key` must be hashable."
            groups.setdefault(key, []).extend(range(cumsum, cumsum + len(ds)))
            cumsum += len(ds)

        return cls(list(groups.values()), batch_size, shuffle, drop_last, seed)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle schedule for an epoch."""
        self.epoch = epoch

    def __iter__(self):
        if not self.shuffle:
            all_batches = list(itertools.chain(*self._samplers))
        else:
            # Isolate data-order randomness from the process-global PyTorch and
            # Python RNGs. In particular, DataLoader consumes random numbers to
            # seed workers while the Rust loader does not; using dedicated RNGs
            # keeps the two loader contracts schedule-equivalent.
            generator = Generator().manual_seed(self.seed + self.epoch)
            shuffled_samplers = [
                BatchSampler(
                    SubsetRandomSampler(group, generator=generator),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for group in self.groups
            ]
            all_batches = list(itertools.chain(*shuffled_samplers))
            random.Random(self.seed + self.epoch).shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        """Calculate total number of batches across all groups."""
        total_batches = 0
        for sampler in self._samplers:
            total_batches += len(sampler)

        return total_batches


class DistributedEquivalenceGroupBatchSampler(Sampler[Batch]):
    """Distributed version of EquivalenceGroupBatchSampler for multi-GPU training.

    Uses composition to delegate batching logic to EquivalenceGroupBatchSampler,
    handling only the distribution and epoch-based shuffling.

    Ensures uniform batch counts across all ranks to prevent DDP hangs at
    collective sync points. Each equivalence group is chunked into logical
    DDP steps of num_replicas batches, so ranks process the same group at the
    same step. For an incomplete per-group step:
    - drop_last=True: drops that incomplete group step
    - drop_last=False: pads it by duplicating batches from the same group

    > Note: Compared to the non-distributed sampler: this one won't shuffle
    > _within_ batches, only between batches, when `shuffle=True`.

    Args:
        datasets: List of TorchTrainDataset instances to group
        group_key: Callable that extracts grouping key from a dataset
        batch_size: Number of samples per batch
        num_replicas: Number of distributed workers (world size)
        rank: Index of current worker (0 to num_replicas-1)
        shuffle: Whether to shuffle batches
        drop_last: Whether to drop incomplete batches within each group, and
            whether to trim (vs pad) when distributing batches across ranks
        seed: Random seed for shuffling (default: 0)
    """

    def __init__(
        self,
        datasets: list[DatasetT],
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        group_key: Callable[[DatasetT], Hashable] | None = None,
    ):
        super().__init__()
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, must be in range [0, {num_replicas})"
            )

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Delegate batching logic to inner sampler (without shuffle for determinism)
        self._inner = EquivalenceGroupBatchSampler.from_datasets(
            datasets=datasets,
            group_key=group_key,
            batch_size=batch_size,
            shuffle=False,  # We handle shuffling with seeded RNG
            drop_last=drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling across workers."""
        self.epoch = epoch

    def __iter__(self):
        # Every rank constructs the same shuffled global chunk order because
        # seed, epoch, and input groups are identical across ranks.
        rng = random.Random(self.seed + self.epoch)

        # chunks: list[DdpStepChunk]
        #   A global logical-step schedule. All ranks build this same list, then
        #   each rank selects its own slot from every chunk below.
        chunks: list[DdpStepChunk] = []
        for sampler in self._inner._samplers:
            # sampler: BatchSampler for one equivalence group/resolution.
            #
            # group_chunks: list[DdpStepChunk]
            #   Each element is one candidate DDP step for this group. The last
            #   one may be short before the drop/pad handling below.
            group_chunks: list[DdpStepChunk] = list(
                itertools.batched(sampler, self.num_replicas)
            )
            if group_chunks and len(group_chunks[-1]) < self.num_replicas:
                if self.drop_last:
                    # Drop the short step for this group. This avoids a final
                    # step where only some ranks have same-group work.
                    group_chunks.pop()
                else:
                    # Pad the incomplete chunk with duplicates from the same
                    # group so every chunk is homogeneous and full-sized.
                    #
                    # partial_chunk: list[Batch]
                    #   The short final DDP step for this group.
                    partial_chunk: list[Batch] = list(group_chunks[-1])
                    # Draw padding from all of this group's batches (not just
                    # the tail) so padding doesn't systematically over-weight
                    # the same examples every epoch.
                    #
                    # all_group_batches: list[Batch]
                    #   Padding source restricted to this group, so no padded
                    #   DDP step can mix resolutions.
                    all_group_batches = list(
                        itertools.chain.from_iterable(group_chunks)
                    )
                    if self.shuffle:
                        rng.shuffle(all_group_batches)
                    padding_pool = itertools.cycle(all_group_batches)
                    while len(partial_chunk) < self.num_replicas:
                        partial_chunk.append(next(padding_pool))
                    # Replace the short tuple[Batch, ...] with a full
                    # DdpStepChunk.
                    group_chunks[-1] = tuple(partial_chunk)
            if self.shuffle:
                # Shuffle this group's DDP steps. Do not shuffle the batches
                # within a step, because position in the tuple maps to rank.
                rng.shuffle(group_chunks)
            chunks.extend(group_chunks)

        if self.shuffle:
            # Shuffle DDP steps across groups while keeping each step internally
            # homogeneous.
            rng.shuffle(chunks)

        # Per-group chunking above already ensures len is a multiple of num_replicas,
        # so every consecutive num_replicas batches belong to the same group.
        #
        # all_batches: list[Batch]
        #   Flattened global schedule:
        #     [step0_rank0, step0_rank1, ..., step1_rank0, step1_rank1, ...]
        all_batches: list[Batch] = list(itertools.chain.from_iterable(chunks))
        assert len(all_batches) % self.num_replicas == 0

        # Each worker takes every num_replicas'th batch starting at rank.
        local_batches = [
            list(all_batches[i])
            for i in range(self.rank, len(all_batches), self.num_replicas)
        ]
        yield from local_batches

    def __len__(self):
        """Number of batches for this worker (same for all ranks)."""
        n = self.num_replicas
        total = 0
        for sampler in self._inner._samplers:
            group_len = len(sampler)
            if self.drop_last:
                total += (group_len // n) * n
            else:
                total += math.ceil(group_len / n) * n
        if self.drop_last:
            return total // n
        else:
            return math.ceil(total / n)
