import itertools
import math
import random
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Self

from torch.utils.data import BatchSampler, Sampler, SubsetRandomSampler

if TYPE_CHECKING:
    from ocean_emulators.datasets import TorchTrainDataset


class _SimpleSubsetSampler(Sampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EquivalenceGroupBatchSampler(Sampler[list[int]]):
    """Groups indices into equivalence classes, batches within groups, and optionally shuffles.

    This sampler partitions dataset indices into groups. It creates batches within each group,
    then chains them together. When shuffle=True, batches are globally shuffled each epoch
    to avoid sequential group processing.

    Args:
        groups: List of index lists, where each inner list contains indices belonging to
            the same equivalence group.
        batch_size: Number of samples per batch
        num_replicas: Coordinate batches across workers to prevent loading asymmetry.
            This should prevent NCCL timeouts. Use 1 to turn worker coordination off.
        shuffle: Whether to shuffle indices within groups and shuffle batches globally
        drop_last: Whether to drop incomplete batches at the end of each group
    """

    def __init__(
        self,
        groups: list[list[int]],
        batch_size: int,
        num_replicas: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        super().__init__()
        self.groups = groups
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Choose sampler based on shuffle setting
        SubsetSampler = SubsetRandomSampler if self.shuffle else _SimpleSubsetSampler

        self._samplers = [
            BatchSampler(
                SubsetSampler(group),
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
        num_replicas: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Self:
        """Create sampler from dataset sizes, treating each as a contiguous group.

        Args:
            dataset_sizes: List of individual dataset sizes. Groups are created based on
                cumulative boundaries, where each dataset forms its own equivalence group.
            batch_size: Number of samples per batch
            num_replicas: Coordinate batches across workers to prevent loading asymmetry.
                This should prevent NCCL timeouts. Use 1 to turn worker coordination off.
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group
        """
        cumsum = 0
        groups = []
        for size in dataset_sizes:
            groups.append(list(range(cumsum, cumsum + size)))
            cumsum += size
        return cls(groups, batch_size, num_replicas, shuffle, drop_last)

    @classmethod
    def from_datasets(
        cls,
        datasets: list["TorchTrainDataset"],
        group_key: Callable[["TorchTrainDataset"], Hashable],
        batch_size: int,
        num_replicas: int,
        shuffle: bool,
        drop_last: bool,
    ) -> Self:
        """Create sampler by grouping datasets using a key function.

        This factory method allows grouping datasets by arbitrary criteria (e.g., resolution,
        regardless of other parameters like stride). Datasets with the same key are batched together.

        Args:
            datasets: List of TorchTrainDataset instances to group
            group_key: Callable that extracts grouping key from a dataset.
            batch_size: Number of samples per batch
            num_replicas: Coordinate batches across workers to prevent loading asymmetry.
                This should prevent NCCL timeouts. Use 1 to turn worker coordination off.
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group

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
        from collections import defaultdict

        # Group indices by their key
        groups: dict[Hashable, list[int]] = defaultdict(list)

        cumsum = 0
        for ds in datasets:
            key = group_key(ds)
            assert isinstance(key, Hashable), "`group_key` must be hashable."
            groups[key].extend(range(cumsum, cumsum + len(ds)))
            cumsum += len(ds)

        # Sort by key for deterministic ordering across runs
        sorted_groups = sorted(groups.items(), key=lambda x: x[0])  # type: ignore
        group_indices = [indices for _, indices in sorted_groups]

        return cls(group_indices, batch_size, num_replicas, shuffle, drop_last)

    def __iter__(self):
        # Batch within each group and align to num_replicas boundaries so that
        # consecutive num_replicas batches always belong to the same group.
        # This prevents different DDP ranks from receiving different-resolution
        # data in the same training step, which causes load-time asymmetry and
        # Gloo/NCCL timeouts.
        per_replica = []
        move_to_end = []
        for sampler in self._samplers:
            pre_replica_per_sample = list(itertools.batched(sampler, self.num_replicas))
            if (
                pre_replica_per_sample
                and len(pre_replica_per_sample[-1]) < self.num_replicas
            ):
                move_to_end.append(pre_replica_per_sample.pop())
            if self.shuffle:
                random.shuffle(pre_replica_per_sample)
            per_replica.append(pre_replica_per_sample)

        whole_replica_batches = list(itertools.chain.from_iterable(per_replica))
        if self.shuffle:
            random.shuffle(whole_replica_batches)

        if self.drop_last:
            all_replica_batches = whole_replica_batches
        else:
            if self.shuffle:
                random.shuffle(move_to_end)
            all_replica_batches = whole_replica_batches + move_to_end
        yield from itertools.chain.from_iterable(all_replica_batches)

    def __len__(self):
        """Calculate total number of batches across all groups."""
        total_batches = 0
        for sampler in self._samplers:
            if self.num_replicas > 1 and self.drop_last:
                total_batches += (len(sampler) // self.num_replicas) * self.num_replicas
            else:
                total_batches += len(sampler)

        return total_batches


class DistributedEquivalenceGroupBatchSampler(Sampler[list[int]]):
    """Distributed version of EquivalenceGroupBatchSampler for multi-GPU training.

    Uses composition to delegate batching logic to EquivalenceGroupBatchSampler,
    handling only the distribution and epoch-based shuffling.

    Ensures uniform batch counts across all ranks to prevent DDP hangs at
    collective sync points. When the total batch count isn't divisible by
    num_replicas:
    - drop_last=True: trims to the largest multiple of num_replicas
    - drop_last=False: pads by duplicating batches from the beginning

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
        datasets: list["TorchTrainDataset"],
        group_key: Callable[["TorchTrainDataset"], Hashable],
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
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
            num_replicas=num_replicas,
            shuffle=False,  # We handle shuffling with seeded RNG
            drop_last=drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling across workers."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Chunk each group's batches by num_replicas so that consecutive
        # num_replicas batches always come from the same resolution group.
        # This prevents different DDP ranks from receiving different-resolution
        # data in the same training step, which causes load-time asymmetry and
        # Gloo/NCCL timeouts.
        chunks: list[tuple] = []
        for sampler in self._inner._samplers:
            group_chunks = list(itertools.batched(sampler, self.num_replicas))
            if group_chunks and len(group_chunks[-1]) < self.num_replicas:
                # Always pad incomplete chunks so every chunk is homogeneous
                # and full-sized. Without padding, an incomplete chunk would
                # break the stride alignment for all subsequent chunks.
                # Dropping here would erase entire groups that have fewer
                # batches than num_replicas; global trimming below handles
                # drop_last instead.
                last = list(group_chunks[-1])
                while len(last) < self.num_replicas:
                    last.append(last[-1])
                group_chunks[-1] = tuple(last)
            if self.shuffle:
                rng.shuffle(group_chunks)
            chunks.extend(group_chunks)

        if self.shuffle:
            rng.shuffle(chunks)

        # Flatten chunks back to a list of batches. Every consecutive
        # num_replicas batches now belong to the same resolution group.
        all_batches = list(itertools.chain.from_iterable(chunks))

        # Ensure uniform batch count across all ranks to prevent DDP hangs.
        total = len(all_batches)
        if self.drop_last:
            num_batches_per_rank = total // self.num_replicas
            all_batches = all_batches[: num_batches_per_rank * self.num_replicas]
        else:
            num_batches_per_rank = (total + self.num_replicas - 1) // self.num_replicas
            padding_size = num_batches_per_rank * self.num_replicas - total
            if padding_size > 0:
                all_batches = all_batches + [
                    all_batches[i % total] for i in range(padding_size)
                ]

        # Each worker takes every num_replicas'th batch starting at rank
        for i in range(self.rank, len(all_batches), self.num_replicas):
            yield all_batches[i]

    def __len__(self):
        """Number of batches for this worker (same for all ranks)."""
        n = self.num_replicas
        total = 0
        for sampler in self._inner._samplers:
            # Per-group chunks are always padded to num_replicas (never
            # dropped), so use ceil here regardless of drop_last.
            total += math.ceil(len(sampler) / n) * n
        if self.drop_last:
            return total // n
        else:
            return math.ceil(total / n)
