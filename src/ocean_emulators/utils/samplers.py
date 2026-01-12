import itertools
import random
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, Self

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
        shuffle: Whether to shuffle indices within groups and shuffle batches globally
        drop_last: Whether to drop incomplete batches at the end of each group
    """

    def __init__(
        self,
        groups: list[list[int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        super().__init__()
        self.groups = groups
        self.batch_size = batch_size
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
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Self:
        """Create sampler from dataset sizes, treating each as a contiguous group.

        Args:
            dataset_sizes: List of individual dataset sizes. Groups are created based on
                cumulative boundaries, where each dataset forms its own equivalence group.
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group
        """
        cumsum = 0
        groups = []
        for size in dataset_sizes:
            groups.append(list(range(cumsum, cumsum + size)))
            cumsum += size
        return cls(groups, batch_size, shuffle, drop_last)

    @classmethod
    def from_datasets(
        cls,
        datasets: list["TorchTrainDataset"],
        group_key: Callable[["TorchTrainDataset"], Any],
        batch_size: int,
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
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group

        Examples:
                - lambda ds: (ds._input_src.data.sizes['lat'], ds._input_src.data.sizes['lon'])  # group by resolution
                - lambda ds: ds._input_src.data.sizes['lat']  # group by latitude size only
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group

        Returns:
            EquivalenceGroupBatchSampler configured to group by the provided key

        Example:
            >>> # Group datasets by resolution, allowing different strides to be batched together
            >>> sampler = EquivalenceGroupBatchSampler.from_datasets(
            ...     datasets=dataset_list,
            ...     group_key=lambda ds: (ds._input_src.data.sizes['lat'], ds._input_src.data.sizes['lon']),
            ...     batch_size=32,
            ...     shuffle=True,
            ...     drop_last=True,
            ... )
        """
        from collections import defaultdict

        # Group indices by their key
        groups: dict[tuple, list[int]] = defaultdict(list)

        cumsum = 0
        for ds in datasets:
            key = group_key(ds)
            # Make key hashable if it isn't already
            if not isinstance(key, Hashable):
                key = tuple(key) if hasattr(key, "__iter__") else (key,)
            groups[key].extend(range(cumsum, cumsum + len(ds)))
            cumsum += len(ds)

        # Sort by key for deterministic ordering across runs
        sorted_groups = sorted(groups.items(), key=lambda x: str(x[0]))
        group_indices = [indices for _, indices in sorted_groups]

        return cls(group_indices, batch_size, shuffle, drop_last)

    def __iter__(self):
        # Create batch samplers for each group
        batch_sampler = itertools.chain(*self._samplers)

        if not self.shuffle:
            # No global shuffle: return batches in sequential group order
            yield from batch_sampler
        else:
            # Shuffle batches globally to avoid sequential group processing
            # This is regenerated each epoch, giving different orderings
            all_batches = list(batch_sampler)
            random.shuffle(all_batches)
            yield from all_batches

    def __len__(self):
        """Calculate total number of batches across all groups."""
        total_batches = 0
        for sampler in self._samplers:
            total_batches += len(sampler)

        return total_batches
