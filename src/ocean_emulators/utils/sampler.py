import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class ReplicatedDistributedSampler(Sampler[int]):
    """Sampler where all ranks iterate the same index sequence.

    Unlike DistributedSampler which partitions data across ranks,
    this sampler replicates the same samples to all ranks. This is
    necessary for ensemble training where all ranks must process
    identical inputs to generate different ensemble members.

    Supports epoch-based reshuffling via set_epoch.

    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle indices each epoch
        seed: Random seed for shuffling
        drop_last: Whether to drop incomplete batches
    """

    def __init__(
        self, dataset, shuffle: bool = True, seed: int = 0, drop_last: bool = False
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = len(dataset)

        if self.drop_last and self.num_samples == 0:
            raise ValueError("Empty dataset with drop_last=True is invalid.")

    def __iter__(self) -> Iterator[int]:
        g = random.Random(self.seed + self.epoch)
        indices = list(range(self.num_samples))

        if self.shuffle:
            g.shuffle(indices)

        # Yield all indices; all ranks produce the *same* order
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        This ensures that each epoch uses a different random ordering
        when shuffle=True, while maintaining consistency across all ranks.
        """
        self.epoch = epoch
