import random

import pytest

from ocean_emulators.utils.samplers import (
    DistributedEquivalenceGroupBatchSampler,
    EquivalenceGroupBatchSampler,
)


class MockDataset:
    """Simple mock dataset for testing samplers."""

    def __init__(self, size: int, grid: tuple = (100, 100)):
        self._size = size
        self.input_src = self
        self.label_src = self
        self.grid = grid

    def __len__(self):
        return self._size


@pytest.fixture(params=["standard", "distributed"])
def sampler_from_datasets(request):
    """Factory fixture that creates either standard or distributed sampler."""

    def _make_sampler(datasets, group_key, batch_size, shuffle, drop_last):
        if request.param == "standard":
            return EquivalenceGroupBatchSampler.from_datasets(
                datasets=datasets,
                group_key=group_key,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            return DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,
                group_key=group_key,
                batch_size=batch_size,
                num_replicas=1,  # Single worker to match standard behavior
                rank=0,
                shuffle=shuffle,
                drop_last=drop_last,
            )

    return _make_sampler


class TestEquivalenceGroupBatchSampler:
    def test_groups_created_from_dataset_sizes(self):
        """Groups should partition indices based on cumulative dataset boundaries."""
        sampler = EquivalenceGroupBatchSampler.from_dataset_sizes(
            dataset_sizes=[3, 5, 2],
            batch_size=2,
            shuffle=False,
        )

        assert sampler.groups == [
            [0, 1, 2],  # first dataset: indices 0-2
            [3, 4, 5, 6, 7],  # second dataset: indices 3-7
            [8, 9],  # third dataset: indices 8-9
        ]

    def test_len_without_drop_last(self):
        """Length should count all batches, including partial ones."""
        # Group sizes: 3, 5, 2 -> batches: ceil(3/2)=2, ceil(5/2)=3, ceil(2/2)=1 = 6
        sampler = EquivalenceGroupBatchSampler.from_dataset_sizes(
            dataset_sizes=[3, 5, 2],
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
        assert len(sampler) == 6

    def test_len_with_drop_last(self):
        """Length should only count complete batches when drop_last=True."""
        # Group sizes: 3, 5, 2 -> batches: 3//2=1, 5//2=2, 2//2=1 = 4
        sampler = EquivalenceGroupBatchSampler.from_dataset_sizes(
            dataset_sizes=[3, 5, 2],
            batch_size=2,
            shuffle=False,
            drop_last=True,
        )
        assert len(sampler) == 4

    def test_iter_shuffle_mixes_batch_order(self):
        batches_per_seed = []

        for seed in [42, 1337, 9]:
            random.seed(seed)
            sampler = EquivalenceGroupBatchSampler.from_dataset_sizes(
                # group 0: indices [0..9], group 1: indices [10..19]
                dataset_sizes=[10, 10],
                batch_size=2,
                shuffle=True,
                drop_last=False,
            )
            batches = list(sampler)
            batches_per_seed.append(batches)

            # Ensure that no mixing across group boundaries occurs.
            for batch in batches:
                if batch[0] < 10:
                    assert all(idx < 10 for idx in batch), "Batch mixes groups"
                else:
                    assert all(idx >= 10 for idx in batch), "Batch mixes groups"

        assert all(batches_per_seed[0] != batches for batches in batches_per_seed[1:])

    def test_all_indices_covered_exactly_once(self):
        """Each index should appear in exactly one batch (no shuffle, no drop_last)."""
        sampler = EquivalenceGroupBatchSampler.from_dataset_sizes(
            dataset_sizes=[3, 5],
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # Should cover all indices 0-7
        assert set(all_indices) == set(range(8))


class TestSamplersFromDatasets:
    """Tests for the high-level API shared by both sampler types."""

    def test_all_indices_covered(self, sampler_from_datasets):
        """All indices should be covered when iterating."""
        datasets = [MockDataset(10), MockDataset(10)]
        sampler = sampler_from_datasets(
            datasets=datasets,
            group_key=lambda ds: ds.grid,
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )

        all_indices = [idx for batch in sampler for idx in batch]
        assert set(all_indices) == set(range(20))

    def test_len_matches_iteration(self, sampler_from_datasets):
        """len() should match actual number of batches yielded."""
        datasets = [MockDataset(10), MockDataset(10)]
        sampler = sampler_from_datasets(
            datasets=datasets,
            group_key=lambda ds: ds.grid,
            batch_size=3,
            shuffle=False,
            drop_last=False,
        )

        batches = list(sampler)
        assert len(sampler) == len(batches)

    def test_respects_group_boundaries(self, sampler_from_datasets):
        """Batches should not mix indices from different groups."""
        # Two groups with different grids
        datasets = [MockDataset(10, grid=(100, 100)), MockDataset(10, grid=(200, 200))]
        sampler = sampler_from_datasets(
            datasets=datasets,
            group_key=lambda ds: ds.grid,
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )

        for batch in sampler:
            if batch[0] < 10:
                assert all(idx < 10 for idx in batch), "Batch mixes groups"
            else:
                assert all(idx >= 10 for idx in batch), "Batch mixes groups"


class TestDistributedBatchSamplerDistribution:
    """Tests for distributed-specific behavior of DistributedEquivalenceGroupBatchSampler."""

    def test_workers_get_disjoint_batches(self):
        """Each worker should receive a disjoint subset of batches."""
        datasets = [MockDataset(12), MockDataset(12)]
        num_replicas = 4

        batches_by_worker = []
        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            batches_by_worker.append([tuple(b) for b in sampler])

        # Check no overlap between any pair of workers
        for i in range(num_replicas):
            for j in range(i + 1, num_replicas):
                overlap = set(batches_by_worker[i]) & set(batches_by_worker[j])
                assert len(overlap) == 0, f"Workers {i} and {j} share batches"

    def test_all_workers_cover_all_batches(self):
        """All workers together should cover all batches exactly once."""
        datasets = [MockDataset(10), MockDataset(10)]
        num_replicas = 3

        all_batches = []
        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            all_batches.extend(list(sampler))

        # All indices should be covered exactly once
        all_indices = [idx for batch in all_batches for idx in batch]
        assert sorted(all_indices) == list(range(20))

    def test_set_epoch_changes_ordering(self):
        """Different epochs should produce different batch orderings."""
        datasets = [MockDataset(10), MockDataset(10)]

        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,  # type: ignore[arg-type]
            group_key=lambda ds: ds.grid,  # type: ignore[attr-defined]
            batch_size=2,
            num_replicas=2,
            rank=0,
            shuffle=True,
            drop_last=False,
        )

        sampler.set_epoch(0)
        batches_epoch_0 = [tuple(b) for b in sampler]

        sampler.set_epoch(1)
        batches_epoch_1 = [tuple(b) for b in sampler]

        sampler.set_epoch(2)
        batches_epoch_2 = [tuple(b) for b in sampler]

        assert batches_epoch_0 != batches_epoch_1
        assert batches_epoch_1 != batches_epoch_2

    def test_len_equals_batches_for_each_worker(self):
        """len() should match actual iteration count for each worker."""
        datasets = [MockDataset(10), MockDataset(10)]
        num_replicas = 3

        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            assert len(sampler) == len(list(sampler)), f"len() mismatch for rank {rank}"

    @pytest.mark.parametrize("rank", [-1, 3, 10])
    def test_invalid_rank_raises(self, rank):
        """Invalid rank should raise ValueError."""
        datasets = [MockDataset(10)]

        with pytest.raises(ValueError, match="Invalid rank"):
            DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=3,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
