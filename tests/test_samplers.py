# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import random

import pytest

from samudra.constants import GridSize
from samudra.utils.samplers import (
    DistributedEquivalenceGroupBatchSampler,
    EquivalenceGroupBatchSampler,
)


class MockDataset:
    """Simple mock dataset for testing samplers."""

    def __init__(self, size: int, grid_size: GridSize = (100, 100)):
        self._size = size
        self.input_src = self
        self.label_src = self
        self.grid_size = grid_size

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
            group_key=lambda ds: ds.grid_size,
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
            group_key=lambda ds: ds.grid_size,
            batch_size=3,
            shuffle=False,
            drop_last=False,
        )

        batches = list(sampler)
        assert len(sampler) == len(batches)

    def test_respects_group_boundaries(self, sampler_from_datasets):
        """Batches should not mix indices from different groups."""
        # Two groups with different grids
        datasets = [
            MockDataset(10, grid_size=(100, 100)),
            MockDataset(10, grid_size=(200, 200)),
        ]
        sampler = sampler_from_datasets(
            datasets=datasets,
            group_key=lambda ds: ds.grid_size,
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
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
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

    def test_all_workers_cover_all_indices_with_padding(self):
        """All workers together should cover all indices (with possible duplicates from padding)."""
        datasets = [MockDataset(10), MockDataset(10)]
        num_replicas = 3  # 10 batches / 3 = 3.33, so we pad to 12 batches

        all_batches = []
        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            all_batches.extend(list(sampler))

        # All indices should be covered (duplicates allowed due to padding)
        all_indices = [idx for batch in all_batches for idx in batch]
        assert set(all_indices) == set(range(20))

    def test_all_workers_cover_all_batches_exactly_once_with_drop_last(self):
        """With drop_last=True and divisible batch count, all batches covered exactly once."""
        datasets = [MockDataset(12), MockDataset(12)]  # 12 batches total
        num_replicas = 4  # 12 batches / 4 = 3, evenly divisible

        all_batches = []
        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            all_batches.extend(list(sampler))

        # All indices covered exactly once when evenly divisible with drop_last
        all_indices = [idx for batch in all_batches for idx in batch]
        assert sorted(all_indices) == list(range(24))

    def test_all_workers_get_same_batch_count(self):
        """All workers must get the same number of batches to prevent DDP hangs."""
        datasets = [MockDataset(10), MockDataset(10)]  # 10 batches total

        for num_replicas in [2, 3, 4, 7]:  # Test various divisibility scenarios
            for drop_last in [True, False]:
                batch_counts = []
                for rank in range(num_replicas):
                    sampler = DistributedEquivalenceGroupBatchSampler(
                        datasets=datasets,  # type: ignore[arg-type]
                        group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                        batch_size=2,
                        num_replicas=num_replicas,
                        rank=rank,
                        shuffle=False,
                        drop_last=drop_last,
                    )
                    batch_counts.append(len(list(sampler)))

                # Critical DDP safety check: all workers must have identical batch counts
                assert len(set(batch_counts)) == 1, (
                    f"Workers have different batch counts with num_replicas={num_replicas}, "
                    f"drop_last={drop_last}: {batch_counts}"
                )

    def test_drop_last_trims_batches(self):
        """With drop_last=True, batches should be trimmed (not padded) for even distribution."""
        datasets = [MockDataset(10), MockDataset(10)]  # 10 batches total
        num_replicas = 3  # 10 // 3 = 3 batches per worker

        all_batches = []
        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            all_batches.extend(list(sampler))

        # With 10 batches and 3 replicas, trimming gives 9 batches (3 per worker)
        # So we should have 9 batches * 2 samples = 18 indices (not all 20)
        all_indices = [idx for batch in all_batches for idx in batch]
        assert len(all_indices) == 18

    def test_set_epoch_changes_ordering(self):
        """Different epochs should produce different batch orderings."""
        datasets = [MockDataset(10), MockDataset(10)]

        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,  # type: ignore[arg-type]
            group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
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
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            assert len(sampler) == len(list(sampler)), f"len() mismatch for rank {rank}"

    def test_fewer_batches_than_replicas_pads_correctly(self):
        """When total batches < num_replicas, padding must wrap multiple times."""
        # 2 samples with batch_size=2 gives 1 batch total
        datasets = [MockDataset(2)]
        num_replicas = 4

        batch_counts = []
        all_batches = []
        for rank in range(num_replicas):
            sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            batches = list(sampler)
            batch_counts.append(len(batches))
            all_batches.extend(batches)

        # Critical: all workers must get exactly 1 batch each
        assert batch_counts == [1, 1, 1, 1], f"Expected [1,1,1,1], got {batch_counts}"
        # All 4 batches should be the same (the single batch, duplicated)
        assert len(all_batches) == 4

    @pytest.mark.parametrize("rank", [-1, 3, 10])
    def test_invalid_rank_raises(self, rank):
        """Invalid rank should raise ValueError."""
        datasets = [MockDataset(10)]

        with pytest.raises(ValueError, match="Invalid rank"):
            DistributedEquivalenceGroupBatchSampler(
                datasets=datasets,  # type: ignore[arg-type]
                group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
                batch_size=2,
                num_replicas=3,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )


def test_group_batch_sampler__distributed__replica_chunks_are_homogeneous():
    """All DDP ranks at the same step must process batches from the same group."""
    ds_size = 20
    n_workers = 8
    datasets = [
        MockDataset(ds_size, grid_size=(10, 20)),
        MockDataset(ds_size, grid_size=(5, 10)),
    ]

    # Collect each rank's batches
    rank_batches = []
    for rank in range(n_workers):
        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,  # type: ignore[arg-type]
            group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
            batch_size=2,
            num_replicas=n_workers,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        rank_batches.append(list(sampler))

    # All ranks must have the same number of batches
    counts = [len(rb) for rb in rank_batches]
    assert len(set(counts)) == 1, f"Ranks have different batch counts: {counts}"

    # At each step, all ranks should process the same group
    batches_per_rank = counts[0]
    for step in range(batches_per_rank):
        step_indices = list(
            itertools.chain.from_iterable(
                rank_batches[r][step] for r in range(n_workers)
            )
        )
        assert all(b >= ds_size for b in step_indices) or all(
            b < ds_size for b in step_indices
        ), f"Step {step}: ranks process mixed groups. Indices: {step_indices}"


def test_group_batch_sampler__n_workers_no_drop_last__all_steps_homogeneous():
    """With per-group padding, every step should be homogeneous across ranks."""
    ds_size = 20
    n_workers = 8
    datasets = [
        MockDataset(ds_size, grid_size=(10, 20)),
        MockDataset(ds_size, grid_size=(5, 10)),
    ]

    # Collect each rank's batches
    rank_batches = []
    for rank in range(n_workers):
        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,  # type: ignore[arg-type]
            group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
            batch_size=2,
            num_replicas=n_workers,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        rank_batches.append(list(sampler))

    # All ranks must have the same number of batches
    counts = [len(rb) for rb in rank_batches]
    assert len(set(counts)) == 1, f"Ranks have different batch counts: {counts}"

    batches_per_rank = counts[0]
    for step in range(batches_per_rank):
        step_indices = list(
            itertools.chain.from_iterable(
                rank_batches[r][step] for r in range(n_workers)
            )
        )
        assert all(b >= ds_size for b in step_indices) or all(
            b < ds_size for b in step_indices
        ), f"Step {step}: ranks process mixed groups. Indices: {step_indices}"


def test_group_batch_sampler__distributed__small_group_is_padded_when_not_dropping():
    """A group with fewer batches than num_replicas is padded (not dropped) with drop_last=False."""
    # Group 0: 10 batches (1 full chunk of 8 + 2 remainder -> padded to 8)
    # Group 1: 2 batches (0 full chunks -> padded to 8 entirely from duplicates)
    n_workers = 8
    datasets = [
        MockDataset(20, grid_size=(10, 20)),  # 10 batches
        MockDataset(4, grid_size=(5, 10)),  # 2 batches (smaller than n_workers)
    ]

    rank_batches = []
    for rank in range(n_workers):
        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,  # type: ignore[arg-type]
            group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
            batch_size=2,
            num_replicas=n_workers,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        rank_batches.append(list(sampler))

    counts = [len(rb) for rb in rank_batches]
    assert len(set(counts)) == 1, f"Ranks have different batch counts: {counts}"

    # Every step must be homogeneous across ranks (small group is padded, not mixed).
    batches_per_rank = counts[0]
    for step in range(batches_per_rank):
        step_indices = list(
            itertools.chain.from_iterable(
                rank_batches[r][step] for r in range(n_workers)
            )
        )
        assert all(idx < 20 for idx in step_indices) or all(
            idx >= 20 for idx in step_indices
        ), f"Step {step} mixes groups: {step_indices}"

    # The small group's samples must appear at least once across all ranks
    # (otherwise it was silently dropped).
    all_indices = {idx for rb in rank_batches for batch in rb for idx in batch}
    assert any(idx >= 20 for idx in all_indices), (
        "Small group was dropped instead of padded"
    )


def test_group_batch_sampler__distributed__padding_cycles_group_batches():
    """Padding should not duplicate one batch repeatedly when a group has alternatives."""
    n_workers = 8
    datasets = [MockDataset(20)]  # 10 batches: one full chunk plus a 2-batch tail

    rank_batches = []
    for rank in range(n_workers):
        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,  # type: ignore[arg-type]
            group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
            batch_size=2,
            num_replicas=n_workers,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        rank_batches.append([tuple(batch) for batch in sampler])

    second_step = [rank_batches[rank][1] for rank in range(n_workers)]
    assert second_step == [
        (16, 17),
        (18, 19),
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
    ]


def test_group_batch_sampler__distributed__shuffle_false_is_deterministic_across_epochs():
    """With shuffle=False, batch order (incl. padding) must not change across epochs."""
    # Choose a size that triggers padding so we exercise the padding path.
    n_workers = 3
    datasets = [MockDataset(10), MockDataset(10)]  # 10 batches total, not divisible

    sampler = DistributedEquivalenceGroupBatchSampler(
        datasets=datasets,  # type: ignore[arg-type]
        group_key=lambda ds: ds.grid_size,  # type: ignore[attr-defined]
        batch_size=2,
        num_replicas=n_workers,
        rank=0,
        shuffle=False,
        drop_last=False,
    )

    sampler.set_epoch(0)
    batches_epoch_0 = [tuple(b) for b in sampler]
    sampler.set_epoch(5)
    batches_epoch_5 = [tuple(b) for b in sampler]

    assert batches_epoch_0 == batches_epoch_5, (
        "shuffle=False sampler should be deterministic across epochs"
    )


def test_distributed_sampler__all_ranks_same_resolution_per_step():
    """All DDP ranks must get batches from the same resolution group at each step."""
    num_replicas = 8
    datasets = [
        MockDataset(13, grid_size=(180, 360)),
        MockDataset(11, grid_size=(360, 720)),
        MockDataset(9, grid_size=(720, 1440)),
    ]
    group_key = lambda ds: ds.grid_size  # noqa: E731
    group_boundary = 13
    second_boundary = 13 + 11

    def index_to_group(idx: int) -> int:
        if idx < group_boundary:
            return 0
        elif idx < second_boundary:
            return 1
        else:
            return 2

    for drop_last in [True, False]:
        for shuffle in [True, False]:
            rank_batches: list[list[list[int]]] = []
            for rank in range(num_replicas):
                sampler = DistributedEquivalenceGroupBatchSampler(
                    datasets=datasets,  # type: ignore[arg-type]
                    group_key=group_key,
                    batch_size=1,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
                rank_batches.append(list(sampler))

            lengths = [len(rb) for rb in rank_batches]
            assert len(set(lengths)) == 1, f"Unequal step counts: {lengths}"

            for step in range(lengths[0]):
                groups_at_step = set()
                for rank in range(num_replicas):
                    batch = rank_batches[rank][step]
                    for idx in batch:
                        groups_at_step.add(index_to_group(idx))
                assert len(groups_at_step) == 1, (
                    f"Mixed resolution groups at step {step} "
                    f"(shuffle={shuffle}, drop_last={drop_last}): "
                    f"groups={groups_at_step}"
                )
