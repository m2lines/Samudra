# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Simulate all DDP ranks locally; no model, GPU, network, or process group needed."""

from collections import Counter
from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from samudra.utils.samplers import DistributedEquivalenceGroupBatchSampler


@dataclass(frozen=True)
class SyntheticDataset:
    size: int
    batch_compatibility_key: int

    def __len__(self):
        return self.size


@st.composite
def schedules(draw):
    datasets = [
        SyntheticDataset(size, key)
        for size, key in draw(
            st.lists(
                st.tuples(st.integers(0, 12), st.integers(0, 2)), min_size=1, max_size=5
            )
        )
    ]
    return (
        datasets,
        draw(st.integers(1, 4)),
        draw(st.integers(1, 4)),
        draw(st.booleans()),
        draw(st.booleans()),
        draw(st.integers(0, 65535)),
    )


def ranks(case):
    datasets, batch_size, replicas, drop_last, shuffle, seed = case
    return [
        DistributedEquivalenceGroupBatchSampler(
            datasets,
            batch_size,
            replicas,
            rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        for rank in range(replicas)
    ]


@given(schedules(), st.integers(0, 100))
def test_distributed_schedule_preserves_coverage_and_step_compatibility(case, epoch):
    datasets, batch_size, replicas, drop_last, _, _ = case
    owners = [
        dataset.batch_compatibility_key
        for dataset in datasets
        for _ in range(len(dataset))
    ]
    samplers = ranks(case)
    actual = []
    for sampler in samplers:
        sampler.set_epoch(epoch)
        actual.append(list(sampler))
        assert len(sampler) == len(actual[-1])
    assert len({len(schedule) for schedule in actual}) == 1
    for step in zip(*actual):
        assert len(step) == replicas
        assert all(0 < len(batch) <= batch_size for batch in step)
        assert all(0 <= index < len(owners) for batch in step for index in batch)
        # Different ranks must process compatible fields at every collective step.
        assert len({owners[index] for batch in step for index in batch}) == 1
    indices = [index for schedule in actual for batch in schedule for index in batch]
    if drop_last:
        assert len(indices) == len(set(indices))
        assert all(
            len(batch) == batch_size for schedule in actual for batch in schedule
        )
        retained = Counter(owners[index] for index in indices)
        for key, size in Counter(owners).items():
            assert 0 <= size - retained[key] < batch_size * replicas
    else:
        # Padding may repeat indices, but every real sample must appear.
        assert set(indices) == set(range(len(owners)))


@given(schedules(), st.integers(0, 100), st.integers(0, 100))
def test_epoch_replay_is_independent_of_previous_iteration(case, first, intervening):
    for sampler in ranks(case):
        sampler.set_epoch(first)
        expected = list(sampler)
        sampler.set_epoch(intervening)
        list(sampler)
        sampler.set_epoch(first)
        assert list(sampler) == expected
        assert list(sampler) == expected
