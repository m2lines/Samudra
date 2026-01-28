import random

from ocean_emulators.utils.samplers import EquivalenceGroupBatchSampler


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
