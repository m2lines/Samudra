# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.evaluate_monthly_checkpoints import assigned_steps


def test_four_rank_assignment_covers_every_checkpoint_once_including_cooldown():
    shards = [assigned_steps(rank, 4) for rank in range(4)]
    flattened = [step for shard in shards for step in shard]
    assert sorted(flattened) == list(range(500, 5001, 500))
    assert len(flattened) == len(set(flattened))
    assert max(map(len, shards)) == 3
    assert sorted(assigned_steps(0, 1)) == sorted(flattened)
