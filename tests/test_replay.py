# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from samudra.replay import (
    ReplayBatchRequest,
    ReplayBatchSlot,
    ReplayBuffer,
    ReplayCursor,
    ReplayEntry,
    ReplaySeedSlot,
    replay_sidecar_path,
)


def make_cursor(
    *,
    source_index: int = 0,
    lead_step: int = 0,
) -> ReplayCursor:
    return ReplayCursor(
        dataset_index=0,
        source_index=source_index,
        lead_step=lead_step,
        stride=1,
        temporal_stride=1,
    )


def make_buffer(buffer_size: int = 4) -> ReplayBuffer:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    return ReplayBuffer(
        buffer_size=buffer_size,
        storage_dtype=torch.float32,
        generator=generator,
        pin_memory=False,
    )


def test_replay_cursor_advance_increments_lead_step():
    cursor = make_cursor(source_index=3, lead_step=2)

    advanced = cursor.advance()

    assert advanced == ReplayCursor(
        dataset_index=0,
        source_index=3,
        lead_step=3,
        stride=1,
        temporal_stride=1,
    )
    assert cursor.lead_step == 2


def test_replay_batch_request_reserved_indices():
    request = ReplayBatchRequest(
        request_id=7,
        train_slots=(ReplayBatchSlot(replay_index=1, cursor=make_cursor()),),
        seed_slots=(
            ReplaySeedSlot(
                replay_index=3,
                cursor=make_cursor(source_index=4),
                reason="warm_start",
            ),
        ),
    )

    assert request.reserved_indices == {1, 3}


def test_replay_buffer_append_replace_and_dtype_conversion():
    buffer = make_buffer(buffer_size=2)
    cursor = make_cursor()

    buffer.append(
        ReplayEntry(
            state=torch.ones(2, 2, dtype=torch.float64),
            cursor=cursor,
        )
    )
    buffer.replace(
        0,
        ReplayEntry(
            state=torch.zeros(2, 2, dtype=torch.float64),
            cursor=cursor.advance(),
        ),
    )

    assert len(buffer) == 1
    assert buffer.entries[0].state.dtype == torch.float32
    assert buffer.entries[0].state.device.type == "cpu"
    assert buffer.entries[0].state.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert buffer.entries[0].cursor.lead_step == 1


def test_replay_buffer_rejects_invalid_size_and_overflow():
    with pytest.raises(ValueError, match="buffer_size"):
        make_buffer(buffer_size=0)

    buffer = make_buffer(buffer_size=1)
    buffer.append(ReplayEntry(state=torch.zeros(1), cursor=make_cursor()))

    with pytest.raises(ValueError, match="full"):
        buffer.append(ReplayEntry(state=torch.zeros(1), cursor=make_cursor()))


def test_replay_buffer_sampling_excludes_reserved_indices():
    buffer = make_buffer(buffer_size=4)
    for index in range(4):
        buffer.append(
            ReplayEntry(
                state=torch.zeros(1, 1, 1),
                cursor=make_cursor(source_index=index, lead_step=0),
            )
        )

    sampled = buffer.sample_indices(
        batch_size=8,
        max_lead_steps=1,
        exclude_reserved={1, 2, 3},
    )
    refreshed = buffer.random_indices(count=8, exclude_reserved={0, 1, 2})

    assert set(sampled) == {0}
    assert set(refreshed) == {3}


def test_replay_buffer_sampling_filters_by_max_lead_steps():
    buffer = make_buffer(buffer_size=3)
    for index, lead_step in enumerate([0, 2, 4]):
        buffer.append(
            ReplayEntry(
                state=torch.zeros(1),
                cursor=make_cursor(source_index=index, lead_step=lead_step),
            )
        )

    assert set(buffer.sample_indices(batch_size=5, max_lead_steps=3)) <= {0, 1}

    with pytest.raises(RuntimeError, match="max_lead_steps"):
        buffer.sample_indices(batch_size=1, max_lead_steps=0)


def test_replay_buffer_state_dict_round_trips_entries_and_generator():
    source = make_buffer(buffer_size=3)
    source.append(
        ReplayEntry(
            state=torch.ones(2, dtype=torch.float64),
            cursor=make_cursor(source_index=2, lead_step=1),
        )
    )

    state = source.state_dict(world_size=4, rank=2)
    restored = make_buffer(buffer_size=3)
    restored.load_state_dict(state)

    assert state["storage_dtype"] == "float32"
    assert state["world_size"] == 4
    assert state["rank"] == 2
    assert len(restored) == 1
    assert torch.equal(restored.entries[0].state, torch.ones(2))
    assert restored.entries[0].cursor == make_cursor(source_index=2, lead_step=1)
    assert torch.equal(restored.generator.get_state(), source.generator.get_state())


def test_replay_buffer_load_truncates_to_current_size():
    source = make_buffer(buffer_size=3)
    for index in range(3):
        source.append(
            ReplayEntry(
                state=torch.full((1,), float(index)),
                cursor=make_cursor(source_index=index),
            )
        )

    restored = make_buffer(buffer_size=2)
    restored.load_state_dict(source.state_dict(world_size=1, rank=0))

    assert len(restored) == 2
    assert [entry.cursor.source_index for entry in restored.entries] == [0, 1]


def test_replay_sidecar_path_includes_rank():
    assert replay_sidecar_path(Path("ckpt.pt"), rank=3) == Path(
        "ckpt.replay_rank3.pt"
    )
