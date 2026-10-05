# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from samudra.experiments.task_schedule import TaskSchedule, sample_indices


@pytest.mark.parametrize("ordering", ["sequential", "mixed"])
def test_exact_exposure_and_restart(ordering):
    schedule = TaskSchedule(6000, 4000, ordering)
    counts = {"om4": 0, "observation": 0}
    stream = []
    for index in range(schedule.total):
        task = schedule.task(index)
        stream.append(task)
        counts[task] += 1
        assert counts == schedule.counts(index + 1)
    assert counts == {"om4": 6000, "observation": 4000}
    resumed = TaskSchedule(6000, 4000, ordering)
    for start in (1, 123, 5000, 9999):
        assert [resumed.task(i) for i in range(start, resumed.total)] == stream[start:]
    if ordering == "mixed":
        assert stream[:1000].count("observation") < stream[-1000:].count("observation")
        assert stream[-1000:].count("om4") > 0


def test_task_samples_are_identical_despite_ordering():
    streams = []
    for ordering in ("sequential", "mixed"):
        schedule = TaskSchedule(60, 40, ordering)
        samples: dict[str, list[int]] = {"om4": [], "observation": []}
        for index in range(schedule.total):
            task = schedule.task(index)
            samples[task].extend(
                sample_indices(13, 1729, schedule.counts(index)[task], 8)
            )
        streams.append(samples)
    assert streams[0] == streams[1]
    for sample in streams[0].values():
        assert sorted(sample[:13]) == list(range(13))
        assert sorted(sample[13:26]) == list(range(13))
    assert sample_indices(13, 1729, 7, 8) == streams[0]["observation"][56:64]


def test_scratch_and_invalid_schedules():
    scratch = TaskSchedule(0, 9, "scratch")
    assert all(scratch.task(i) == "observation" for i in range(9))
    for arguments in (
        (1, 9, "scratch"),
        (1, 9, "mixed"),
        (-1, 1, "mixed"),
        (1, 0, "sequential"),
    ):
        with pytest.raises(ValueError):
            TaskSchedule(*arguments)
    with pytest.raises(ValueError):
        scratch.task(9)
    with pytest.raises(ValueError):
        scratch.counts(-1)
