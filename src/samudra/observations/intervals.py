# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar-month overlap weights on a forecast's original five-day intervals."""

from collections.abc import Sequence
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from samudra.observations.constants import STEP_DAYS

type CalendarDate = datetime | cftime.datetime


def calendar_weights(
    starts: Sequence[CalendarDate] | NDArray,
    ends: Sequence[CalendarDate] | NDArray,
    first: CalendarDate,
    last: CalendarDate,
) -> NDArray[np.float32]:
    """Return one overlap fraction per interval, covering the complete month."""
    overlap = np.asarray(
        [
            max(0.0, (min(end, last) - max(start, first)).total_seconds())
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=np.float64,
    )
    duration = (last - first).total_seconds()
    if not np.isclose(overlap.sum(), duration, rtol=0, atol=1e-6):
        raise ValueError("Intervals do not cover the complete calendar month")
    return (overlap / duration).astype(np.float32)


def replay_month(
    origin: str | datetime, target_month: str | datetime
) -> tuple[int, int, int]:
    """Return first lead, exclusive final lead and next-month lead, in model steps."""
    origin, first = pd.Timestamp(origin), pd.Timestamp(target_month)
    if origin != origin.normalize() or origin.day != 1 or first.day != 1:
        raise ValueError("Origins and target months must be month starts")
    last = first + pd.offsets.MonthBegin()
    start = (first - origin).days // STEP_DAYS
    stop = int(np.ceil((last - origin).days / STEP_DAYS))
    if start < 0:
        raise ValueError("Replay target precedes origin")
    return start, stop, (last - origin).days // STEP_DAYS
