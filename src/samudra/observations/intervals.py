# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar overlap without rebasing a replay trajectory at month boundaries."""

import numpy as np
import pandas as pd


def calendar_weights(starts, ends, first, last):
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


def replay_month(origin, target_month):
    origin, first = pd.Timestamp(origin), pd.Timestamp(target_month)
    if origin != origin.normalize() or origin.day != 1 or first.day != 1:
        raise ValueError("Origins and target months must be month starts")
    last = first + pd.offsets.MonthBegin()
    start = (first - origin).days // 5
    stop = int(np.ceil((last - origin).days / 5))
    if start < 0:
        raise ValueError("Replay target precedes origin")
    return start, stop, (last - origin).days // 5
