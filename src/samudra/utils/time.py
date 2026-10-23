# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import datetime

import numpy as np


def time_discontinuities(
    times: np.ndarray, expected_time_step: datetime.timedelta | None = None
) -> np.ndarray:
    """Mark intervals differing from the declared source cadence.

    Subtract native timestamps to preserve CF calendars. Without a declared
    cadence, require uniform intervals rather than guessing which are gaps.
    """
    deltas = np.diff(times)
    if not deltas.size:
        return np.zeros(0, dtype=bool)
    zero = deltas[0] - deltas[0]
    if not np.all(deltas > zero):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    if expected_time_step is None:
        if np.any(deltas != deltas[0]):
            raise ValueError(
                "Source timestamps have irregular intervals, but no expected time "
                "step is declared; specify the source cadence to identify gaps"
            )
        return np.zeros(deltas.size, dtype=bool)
    if expected_time_step <= datetime.timedelta(0):
        raise ValueError("Expected time step must be positive")
    cadence = (
        np.timedelta64(expected_time_step)
        if np.issubdtype(deltas.dtype, np.timedelta64)
        else expected_time_step
    )
    return deltas != cadence
