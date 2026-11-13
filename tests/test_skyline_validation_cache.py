# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from samudra.skyline.validation_cache import ValidationFrameCache


def test_validation_cache_is_bounded_immutable_and_separate_from_training():
    calls = []

    def read(origin):
        calls.append(origin)
        return {"origin": origin, "input": np.ones((2, 3)), "target": np.zeros(3)}

    source = SimpleNamespace(
        origins={"validation": [3, 4, 5]}, read=read, metadata="same"
    )
    cached = ValidationFrameCache(source, 2)
    first = cached.read(3)
    assert cached.read(3) is first and calls == [3]
    assert cached.metadata == source.metadata
    with pytest.raises(ValueError):
        first["input"][0, 0] = 99
    cached.read(4)
    with pytest.raises(ValueError, match="capacity"):
        cached.read(5)
    with pytest.raises(ValueError, match="training/test"):
        cached.read(0)
    frame = source.read(0)
    assert frame["input"].flags.writeable
    assert calls == [3, 4, 0]
