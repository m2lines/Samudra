# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reuse immutable validation frames without changing training frame delivery."""

import numpy as np


class ValidationFrameCache:
    def __init__(self, source, capacity):
        if capacity < 1:
            raise ValueError("Validation cache needs a positive frame capacity")
        self.source = source
        self.capacity = capacity
        self.frames = {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def read(self, origin):
        if origin not in self.source.origins["validation"]:
            raise ValueError("Validation cache cannot serve a training/test frame")
        if origin not in self.frames:
            if len(self.frames) >= self.capacity:
                raise ValueError("Validation frame selection exceeds cache capacity")
            frame = self.source.read(origin)
            for value in frame.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            self.frames[origin] = frame
        return self.frames[origin]
