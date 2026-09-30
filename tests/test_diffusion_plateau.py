# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from samudra.experiments.diffusion_plateau import assess


def test_plateau_uses_cumulative_window_improvement_and_incumbent_best():
    parent = [(2000, 1.0), (2059, 1.1)]
    assert assess(parent, [(2500, 1.02), (3000, 0.995), (3500, 1.01)])[
        "provisional_plateau"
    ]
    assert not assess(parent, [(2500, 0.994), (3000, 0.988), (3500, 0.982)])[
        "provisional_plateau"
    ]


def test_missing_or_final_off_schedule_check_does_not_create_plateau():
    parent = [(2026, 1.0)]
    assert not assess(parent, [(2500, 1.0), (3500, 1.0), (4000, 1.0)])[
        "provisional_plateau"
    ]
    assert not assess(parent, [(2500, 1.0), (3000, 1.0), (3020, 1.0)])[
        "provisional_plateau"
    ]
