# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import pytest

from samudra.viz.plotting import box_plot, linear_piecewise_scale


def test_ocean_depth_scaling():
    fig, ax = plt.subplots()
    try:
        ax.set_ylim(5000, 0)
        assert linear_piecewise_scale(1000, 5, ax=ax) is ax
        assert ax.get_ylim() == (5000, 0)
        # Preserve the upstream 2.0.1 transform used in ocean profiles.
        transform = ax.yaxis.get_transform()
        np.testing.assert_array_equal(
            transform.transform(np.array([0.0, 500.0, 1000.0, 3000.0, 5000.0])),
            [-5000.0, -2000.0, 1000.0, 3000.0, 5000.0],
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("box", "count"),
    [
        ([118, 260, -5, 5], 1),
        ([260, 118, -5, 5], 2),
        ([118, 260, 5, -5], 2),
        ([260, 118, 5, -5], 4),
    ],
)
def test_geographic_box(box, count):
    fig, ax = plt.subplots()
    try:
        ax.set_xlim(70, 320)
        ax.set_ylim(-25, 25)
        box_plot(box, ax=ax, color="orange", label="Full Profile")
        assert len(ax.lines) == count
        assert all(line.get_color() == "orange" for line in ax.lines)
        if count == 1:
            np.testing.assert_array_equal(
                ax.lines[0].get_xdata(), [118, 118, 260, 260, 118]
            )
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), [-5, 5, 5, -5, -5])
    finally:
        plt.close(fig)
