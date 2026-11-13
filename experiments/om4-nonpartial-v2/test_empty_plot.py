# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Entirely dry depths must not crash online validation image rendering."""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from samudra.aggregator import plotting


@pytest.mark.parametrize("diverging", [False, True])
def test_empty_channel_image_has_no_numeric_colour_scale(monkeypatch, diverging):
    def render(fig, caption):
        FigureCanvasAgg(fig).draw()
        assert "No finite data." in caption
        assert "vmin=" not in caption
        assert fig.axes[0].images[0].get_array().shape == (2, 3)
        return "rendered"

    monkeypatch.setattr(
        plotting.WandBLogger, "get_instance", lambda: SimpleNamespace(Image=render)
    )
    assert (
        plotting.plot_paneled_data([[np.full((2, 3), np.nan)]], diverging) == "rendered"
    )
