# SPDX-FileCopyrightText: 2017 Julius Busecke
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: MIT

"""Plotting helpers copied from xarrayutils 2.0.1.

Copyright (c) 2017 Julius Busecke. Licensed under the MIT license;
see LICENSES/MIT.txt.

Source: https://github.com/jbusecke/xarrayutils/blob/c6e371c9ba45660f2bef2d47ae48c84a2c65ba1e/xarrayutils/plotting.py

Only box_plot and linear_piecewise_scale, with their NumPy and Matplotlib
imports, are included. Module-level Tkinter, Xarray, GSW and Cartopy imports
and the forced Matplotlib backend are omitted. Function behavior is unchanged;
formatting, docstring layout and SPDX headers follow Samudra's checks.

To update, extract these functions from the upstream release, preserve its
notices, and run tests/test_viz_plotting.py and the other visualization tests.
"""

import matplotlib.pyplot as plt
import numpy as np


def box_plot(box, ax=None, split_detection="True", **kwargs):
    """Plots box despite coordinate discontinuities.

    INPUT
    -----
    box: np.array
        Defines the box in the coordinates of the current axis.
        Describing the box corners [x1, x2, y1, y2]
    ax: matplotlib.axis
        axis for plotting. Defaults to plt.gca()
    kwargs: optional
        anything that can be passed to plot can be put as kwarg
    """
    if len(box) != 4:
        raise RuntimeError(
            "'box' must be a 4 element np.array, \
            describing the box corners [x1, x2, y1, y2]"
        )
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    x_split = False
    y_split = False

    if ax is None:
        ax = plt.gca()

    if split_detection:
        if np.diff([box[0], box[1]]) < 0:
            x_split = True

        if np.diff([box[2], box[3]]) < 0:
            y_split = True

    if y_split and not x_split:
        ax.plot(
            [box[0], box[0], box[1], box[1], box[0]],
            [ylim[1], box[2], box[2], ylim[1], ylim[1]],
            **kwargs,
        )

        ax.plot(
            [box[0], box[0], box[1], box[1], box[0]],
            [ylim[0], box[3], box[3], ylim[0], ylim[0]],
            **kwargs,
        )

    elif x_split and not y_split:
        ax.plot(
            [xlim[1], box[0], box[0], xlim[1], xlim[1]],
            [box[2], box[2], box[3], box[3], box[2]],
            **kwargs,
        )

        ax.plot(
            [xlim[0], box[1], box[1], xlim[0], xlim[0]],
            [box[2], box[2], box[3], box[3], box[2]],
            **kwargs,
        )

    elif x_split and y_split:
        ax.plot([xlim[1], box[0], box[0]], [box[2], box[2], ylim[1]], **kwargs)

        ax.plot([xlim[0], box[1], box[1]], [box[2], box[2], ylim[1]], **kwargs)

        ax.plot([xlim[1], box[0], box[0]], [box[3], box[3], ylim[0]], **kwargs)

        ax.plot([xlim[0], box[1], box[1]], [box[3], box[3], ylim[0]], **kwargs)

    elif not x_split and not y_split:
        ax.plot(
            [box[0], box[0], box[1], box[1], box[0]],
            [box[2], box[3], box[3], box[2], box[2]],
            **kwargs,
        )


def linear_piecewise_scale(
    cut, scale, ax=None, axis="y", scaled_half="upper", add_cut_line=False
):
    """This function sets a piecewise linear scaling for a given axis to highlight e.g. processes in the upper ocean vs deep ocean.

    Parameters
    ----------
    cut : float
        value along the chosen axis used as transition between the two linear scalings.
    scale : float
        scaling coefficient for the chosen axis portion (determined by `axis` and `scaled_half`).
        A higher number means the chosen portion of the axis will be more compressed. Must be positive. 0 means no compression.
    ax : matplotlib.axis, optional
        The plot axis object. Defaults to current matplotlib axis
    axis : str, optional
        Which axis of the plot to act on.
        * 'y' (Default)
        * 'x'
    scaled_half: str, optional
        Determines which half of the axis is scaled (compressed).
        * 'upper' (default). Values larger than `cut` are compressed
        * 'lower'. Values smaller than `cut` are compressed


    Returns:
    -------
    ax_scaled : matplotlib.axis

    """
    if ax is None:
        ax = plt.gca()

    if scale < 0:
        raise ValueError(f"`Scale can not be negative. Got value of {scale}")

    if scale == 0:
        # do nothing
        return ax
    else:
        if scaled_half == "upper":

            def inverse(x):
                return np.piecewise(
                    x,
                    [x <= cut, x > cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

            def forward(x):
                return np.piecewise(
                    x,
                    [x <= cut, x > cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

        elif scaled_half == "lower":

            def inverse(x):
                return np.piecewise(
                    x,
                    [x >= cut, x < cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

            def forward(x):
                return np.piecewise(
                    x,
                    [x >= cut, x < cut],
                    [lambda x: x + (scale * (x - cut)), lambda x: x],
                )

        else:
            raise ValueError(
                f"`scaled_half` value not recognized. Must be ['upper', 'lower']. Got {scaled_half}"
            )

        if axis == "y":
            axlim = ax.get_ylim()
            ax.set_yscale("function", functions=(forward, inverse))
            ax.set_ylim(axlim)
        elif axis == "x":
            axlim = ax.get_xlim()
            ax.set_xscale("function", functions=(forward, inverse))
            ax.set_xlim(axlim)
        else:
            raise ValueError(
                f"`axis` value not recognized. Must be ['x', 'y']. Got {axis}"
            )

        return ax
