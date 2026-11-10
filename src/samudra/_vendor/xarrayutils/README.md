<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# xarrayutils plotting helpers

`box_plot` and `linear_piecewise_scale` are copied from xarrayutils **2.0.1**:

https://github.com/jbusecke/xarrayutils/blob/c6e371c9ba45660f2bef2d47ae48c84a2c65ba1e/xarrayutils/plotting.py

Copyright (c) 2017 Julius Busecke. Licensed under the MIT license;
the original permission and copyright notice is in `LICENSE.txt` and ships
with the package.

Only the two functions used by Samudra are included, with their NumPy and
Matplotlib imports. The module-level Tkinter, Xarray, GSW and Cartopy imports
and forced Matplotlib backend are omitted. Function behavior is unchanged;
formatting and docstring layout follow Samudra's checks. To update, extract
these two functions from the upstream release, preserve its notices, and run
`tests/test_vendored_plotting.py` and the visualization tests.
