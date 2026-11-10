<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Aurora positional encodings

Vendored from Microsoft Aurora **v1.8.0**, commit
`fac03c738554cfa804373267ca5621ba85b2614c`:

- https://github.com/microsoft/aurora/blob/fac03c738554cfa804373267ca5621ba85b2614c/aurora/model/fourier.py
- https://github.com/microsoft/aurora/blob/fac03c738554cfa804373267ca5621ba85b2614c/aurora/model/posencoding.py
- https://github.com/microsoft/aurora/blob/fac03c738554cfa804373267ca5621ba85b2614c/LICENSE.txt

Copyright (c) Microsoft Corporation. Licensed under the MIT license;
the original permission and copyright notice is in `LICENSE.txt` and ships
with the package. The upstream ClimaX inspiration attribution is preserved
in `posencoding.py`.

Only position/scale Fourier expansions and their latitude/longitude helpers
are included. Changes from upstream:

- Removed unused time and pressure expansions.
- Replaced NumPy's pi with the identical `math.pi` constant.
- Froze the original float64 minimum patch area to avoid importing the
  unrelated polygon-area implementation. Earth radius and wavelength bounds
  retain their original values, including the original scale conventions.
- Replaced timm's tuple helper with local scalar-to-pair normalization.
- Corrected the return annotation and used PyTorch's `dim` keyword.
- Applied Samudra's formatting and SPDX headers.

Keep encoding values and patch ordering compatible with v1.8.0 when updating:
trained encoder/decoder checkpoints depend on them. The parity tests use
fixtures recorded from that upstream release, without requiring Aurora.
