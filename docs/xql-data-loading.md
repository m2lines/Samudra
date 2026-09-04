<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Experimental Xarray-SQL data loading

Samudra can route `CanonicalSource.read` requests through Xarray-SQL (XQL).
The experiment leaves source opening and canonicalization unchanged, so it
currently reads the existing Zarr v2 stores through Xarray and Zarr-Python.
XQL performs projection and time-partition filtering at the canonical reader
boundary.

Install the optional dependency and wrap a source before constructing a
`TorchTrainDataset`:

```bash
uv sync --group xql
```

```python
from samudra.utils.xql import with_xql_reader

xql_source = with_xql_reader(source, time_chunk_size=1)
```

The XQL context is deliberately created on the first `read`, rather than when
the source is constructed. PyTorch DataLoader workers therefore initialize
DataFusion after they fork, avoiding inheritance of DataFusion's Tokio runtime
(xarray-sql issue #145). Do not perform an XQL read in the parent before
forking workers; XQL does not yet expose the shutdown hook needed to make that
lifecycle safe.

Run the focused tests, including a correctness check against the public 2° OM4
store, with:

```bash
uv run --group xql pytest tests/test_xql_data.py -m "not manual"
uv run --group xql pytest tests/test_xql_data.py -m manual
```

The public store is Zarr v2. VirtualiZarr can represent it as virtual Zarr v3
references in Icechunk without copying chunks, but that path should be added
only alongside a native backend that can consume the Icechunk session cleanly.
