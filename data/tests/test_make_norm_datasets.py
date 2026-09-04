# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from ocean_preprocessing.make_norm_datasets import write_zarr_with_retries


def test_write_zarr_uses_dask_task_retries():
    ds = Mock()
    client = Mock()
    delayed = object()
    ds.to_zarr.return_value = delayed

    write_zarr_with_retries(ds, "s3://test/output.zarr", client, write_retries=7)

    ds.to_zarr.assert_called_once_with(
        "s3://test/output.zarr",
        mode="w",
        zarr_format=2,
        consolidated=True,
        compute=False,
    )
    client.compute.assert_called_once_with(delayed, retries=7, sync=True)
