import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from numcodecs import Blosc  # type: ignore[import-untyped]

from ocean_emulators.config import GpuDataLoadingConfig
from ocean_emulators.train import Trainer
from ocean_emulators.utils.location import LocalLocation
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")


@pytest.mark.cuda
def test_direct_zarr_gpu_decode_roundtrip_v3():
    _require_cuda()
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    zarr = pytest.importorskip("zarr")
    codecs = pytest.importorskip("zarr.codecs")
    BloscCodec = codecs.BloscCodec
    BloscShuffle = codecs.BloscShuffle

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = Path(tmp_dir) / "tiny_v3.zarr"
        expected = np.arange(64, dtype=np.float32).reshape(4, 4, 4)

        arr = zarr.create_array(
            store=str(store),
            shape=expected.shape,
            chunks=(2, 2, 2),
            dtype="float32",
            zarr_format=3,
            compressors=[
                BloscCodec(cname="zstd", shuffle=BloscShuffle.bitshuffle),
            ],
            overwrite=True,
        )
        arr[:] = expected

        with zarr.config.enable_gpu():
            gpu_arr = zarr.open_array(str(store), mode="r")
            out = gpu_arr[:]

        assert isinstance(out, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(out), expected)


@pytest.mark.cuda
def test_local_location_open_with_gpu_decode_reports_xarray_failure():
    _require_cuda()
    pytest.importorskip("zarr")
    cupy = pytest.importorskip("cupy")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zarr_path = Path(tmp_dir) / "tiny_v2.zarr"
        ds = xr.Dataset(
            {
                "temp": (
                    ("time", "lat", "lon"),
                    np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
                )
            }
        )
        ds.to_zarr(
            zarr_path,
            mode="w",
            zarr_format=2,
            encoding={
                "temp": {
                    "compressor": Blosc(
                        cname="zstd",
                        clevel=5,
                        shuffle=Blosc.BITSHUFFLE,
                    )
                }
            },
        )

        loc = LocalLocation(path=zarr_path)
        try:
            opened = loc.open(use_gpu_zarr_decode=True)
        except RuntimeError as exc:
            assert "GPU zarr decode failed while opening" in str(exc)
            return

        # If xarray+GPU decoding becomes compatible in the environment,
        # still validate the data path.
        loaded = opened["temp"].data
        compute = getattr(loaded, "compute", None)
        if callable(compute):
            loaded = loaded.compute()
        if isinstance(loaded, cupy.ndarray):
            loaded = cupy.asnumpy(loaded)
        else:
            loaded = np.asarray(loaded)
        np.testing.assert_array_equal(
            loaded,
            ds["temp"].values,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["cuda"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock", DEFAULT_CONFIG)],
    indirect=True,
)
def test_tiny_training_gpu_decode_smoke(train_config):
    _require_cuda()

    train_config.data.loading = GpuDataLoadingConfig()
    train_config.epochs = 1
    train_config.save_freq = 1000

    with MultitonScope():
        try:
            trainer = Trainer(train_config)
        except RuntimeError as exc:
            assert "GPU zarr decode failed while opening" in str(exc)
            return

        trainer.run()
