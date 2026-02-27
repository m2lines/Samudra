import pytest

from ocean_emulators.config import DataConfig, DataSourceConfig
from ocean_emulators.utils.location import LocalLocation, UnresolvedLocation


def test_data_config_requires_zero_workers_for_gpu_zarr_decode(tmp_path):
    cfg = DataConfig(
        sources=[
            DataSourceConfig(
                data_location=UnresolvedLocation(path="data.zarr"),
                data_means_location=UnresolvedLocation(path="means.zarr"),
                data_stds_location=UnresolvedLocation(path="stds.zarr"),
            )
        ],
        num_workers=1,
        zarr_gpu_decode=True,
    )

    with pytest.raises(
        ValueError, match=r"zarr_gpu_decode=true requires data.num_workers=0"
    ):
        cfg.build(
            data_root=LocalLocation(path=tmp_path),
            prognostic_var_names=["thetao_0"],
            boundary_var_names=["hfds"],
        )
