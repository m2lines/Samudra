from pathlib import Path

import pytest

from ocean_emulators.config import DataConfig, DataSourceConfig, TrainConfig
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


def test_llc_train_config_uses_sources_and_gpu_decode(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "samudra_llc" / "train.yaml"

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert len(cfg.data.sources) == 1
    assert cfg.data.sources[0].data_location == UnresolvedLocation(
        path="003/LLC4320/LLC4320"
    )
    assert cfg.data.sources[0].data_means_location == UnresolvedLocation(
        path="002/cody/LLC_means_stds/var_96_LLC_means.zarr"
    )
    assert cfg.data.sources[0].data_stds_location == UnresolvedLocation(
        path="002/cody/LLC_means_stds/var_96_LLC_stds.zarr"
    )
    assert cfg.data.num_workers == 0
    assert cfg.data.zarr_gpu_decode is True
