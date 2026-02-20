import torch

from ocean_emulators.aggregator.inference.main import to_inference_logs
from ocean_emulators.aggregator.spectra import SpectraLogger


def test_spectra_logger_logs_only_prognostic_vars():
    lat = torch.linspace(-90.0, 90.0, 32)
    lon = torch.linspace(0.0, 360.0, 64)

    spectra_logger = SpectraLogger(
        lat=lat,
        lon=lon,
        locations=[("test_box", (180.0, 243.0), (-40.0, 35.0))],
        prognostic_var_names=["vo_12", "thetao_1"],
        metadata=None,
    )

    target_data = {
        "vo_12": torch.rand(1, 1, 32, 64),
        "thetao_1": torch.rand(1, 1, 32, 64),
        "ke": torch.rand(1, 1, 32, 64),
    }
    gen_data = {
        "vo_12": torch.rand(1, 1, 32, 64),
        "thetao_1": torch.rand(1, 1, 32, 64),
        "ke": torch.rand(1, 1, 32, 64),
    }

    logs = spectra_logger.get_logs_for_single_step(
        target_data=target_data,
        gen_data=gen_data,
        time_index=0,
        sample_index=0,
    )

    assert "spectra_test_box/vo_12" in logs
    assert "spectra_test_box/thetao_1" in logs
    assert "spectra_test_box/ke" not in logs


def test_spectra_logger_returns_per_step_logs_for_inference():
    lat = torch.linspace(-90.0, 90.0, 32)
    lon = torch.linspace(0.0, 360.0, 64)

    spectra_logger = SpectraLogger(
        lat=lat,
        lon=lon,
        locations=[("test_box", (180.0, 243.0), (-40.0, 35.0))],
        prognostic_var_names=["vo_12"],
        metadata=None,
    )

    target_data = {"vo_12": torch.rand(1, 3, 32, 64)}
    gen_data = {"vo_12": torch.rand(1, 3, 32, 64)}

    logs = spectra_logger.get_logs_for_all_steps(
        target_data=target_data,
        gen_data=gen_data,
        sample_index=0,
        key_prefix="snapshot/",
        forecast_step_offset=7,
    )

    assert len(logs) == 3
    for step_log in logs:
        assert "snapshot/spectra_test_box/vo_12" in step_log


def test_to_inference_logs_honors_row_hint():
    logs = to_inference_logs({}, n_rows_hint=4)
    assert len(logs) == 4
