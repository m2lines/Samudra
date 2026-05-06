import torch

from ocean_emulators.aggregator.validate.main import (
    SURFACE_SNAPSHOT_NAMES,
    ValidateAggregator,
)
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator


def _field(value: float) -> torch.Tensor:
    return torch.full((1, 2, 1, 1), value)


def test_validate_aggregator_surface_snapshot_skips_mean_map(monkeypatch):
    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.main.Normalize.get_instance",
        lambda: object(),
    )
    agg = ValidateAggregator(
        metadata={},
        hist=1,
        area_weights=torch.ones(1, 1),
        wet=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        num_prognostic_channels=5,
        surface_snapshot=True,
    )

    assert set(agg._aggregators.keys()) == {"snapshot", "reduced"}
    assert agg._aggregators["snapshot"]._include_names == SURFACE_SNAPSHOT_NAMES


def test_validate_aggregator_full_mode_keeps_mean_map(monkeypatch):
    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.main.Normalize.get_instance",
        lambda: object(),
    )
    agg = ValidateAggregator(
        metadata={},
        hist=1,
        area_weights=torch.ones(1, 1),
        wet=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        num_prognostic_channels=5,
        surface_snapshot=False,
    )

    assert set(agg._aggregators.keys()) == {"snapshot", "mean_map", "reduced"}


def test_snapshot_aggregator_surface_filter_only_logs_selected_fields(monkeypatch):
    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.snapshot.plot_paneled_data",
        lambda data, diverging, caption: caption,
    )

    agg = SnapshotAggregator(
        metadata={
            "Theta": {
                "long_name": "Sea Water Potential Temperature",
                "units": "degC",
            },
            "Eta": {
                "long_name": "Sea surface height above geoid",
                "units": "m",
            },
        },
        hist=1,
        include_names=("Theta_0", "Eta"),
    )
    agg.record_batch(
        loss=torch.tensor(0.0),
        target_data={"Theta_0": _field(1.0), "Theta_1": _field(2.0), "Eta": _field(3.0)},
        gen_data={"Theta_0": _field(1.5), "Theta_1": _field(2.5), "Eta": _field(3.5)},
        input_data={"Theta_0": _field(0.5), "Theta_1": _field(1.5), "Eta": _field(2.5)},
        target_data_norm={},
        gen_data_norm={},
        input_data_norm={},
    )

    logs = agg.get_logs(label="snapshot")

    assert set(logs.keys()) == {
        "snapshot/image-error/Eta",
        "snapshot/image-error/Theta_0",
        "snapshot/image-full-field/Eta",
        "snapshot/image-full-field/Theta_0",
        "snapshot/image-residual/Eta",
        "snapshot/image-residual/Theta_0",
    }
    assert "Sea Water Potential Temperature" in logs["snapshot/image-full-field/Theta_0"]
