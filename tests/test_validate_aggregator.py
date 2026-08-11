import torch

from ocean_emulators.aggregator.validate.main import (
    SURFACE_SNAPSHOT_NAMES,
    ValidateAggregator,
)
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.multiton import MultitonScope


def _field(value: float) -> torch.Tensor:
    return torch.full((1, 2, 1, 1), value)


def test_validate_aggregator_surface_snapshot_keeps_only_snapshots(monkeypatch):
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

    assert set(agg._aggregators.keys()) == {"snapshot"}
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


def test_validate_aggregator_reports_one_step_loss(monkeypatch):
    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.main.Normalize.get_instance",
        lambda: object(),
    )
    for module in ("snapshot", "map"):
        monkeypatch.setattr(
            f"ocean_emulators.aggregator.validate.{module}.plot_paneled_data",
            lambda data, diverging, caption: caption,
        )

    # The gradient-magnitude metric needs at least a 2x2 grid.
    def grid(value: float) -> torch.Tensor:
        return torch.full((1, 2, 2, 2), value)

    def build(surface_snapshot: bool) -> ValidateAggregator:
        agg = ValidateAggregator(
            metadata={},
            hist=0,
            area_weights=torch.ones(2, 2),
            wet=torch.ones(1, 2, 2, dtype=torch.bool),
            num_prognostic_channels=1,
            surface_snapshot=surface_snapshot,
        )
        agg._n_batches = 1
        agg._loss = torch.tensor(0.5)
        agg._loss_per_channel = torch.tensor([0.5])
        for sub_aggregator in agg._aggregators.values():
            sub_aggregator.record_batch(
                loss=torch.tensor(0.5),
                target_data={"Eta": grid(1.0)},
                gen_data={"Eta": grid(1.5)},
                input_data={"Eta": grid(0.5)},
                target_data_norm={"Eta": grid(1.0)},
                gen_data_norm={"Eta": grid(1.5)},
                input_data_norm={"Eta": grid(0.5)},
            )
        return agg

    surface_logs = build(surface_snapshot=True).get_logs(label="val")
    assert surface_logs["val/mean/one-step-loss"] == 0.5
    # Surface-snapshot mode reports the one-step loss and the snapshots, nothing else.
    assert "val/mean/loss" not in surface_logs
    assert not [key for key in surface_logs if key.startswith("val/reduced/")]

    with MultitonScope():
        TensorMap.init_instance("single_1", "all")
        full_logs = build(surface_snapshot=False).get_logs(label="val")
    assert full_logs["val/mean/one-step-loss"] == 0.5
    # Full mode still carries every legacy key so old dashboards keep working.
    assert full_logs["val/mean/loss"] == 0.5


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
