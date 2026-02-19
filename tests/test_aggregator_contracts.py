import numpy as np
import pytest
import torch
import wandb
import xarray as xr

from ocean_emulators.aggregator.inference.main import (
    InferenceEvaluatorAggregator,
    to_inference_logs,
)
from ocean_emulators.aggregator.inference.reduced import (
    MeanAggregator as InfMeanAggregator,
)
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.main import ValidateAggregator
from ocean_emulators.aggregator.validate.map import MapAggregator
from ocean_emulators.aggregator.validate.reduced import (
    MeanAggregator as ValMeanAggregator,
)
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator
from ocean_emulators.constants import DEPTH_LEVELS, TensorMap
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.output import (
    ModelInferenceOutput,
    TrainBatchOutput,
    ValBatchOutput,
)
from ocean_emulators.utils.wandb import WandBLogger


def _make_source_thetao_1() -> DataSource:
    time = 8
    lat = 2
    lon = 2
    lev = 19
    thetao = np.arange(time * lat * lon, dtype=np.float32).reshape(time, lat, lon)
    hfds = thetao + 100.0
    wetmask = np.ones((time, lev, lat, lon), dtype=np.float32)

    data = xr.Dataset(
        {
            "thetao_0": (("time", "lat", "lon"), thetao),
            "hfds": (("time", "lat", "lon"), hfds),
            "wetmask": (("time", "lev", "lat", "lon"), wetmask),
        },
        coords={
            "time": np.arange(time),
            "lev": DEPTH_LEVELS,
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )
    means = data.mean() * 0.0
    stds = data.std() * 0.0 + 1.0
    tensor_map = TensorMap.get_instance()
    return DataSource.from_datasets(
        data,
        means,
        stds,
        name="test-source",
        prognostic_var_names=tensor_map.prognostic_var_names,
        boundary_var_names=tensor_map.boundary_var_names,
    )


def _image_type():
    return type(wandb.Image(np.zeros((2, 2))))


def _coerce_index_tensors_to_int(tensor_map: TensorMap):
    for key in list(tensor_map.DP_3D_IDX):
        tensor_map.DP_3D_IDX[key] = tensor_map.DP_3D_IDX[key].to(torch.int64)
    for key in list(tensor_map.VAR_3D_IDX):
        tensor_map.VAR_3D_IDX[key] = tensor_map.VAR_3D_IDX[key].to(torch.int64)


def _round_number(value, digits: int = 3):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if np.isnan(float_value):
            return "nan"
        if np.isposinf(float_value):
            return "inf"
        if np.isneginf(float_value):
            return "-inf"
        return round(float_value, digits)
    return value


def _canonicalize(value):
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _round_number(value.item())
        return _canonicalize(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, wandb.Table):
        return {
            "__type": "wandb.Table",
            "columns": list(value.columns),
            "data": [[_canonicalize(cell) for cell in row] for row in value.data],
        }
    if isinstance(value, _image_type()):
        pixels = np.asarray(value.image)
        return {
            "__type": "wandb.Image",
            "caption": getattr(value, "_caption", None),
            "shape": list(pixels.shape),
            "pixels": _canonicalize(pixels),
        }
    return _round_number(value)


def test_train_aggregator_get_logs_contract(snapshot):
    with MultitonScope():
        tensor_map = TensorMap.init_instance("thermo_dynamic_5", "hfds")
        agg = TrainAggregator()
        n_channels = len(tensor_map.prognostic_var_names)

        agg.record_batch(
            TrainBatchOutput(
                loss=torch.tensor(2.0),
                loss_per_channel=torch.arange(n_channels, dtype=torch.float32),
            )
        )
        agg.record_batch(
            TrainBatchOutput(
                loss=torch.tensor(4.0),
                loss_per_channel=torch.arange(n_channels, dtype=torch.float32) + 2.0,
            )
        )

        logs = agg.get_logs(label="train")
        assert list(logs.keys()) == sorted(logs.keys())
        assert logs["train/mean/loss"] == pytest.approx(3.0)
        assert logs[
            f"train/loss/channel/{tensor_map.prognostic_var_names[0]}_loss"
        ] == pytest.approx(1.0)
        assert "train/loss/depth/depth_0_loss" in logs
        assert "train/loss/variable/thetao_loss" in logs
        assert _canonicalize(logs) == snapshot


def test_snapshot_aggregator_get_logs_contract(snapshot):
    image_type = _image_type()
    with MultitonScope():
        WandBLogger.init_instance()
        agg = SnapshotAggregator(
            metadata={"thetao_0": {"long_name": "theta", "units": "K"}},
            hist=0,
        )
        target = {"thetao_0": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])}
        gen = {"thetao_0": torch.tensor([[[[1.1, 1.9], [2.8, 4.2]]]])}
        input_data = {"thetao_0": torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])}
        agg.record_batch(
            loss=torch.tensor(1.0),
            target_data=target,
            gen_data=gen,
            input_data=input_data,
            target_data_norm=target,
            gen_data_norm=gen,
            input_data_norm=input_data,
        )
        logs = agg.get_logs(label="snapshot")
        assert set(logs.keys()) == {
            "snapshot/image-error/thetao_0",
            "snapshot/image-full-field/thetao_0",
            "snapshot/image-residual/thetao_0",
        }
        assert all(isinstance(v, image_type) for v in logs.values())
        assert _canonicalize(logs) == snapshot


def test_map_aggregator_get_logs_contract(snapshot):
    image_type = _image_type()
    with MultitonScope():
        WandBLogger.init_instance()
        agg = MapAggregator(
            metadata={"thetao_0": {"long_name": "theta", "units": "K"}},
            hist=0,
        )
        target = {
            "thetao_0": torch.tensor(
                [
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    [[[2.0, 3.0], [4.0, 5.0]]],
                ]
            )
        }
        gen = {
            "thetao_0": torch.tensor(
                [
                    [[[1.2, 2.1], [2.9, 3.8]]],
                    [[[2.1, 3.2], [4.2, 5.1]]],
                ]
            )
        }
        agg.record_batch(
            loss=torch.tensor(1.0),
            target_data=target,
            gen_data=gen,
            input_data=target,
            target_data_norm=target,
            gen_data_norm=gen,
            input_data_norm=target,
        )
        logs = agg.get_logs(label="mean_map")
        assert set(logs.keys()) == {
            "mean_map/image-error/thetao_0",
            "mean_map/image-full-field/thetao_0",
        }
        assert all(isinstance(v, image_type) for v in logs.values())
        assert _canonicalize(logs) == snapshot


def test_validate_reduced_aggregator_get_logs_contract(snapshot):
    agg = ValMeanAggregator(area_weights=torch.ones((2, 2)) / 4.0, target_time=0)
    target = {
        "thetao_0": torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[2.0, 3.0], [4.0, 5.0]]],
            ]
        )
    }
    gen = {
        "thetao_0": torch.tensor(
            [
                [[[1.2, 2.1], [2.9, 3.8]]],
                [[[2.1, 3.2], [4.2, 5.1]]],
            ]
        )
    }
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="reduced")
    assert set(logs.keys()) == {
        "reduced/weighted_bias/thetao_0",
        "reduced/weighted_grad_mag_percent_diff/thetao_0",
        "reduced/weighted_rmse/thetao_0",
    }
    assert all(np.isfinite(value) for value in logs.values())
    assert _canonicalize(logs) == snapshot


def test_inference_reduced_aggregator_table_contract(snapshot):
    agg = InfMeanAggregator(
        target="denorm",
        n_timesteps=5,
        area_weights=torch.ones((2, 2)) / 4.0,
    )
    target = {
        "thetao_0": torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]],
            ]
        )
    }
    gen = {
        "thetao_0": torch.tensor(
            [
                [[[1.1, 2.1], [3.1, 4.1]], [[2.2, 3.2], [4.1, 5.2]]],
            ]
        )
    }
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
        i_time_start=1,
    )
    logs = agg.get_logs(label="mean", step_slice=slice(1, 3))
    table = logs["mean/series"]
    assert table.columns[0] == "forecast_step"
    assert table.columns[1:] == sorted(table.columns[1:])
    assert len(table.data) == 2
    assert table.data[0][0] == 1
    assert table.data[1][0] == 2
    for row in table.data:
        assert len(row) == len(table.columns)
    assert _canonicalize(logs) == snapshot


def test_to_inference_logs_expands_wandb_tables(snapshot):
    table = wandb.Table(columns=["forecast_step", "metric"])
    table.add_data(0, 1.5)
    table.add_data(1, 2.5)
    logs = to_inference_logs({"mean/series": table, "time_mean/rmse": 0.5})
    assert logs == [
        {"mean/forecast_step": 0, "mean/metric": 1.5},
        {"mean/forecast_step": 1, "mean/metric": 2.5, "time_mean/rmse": 0.5},
    ]
    assert _canonicalize(logs) == snapshot


def test_inference_evaluator_aggregator_contract(snapshot):
    with MultitonScope():
        TensorMap.init_instance("thetao_1", "hfds")
        source = _make_source_thetao_1()
        Normalize.init_instance(
            source,
            prognostic_var_names=TensorMap.get_instance().prognostic_var_names,
            boundary_var_names=TensorMap.get_instance().boundary_var_names,
        )
        agg = InferenceEvaluatorAggregator(
            n_timesteps=3,
            metadata=source.metadata,
            hist=0,
            area_weights=torch.ones((2, 2)) / 4.0,
            wet=source.masks.prognostic,
            num_prognostic_channels=1,
            record_step_20=False,
            channel_mean_names=["thetao_0"],
        )

        initial_prognostic = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        init_logs = agg.record_initial_prognostic(initial_prognostic)
        assert len(init_logs) == 1
        assert init_logs[0]["mean/forecast_step"] == 0
        assert init_logs[0]["mean_norm/forecast_step"] == 0

        prediction = torch.tensor(
            [
                [[[2.0, 3.1], [3.8, 5.0]]],
                [[[3.0, 4.2], [4.9, 6.0]]],
            ]
        )
        target = torch.tensor(
            [
                [[[2.1, 3.0], [3.9, 4.8]]],
                [[[2.9, 4.0], [5.1, 5.8]]],
            ]
        )
        output = ModelInferenceOutput(
            prediction=prediction,
            target=target,
            time=xr.DataArray(np.array([0, 1]), dims=["time"]),
        )
        step_logs = agg.record_batch(output)
        assert len(step_logs) == 2
        assert step_logs[0]["mean/forecast_step"] == 1
        assert step_logs[1]["mean/forecast_step"] == 2

        summary_logs = agg.get_summary_logs()
        assert "time_mean/rmse/thetao_0" in summary_logs
        assert "time_mean_norm/rmse/thetao_0" in summary_logs
        assert "time_mean_norm/rmse/channel_mean" in summary_logs
        assert summary_logs["time_mean_norm/rmse/channel_mean"] == pytest.approx(
            summary_logs["time_mean_norm/rmse/thetao_0"]
        )
        snapshot_payload = {
            "init_logs": init_logs,
            "step_logs": step_logs,
            "summary_logs": summary_logs,
        }
        assert _canonicalize(snapshot_payload) == snapshot


def test_validate_aggregator_orchestration_contract(snapshot):
    with MultitonScope():
        tensor_map = TensorMap.init_instance("thetao_1", "hfds")
        _coerce_index_tensors_to_int(tensor_map)
        source = _make_source_thetao_1()
        Normalize.init_instance(
            source,
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
        )
        WandBLogger.init_instance()

        agg = ValidateAggregator(
            metadata=source.metadata,
            hist=0,
            area_weights=torch.ones((2, 2)) / 4.0,
            wet=source.masks.prognostic,
            num_prognostic_channels=1,
        )

        batch_1 = ValBatchOutput(
            loss=torch.tensor(1.2),
            loss_per_channel=torch.tensor([1.2]),
            input_data=torch.tensor(
                [
                    [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 11.0], [12.0, 13.0]]],
                    [[[2.0, 3.0], [4.0, 5.0]], [[20.0, 21.0], [22.0, 23.0]]],
                ]
            ),
            target_data=torch.tensor(
                [
                    [[[1.5, 2.5], [3.5, 4.5]]],
                    [[[2.5, 3.5], [4.5, 5.5]]],
                ]
            ),
            gen_data=torch.tensor(
                [
                    [[[1.4, 2.6], [3.3, 4.7]]],
                    [[[2.6, 3.4], [4.6, 5.3]]],
                ]
            ),
        )
        batch_2 = ValBatchOutput(
            loss=torch.tensor(2.2),
            loss_per_channel=torch.tensor([2.2]),
            input_data=torch.tensor(
                [
                    [[[2.0, 3.0], [4.0, 5.0]], [[30.0, 31.0], [32.0, 33.0]]],
                    [[[3.0, 4.0], [5.0, 6.0]], [[40.0, 41.0], [42.0, 43.0]]],
                ]
            ),
            target_data=torch.tensor(
                [
                    [[[2.5, 3.5], [4.5, 5.5]]],
                    [[[3.5, 4.5], [5.5, 6.5]]],
                ]
            ),
            gen_data=torch.tensor(
                [
                    [[[2.4, 3.7], [4.6, 5.6]]],
                    [[[3.6, 4.3], [5.4, 6.8]]],
                ]
            ),
        )

        agg.record_validation_batch(batch_1)
        agg.record_validation_batch(batch_2)
        logs = agg.get_logs(label="val")

        assert "val/mean/loss" in logs
        assert "val/reduced/weighted_rmse/thetao_0" in logs
        assert "val/snapshot/image-full-field/thetao_0" in logs
        assert "val/mean_map/image-error/thetao_0" in logs
        assert _canonicalize(logs) == snapshot
