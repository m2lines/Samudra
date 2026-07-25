# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import xarray as xr

from samudra.aggregator.inference import InferenceEvaluatorAggregator
from samudra.config import OtterWeightedRmseLossConfig, build_loss_fn
from samudra.constants import DatasetSpec, TensorMap
from samudra.datasets import (
    InferenceDataset,
    TorchTrainDataset,
    temporal_fourier_embedding,
)
from samudra.models.base import BaseModel
from samudra.models.modules.otter import FourierPositionEmbedding
from samudra.stepper import run_rollout
from samudra.utils.ctx import GridContext
from samudra.utils.data import DataSource, Masks
from samudra.utils.loss import OtterWeightedRmseLoss
from samudra.utils.optimizer import CompositeOptimizer, MuonOptimizerConfig
from samudra.utils.schedule import WarmupCosineUpdatesConfig


def make_spec() -> DatasetSpec:
    return DatasetSpec(
        type="om4",
        depth_levels=(1.0, 2.0),
        depth_thickness=(1.0, 3.0),
        mask_vars=("mask_0", "mask_1"),
        mask_all_levels_var="wetmask",
        seconds_per_time_step=1,
        prognostic_var_names=["thetao_0", "thetao_1", "zos"],
        boundary_var_names=["hfds"],
        default_metadata={},
        ocean_heat_temperature_var="thetao",
        surface_heat_flux_var="hfds",
    )


def test_otter_weighted_rmse_uses_normalized_thickness_and_sum_reduction():
    tensor_map = TensorMap(make_spec())
    loss_fn = build_loss_fn(
        OtterWeightedRmseLossConfig(),
        device=torch.device("cpu"),
        num_channels=3,
        pad_mode="circular",
        tensor_map=tensor_map,
    )
    assert isinstance(loss_fn, OtterWeightedRmseLoss)
    pred = torch.ones((1, 3, 1, 1))
    target = torch.zeros_like(pred)
    coordinate = torch.zeros(1)
    ctx = GridContext(
        torch.ones((3, 1, 1), dtype=torch.bool),
        (coordinate, coordinate),
        (coordinate, coordinate),
    )

    per_channel = loss_fn(pred, target, ctx)

    torch.testing.assert_close(per_channel, torch.tensor([0.25, 0.75, 1.0]))
    torch.testing.assert_close(loss_fn.reduce(per_channel), torch.tensor(2.0))


def test_otter_weighted_rmse_has_finite_gradient_at_exact_target():
    tensor_map = TensorMap(make_spec())
    loss_fn = build_loss_fn(
        OtterWeightedRmseLossConfig(),
        device=torch.device("cpu"),
        num_channels=3,
        pad_mode="circular",
        tensor_map=tensor_map,
    )
    assert isinstance(loss_fn, OtterWeightedRmseLoss)
    pred = torch.zeros((1, 3, 2, 2), requires_grad=True)
    target = torch.zeros_like(pred)
    latitude = torch.tensor([-45.0, 45.0])
    longitude = torch.tensor([0.0, 180.0])
    ctx = GridContext(
        torch.ones((3, 2, 2), dtype=torch.bool),
        (latitude, longitude),
        (latitude, longitude),
    )

    loss = loss_fn.reduce(loss_fn(pred, target, ctx))
    loss.backward()

    assert loss.item() == 0.0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) == 0


def test_time_embedding_matches_official_reference_and_is_continuous():
    reference = temporal_fourier_embedding(
        [np.datetime64("1979-01-01T00:00:00")],
        num_scales=16,
        min_hours=3.0,
        max_hours=8760.0,
    )
    later = temporal_fourier_embedding(
        [np.datetime64("1979-01-01T03:00:00")],
        num_scales=16,
        min_hours=3.0,
        max_hours=8760.0,
    )

    assert reference.shape == (1, 32)
    torch.testing.assert_close(reference[0, :16], torch.zeros(16))
    torch.testing.assert_close(reference[0, 16:], torch.ones(16))
    assert not torch.equal(reference, later)


def test_paper_spatial_embedding_uses_128_features_per_coordinate():
    embedding = FourierPositionEmbedding(
        token_dim=16,
        num_features=256,
        min_scale=0.1,
        max_scale=720.0,
    )

    assert embedding.scales.numel() == 64
    assert embedding.projection.in_features == 256


def make_index_source() -> DataSource:
    time = np.arange(
        np.datetime64("1979-01-01"),
        np.datetime64("1979-01-11"),
        dtype="datetime64[D]",
    )
    coords = {"time": time, "lat": [0.0], "lon": [0.0]}
    values = np.arange(time.size, dtype=np.float32)[:, None, None]
    data = xr.Dataset(
        {
            "thetao_0": (("time", "lat", "lon"), values),
            "thetao_1": (("time", "lat", "lon"), values + 100),
            "zos": (("time", "lat", "lon"), values + 200),
            "hfds": (("time", "lat", "lon"), values + 300),
        },
        coords=coords,
    )
    means = data.mean(("time", "lat", "lon"))
    stds = xr.ones_like(means)
    return DataSource(
        name="index",
        data=data,
        means=means,
        stds=stds,
        masks=Masks(torch.ones(3, 1, 1), torch.ones(1, 1)),
        dataset_spec=make_spec(),
    )


def test_four_in_one_out_dataset_indices_and_timeline_are_exact():
    source = make_index_source()
    dataset = TorchTrainDataset(
        src=source,
        prognostic_var_names=make_spec().prognostic_var_names,
        boundary_var_names=make_spec().boundary_var_names,
        hist=3,
        output_steps=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )

    assert len(dataset) == 5
    sample = dataset[0]
    first_input, first_boundary, first_label, _ = sample.raw_data[0]
    second_input, second_boundary, second_label, _ = sample.raw_data[1]
    torch.testing.assert_close(first_input[:, 0, 0, 0], torch.arange(4.0))
    torch.testing.assert_close(first_boundary[:, 0, 0, 0], torch.arange(300.0, 304.0))
    torch.testing.assert_close(first_label[:, 0, 0, 0], torch.tensor([4.0]))
    torch.testing.assert_close(second_input[:, 0, 0, 0], torch.arange(1.0, 5.0))
    torch.testing.assert_close(second_boundary[:, 0, 0, 0], torch.arange(301.0, 305.0))
    torch.testing.assert_close(second_label[:, 0, 0, 0], torch.tensor([5.0]))

    inference = InferenceDataset(
        src=source,
        prognostic_var_names=make_spec().prognostic_var_names,
        boundary_var_names=make_spec().boundary_var_names,
        hist=3,
        output_steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        long_rollout=True,
    )
    assert len(inference) == 6
    assert inference.num_timeline_steps == 10


def test_muon_uses_official_lr_adjustment_and_refreshes_loaded_groups():
    model = torch.nn.Linear(4, 3)
    optimizer = MuonOptimizerConfig().build(model, 2e-4)
    assert isinstance(optimizer, CompositeOptimizer)
    assert optimizer.optimizers[0].defaults["adjust_lr_fn"] == "match_rms_adamw"
    state = optimizer.state_dict()

    restored = MuonOptimizerConfig().build(torch.nn.Linear(4, 3), 2e-4)
    assert isinstance(restored, CompositeOptimizer)
    restored.load_state_dict(state)

    assert restored.param_groups[0] is restored.optimizers[0].param_groups[0]
    assert restored.param_groups[-1] is restored.optimizers[-1].param_groups[-1]


def test_update_warmup_cosine_hits_peak_and_floor_and_resumes():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=2e-4)
    config = WarmupCosineUpdatesConfig(
        total_updates=12,
        warmup_updates=4,
        min_lr=2e-6,
    )
    scheduler = config.build(optimizer, epochs=1)
    history = []
    for _ in range(6):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()

    resumed_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=2e-4)
    resumed = config.build(resumed_optimizer, epochs=1)
    resumed.load_state_dict(state)
    resumed_optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
    for _ in range(6):
        history.append(resumed_optimizer.param_groups[0]["lr"])
        resumed_optimizer.step()
        resumed.step()

    assert abs(history[0] - 2e-14) < 1e-25
    assert max(history) == 2e-4
    assert abs(history[-1] - 2e-6) < 1e-12


class ContextSumModel(BaseModel):
    def forward_once(self, prognostic, boundary, ctx):
        del boundary, ctx
        return prognostic.sum(dim=1, keepdim=True)


class TinyInferenceDataset(InferenceDataset):
    def __init__(self):
        self._initial_prognostic = torch.arange(4.0).reshape(1, 4, 1, 1)
        coordinate = torch.zeros(1)
        self.ctx = GridContext(
            torch.ones((1, 1, 1), dtype=torch.bool),
            (coordinate, coordinate),
            (coordinate, coordinate),
        )

    @property
    def initial_prognostic(self):
        return self._initial_prognostic

    def __len__(self):
        return 5

    def __getitem__(self, index):
        del index
        return (
            self.initial_prognostic,
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
        )

    def to(self, device):
        del device
        return self

    def get_boundary(self, step):
        del step
        return torch.zeros((1, 1, 1, 1))

    def get_target_time(self, start_step, num_steps):
        return xr.DataArray(
            np.arange(start_step, start_step + num_steps), dims=["time"]
        )

    def inference_target(self, step):
        count = step.stop - step.start
        return torch.zeros((count, 1, 1, 1))


class RecordingAggregator(InferenceEvaluatorAggregator):
    hist = 0

    def __init__(self):
        self.predictions = []

    def record_initial_prognostic(self, initial_prognostic):
        del initial_prognostic
        return {}

    def record_batch(self, output):
        self.predictions.extend(output.prediction.flatten().tolist())
        return {}

    def get_summary_logs(self):
        return {}


def test_chunked_inference_preserves_many_to_one_context(monkeypatch):
    monkeypatch.setattr(
        "samudra.stepper.get_record_to_wandb", lambda label: lambda logs: None
    )
    model = ContextSumModel(
        in_channels=4,
        out_channels=1,
        hist=3,
        pred_residuals=False,
        last_kernel_size=1,
        pad="circular",
        gradient_detach_interval=0,
    )
    aggregator = RecordingAggregator()

    run_rollout(
        model=model,
        dataset=TinyInferenceDataset(),
        inf_aggregator=aggregator,
        epoch=0,
        num_model_steps_forward=2,
    )

    assert aggregator.predictions == [6.0, 12.0, 23.0, 44.0, 85.0]
