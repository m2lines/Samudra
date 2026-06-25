import tempfile
from pathlib import Path

import torch
import xarray as xr

from ocean_emulators.datasets import ReplayCursor, TorchTrainDataset
from ocean_emulators.models.base import BaseModel
from ocean_emulators.replay import ReplayBuffer, ReplayEntry, replay_sidecar_path
from ocean_emulators.train import Trainer
from ocean_emulators.utils.data import DataSource, Masks


class ConstantResidualModel(BaseModel):
    def __init__(self, *, pred_residuals: bool):
        super().__init__(
            in_channels=2,
            out_channels=1,
            wet=torch.ones(1, 1, 1, dtype=torch.bool),
            hist=0,
            pred_residuals=pred_residuals,
            last_kernel_size=3,
            pad="constant",
            static_data=None,
            gradient_detach_interval=0,
        )
        self.residual = torch.nn.Parameter(torch.tensor(2.0))

    def forward_once(self, fts):
        return torch.ones_like(fts[:, : self.out_channels]) * self.residual


def make_replay_dataset(stride: int = 3) -> TorchTrainDataset:
    coords = {"time": range(30), "lat": [0], "lon": [0]}
    times = torch.arange(30, dtype=torch.float32)
    data = xr.Dataset(
        {
            "prognostic1": (
                ["time", "lat", "lon"],
                (times * 10).reshape(30, 1, 1).numpy(),
            ),
            "boundary1": (
                ["time", "lat", "lon"],
                (times * 10 + 1).reshape(30, 1, 1).numpy(),
            ),
        },
        coords=coords,
    )
    means = xr.Dataset(
        {
            "prognostic1": 0.0,
            "boundary1": 0.0,
        },
        coords={"lat": [0], "lon": [0]},
    )
    stds = xr.Dataset(
        {
            "prognostic1": 1.0,
            "boundary1": 1.0,
        },
        coords={"lat": [0], "lon": [0]},
    )
    masks = Masks(
        prognostic=torch.ones(1, 1, 1, dtype=torch.bool),
        boundary=torch.ones(1, 1, dtype=torch.bool),
    )
    src = DataSource("replay-test", data, means, stds, masks=masks)
    return TorchTrainDataset(
        src=src,
        prognostic_var_names=["prognostic1"],
        boundary_var_names=["boundary1"],
        hist=0,
        steps=7,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=stride,
        temporal_stride=1,
    )


def test_replay_transition_uses_drifted_state_and_true_future_target():
    dataset = make_replay_dataset(stride=3)
    drifted_state = torch.tensor([[[999.0]]])

    input, label = dataset.get_replay_transition(
        source_index=0,
        lead_step=2,
        prognostic_state=drifted_state,
    )

    assert input.flatten().tolist() == [999.0, 61.0]
    assert label.flatten().tolist() == [90.0]


def test_replay_seeded_and_advanced_states_have_same_stored_shape():
    dataset = make_replay_dataset(stride=3)

    seeded_state = dataset.get_replay_initial_state(source_index=0)
    seeded_state = dataset.remask_prognostic_state(seeded_state)[0]
    advanced_state = dataset.remask_prognostic_state(
        torch.full_like(seeded_state, 123.0)
    )

    assert seeded_state.shape == advanced_state.shape == (1, 1, 1)

    seeded_input, _ = dataset.get_replay_transition(
        source_index=0,
        lead_step=0,
        prognostic_state=seeded_state,
    )
    advanced_input, _ = dataset.get_replay_transition(
        source_index=0,
        lead_step=1,
        prognostic_state=advanced_state,
    )

    assert seeded_input[:, : dataset.num_prognostic_channels].shape == (1, 1, 1, 1)
    assert advanced_input[:, : dataset.num_prognostic_channels].shape == (1, 1, 1, 1)


def test_replay_transition_advances_forcing_and_target_with_cursor():
    dataset = make_replay_dataset(stride=3)
    source_index = 2
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=source_index,
        lead_step=0,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )

    for expected_lead_step in [1, 2, 3]:
        cursor = cursor.advance()
        current_time_index = (
            source_index * dataset.temporal_stride
            + expected_lead_step * dataset.stride
        )
        target_time_index = current_time_index + dataset.stride
        drifted_state = torch.tensor(
            [[[1000.0 + expected_lead_step]]],
            dtype=torch.float32,
        )

        input, label = dataset.get_replay_transition(
            source_index=cursor.source_index,
            lead_step=cursor.lead_step,
            prognostic_state=drifted_state,
        )

        assert cursor.lead_step == expected_lead_step
        assert input.flatten().tolist() == [
            1000.0 + expected_lead_step,
            current_time_index * 10.0 + 1.0,
        ]
        assert label.flatten().tolist() == [target_time_index * 10.0]


def test_predict_step_residual_uses_actual_replay_input_state():
    drifted_input = torch.tensor([[[[999.0]], [[61.0]]]])
    true_next = torch.tensor([[[[90.0]]]])

    residual_model = ConstantResidualModel(pred_residuals=True)
    residual_pred = residual_model.predict_step(drifted_input)
    residual_loss = torch.nn.functional.mse_loss(
        residual_pred,
        true_next,
        reduction="none",
    )

    assert residual_pred.item() == 1001.0
    assert residual_loss.item() == (1001.0 - 90.0) ** 2

    absolute_model = ConstantResidualModel(pred_residuals=False)
    absolute_pred = absolute_model.predict_step(drifted_input)

    assert absolute_pred.item() == 2.0


def test_replay_sidecar_world_size_mismatch_reseeds(monkeypatch):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    buffer = ReplayBuffer(
        buffer_size=1,
        storage_dtype=torch.float32,
        generator=generator,
        pin_memory=False,
    )
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=0,
        stride=1,
        temporal_stride=1,
    )
    buffer.append(ReplayEntry(state=torch.zeros(1, 1, 1), cursor=cursor))

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "ckpt.pt"
        sidecar_path = replay_sidecar_path(checkpoint_path, rank=0)
        state = buffer.state_dict(world_size=4, rank=0)
        state["dataset_signature"] = []
        torch.save(state, sidecar_path)

        trainer = Trainer.__new__(Trainer)
        trainer.replay_buffer = ReplayBuffer(
            buffer_size=1,
            storage_dtype=torch.float32,
            generator=torch.Generator(device="cpu"),
            pin_memory=False,
        )
        trainer.train_datasets = []
        trainer.replay_dataset_signature = lambda: []

        monkeypatch.setattr("ocean_emulators.train.get_world_size", lambda: 1)

        assert not trainer.load_replay_buffer_sidecar(checkpoint_path)
