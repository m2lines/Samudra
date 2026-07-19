import queue
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import xarray as xr

from ocean_emulators.datasets import (
    ReplayBatchRequest,
    ReplayBatchSlot,
    ReplayCursor,
    ReplayRequestDataset,
    ReplaySeedSlot,
    TorchTrainDataset,
)
from ocean_emulators.models.base import BaseModel
from ocean_emulators.replay import ReplayBuffer, ReplayEntry, replay_sidecar_path
from ocean_emulators.train import Trainer, _ReplayPrefetchPipeline
from ocean_emulators.utils.data import DataSource, Masks


class _IdentityDomainContext:
    is_domain_leader = True

    def scatter_spatial(self, tensor, *, ndim):
        assert ndim == 4
        assert tensor is not None
        return tensor.clone()

    @staticmethod
    def from_local_spatial(tensor):
        return tensor

    @staticmethod
    def local_tensor(tensor):
        return tensor

    @staticmethod
    def broadcast_from_leader(value):
        return value


class _FollowerDomainContext(_IdentityDomainContext):
    is_domain_leader = False


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


def test_raw_replay_request_dataset_preserves_cursor_alignment():
    dataset = make_replay_dataset(stride=3)
    train_cursor = ReplayCursor(
        dataset_index=0,
        source_index=2,
        lead_step=3,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    seed_cursor = ReplayCursor(
        dataset_index=0,
        source_index=4,
        lead_step=0,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    request = ReplayBatchRequest(
        request_id=7,
        train_slots=(ReplayBatchSlot(replay_index=0, cursor=train_cursor),),
        seed_slots=(
            ReplaySeedSlot(
                replay_index=1,
                cursor=seed_cursor,
                reason="scheduled",
            ),
        ),
    )
    request_queue = queue.Queue()
    request_queue.put(request)
    request_queue.put(None)

    raw_batch = next(iter(ReplayRequestDataset([dataset], request_queue)))

    assert raw_batch.request.request_id == 7
    assert raw_batch.train_transitions[0].current_time_index == 11
    assert raw_batch.train_transitions[0].target_time_index == 14
    assert raw_batch.seed_transitions[0].current_time_index == 4
    assert raw_batch.seed_transitions[0].target_time_index == 7


def test_prepare_raw_replay_batch_uses_drifted_buffer_state():
    dataset = make_replay_dataset(stride=3)
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=2,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    request = ReplayBatchRequest(
        request_id=1,
        train_slots=(ReplayBatchSlot(replay_index=0, cursor=cursor),),
        seed_slots=(),
    )
    raw_transition = dataset.get_raw_replay_transition(
        dataset_index=0,
        source_index=cursor.source_index,
        lead_step=cursor.lead_step,
    )
    raw_batch = type(
        "RawBatch",
        (),
        {
            "request": request,
            "train_transitions": [raw_transition],
            "seed_transitions": [],
            "load_stats": None,
        },
    )()

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.num_out = dataset.num_prognostic_channels
    trainer.train_datasets = [dataset]
    trainer.replay_buffer = ReplayBuffer(
        buffer_size=1,
        storage_dtype=torch.float32,
        generator=torch.Generator(device="cpu"),
        pin_memory=False,
    )
    trainer.replay_buffer.append(
        ReplayEntry(state=torch.tensor([[[999.0]]]), cursor=cursor)
    )

    prepared = trainer.prepare_raw_replay_batch(raw_batch, ready_event=None)
    input, label = prepared.data[0]

    assert input.flatten().tolist() == [999.0, 61.0]
    assert label.flatten().tolist() == [90.0]


def test_prepare_raw_replay_batch_keeps_loss_label_float32_with_bf16_transport():
    dataset = make_replay_dataset(stride=3)
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=2,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    request = ReplayBatchRequest(
        request_id=1,
        train_slots=(ReplayBatchSlot(replay_index=0, cursor=cursor),),
        seed_slots=(),
    )
    raw_transition = dataset.get_raw_replay_transition(
        dataset_index=0,
        source_index=cursor.source_index,
        lead_step=cursor.lead_step,
    )
    raw_transition.target_prognostic = raw_transition.target_prognostic.to(
        dtype=torch.bfloat16
    )
    raw_transition.boundary = raw_transition.boundary.to(dtype=torch.bfloat16)
    raw_batch = type(
        "RawBatch",
        (),
        {
            "request": request,
            "train_transitions": [raw_transition],
            "seed_transitions": [],
            "load_stats": None,
        },
    )()

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.num_out = dataset.num_prognostic_channels
    trainer.train_datasets = [dataset]
    trainer.replay_buffer = ReplayBuffer(
        buffer_size=1,
        storage_dtype=torch.bfloat16,
        generator=torch.Generator(device="cpu"),
        pin_memory=False,
    )
    trainer.replay_buffer.append(
        ReplayEntry(state=torch.tensor([[[999.0]]], dtype=torch.bfloat16), cursor=cursor)
    )

    prepared = trainer.prepare_raw_replay_batch(raw_batch, ready_event=None)
    input, label = prepared.data[0]
    pred = torch.zeros_like(label, requires_grad=True)
    loss = torch.nn.functional.mse_loss(pred, label)
    loss.backward()

    assert input.dtype == torch.bfloat16
    assert label.dtype == torch.float32
    assert pred.grad is not None


def test_prepare_domain_replay_batch_combines_local_state_with_scattered_gold():
    dataset = make_replay_dataset(stride=3)
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=2,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    seed_cursor = ReplayCursor(
        dataset_index=0,
        source_index=4,
        lead_step=0,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    request = ReplayBatchRequest(
        request_id=1,
        train_slots=(ReplayBatchSlot(replay_index=0, cursor=cursor),),
        seed_slots=(
            ReplaySeedSlot(
                replay_index=1,
                cursor=seed_cursor,
                reason="scheduled",
            ),
        ),
    )
    raw_batch = SimpleNamespace(
        request=request,
        train_transitions=[
            dataset.get_raw_replay_train_transition(
                dataset_index=0,
                source_index=cursor.source_index,
                lead_step=cursor.lead_step,
            )
        ],
        seed_transitions=[
            dataset.get_raw_replay_seed_transition(
                dataset_index=0,
                source_index=seed_cursor.source_index,
                lead_step=seed_cursor.lead_step,
            )
        ],
        load_stats=None,
    )

    trainer = Trainer.__new__(Trainer)
    trainer.dp_ctx = _IdentityDomainContext()
    trainer.device = torch.device("cpu")
    trainer.num_out = dataset.num_prognostic_channels
    trainer.train_datasets = [dataset]
    trainer.pin_mem = False
    trainer.replay_storage_dtype = torch.float32
    trainer.replay_buffer = ReplayBuffer(
        buffer_size=2,
        storage_dtype=torch.float32,
        generator=torch.Generator(device="cpu"),
        pin_memory=False,
    )
    trainer.replay_buffer.append(
        ReplayEntry(state=torch.tensor([[[999.0]]]), cursor=cursor)
    )

    prepared = trainer.prepare_raw_replay_batch(raw_batch, ready_event=None)
    input, label = prepared.data[0]

    assert input.flatten().tolist() == [999.0, 61.0]
    assert label.flatten().tolist() == [90.0]
    assert prepared.seed_entries[1].state.flatten().tolist() == [40.0]


def test_domain_replay_update_stores_only_local_prediction_tile():
    dataset = make_replay_dataset(stride=3)
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=2,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    request = ReplayBatchRequest(
        request_id=1,
        train_slots=(ReplayBatchSlot(replay_index=0, cursor=cursor),),
        seed_slots=(),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.dp_ctx = _IdentityDomainContext()
    trainer.device = torch.device("cpu")
    trainer.domain_wet = torch.ones(1, 1, 1, dtype=torch.bool)
    trainer.train_datasets = [dataset]
    trainer.pin_mem = False
    trainer.replay_storage_dtype = torch.float32
    trainer.replay_buffer = ReplayBuffer(
        buffer_size=1,
        storage_dtype=torch.float32,
        generator=torch.Generator(device="cpu"),
        pin_memory=False,
    )
    trainer.replay_buffer.append(
        ReplayEntry(state=torch.tensor([[[999.0]]]), cursor=cursor)
    )
    prepared = SimpleNamespace(
        request=request,
        seed_entries={},
    )

    trainer.apply_replay_prefetch_updates(
        prepared,
        pred=torch.tensor([[[[123.0]]]]),
    )

    entry = trainer.replay_buffer.entries[0]
    assert entry.state.shape == (1, 1, 1)
    assert entry.state.item() == 123.0
    assert entry.cursor == cursor.advance()


def test_prepare_raw_replay_batch_supports_seed_only_requests():
    dataset = make_replay_dataset(stride=3)
    seed_cursor = ReplayCursor(
        dataset_index=0,
        source_index=4,
        lead_step=0,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )
    request = ReplayBatchRequest(
        request_id=1,
        train_slots=(),
        seed_slots=(
            ReplaySeedSlot(
                replay_index=5,
                cursor=seed_cursor,
                reason="seed",
            ),
        ),
    )
    raw_transition = dataset.get_raw_replay_seed_transition(
        dataset_index=0,
        source_index=seed_cursor.source_index,
        lead_step=seed_cursor.lead_step,
    )
    raw_batch = type(
        "RawBatch",
        (),
        {
            "request": request,
            "train_transitions": [],
            "seed_transitions": [raw_transition],
            "load_stats": None,
        },
    )()

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.num_out = dataset.num_prognostic_channels
    trainer.train_datasets = [dataset]
    trainer.replay_buffer = ReplayBuffer(
        buffer_size=1,
        storage_dtype=torch.float32,
        generator=torch.Generator(device="cpu"),
        pin_memory=False,
    )

    prepared = trainer.prepare_raw_replay_batch(raw_batch, ready_event=None)

    assert len(prepared.data) == 0
    assert prepared.seed_entries[5].state.flatten().tolist() == [40.0]


def test_replay_prefetch_pipeline_yields_prepared_batch():
    dataset = make_replay_dataset(stride=3)
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=0,
        stride=dataset.stride,
        temporal_stride=dataset.temporal_stride,
    )

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.num_out = dataset.num_prognostic_channels
    trainer.train_datasets = [dataset]
    trainer.batch_size = 1
    trainer.num_workers = 0
    trainer.prefetch_factor = 2
    trainer.pin_mem = False
    trainer.mp_context = None
    trainer.train_loader = SimpleNamespace(
        _dataloader=SimpleNamespace(
            num_workers=0,
            pin_memory=False,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
        )
    )
    trainer.replay_copy_stream = None
    trainer.replay_storage_dtype = torch.float32
    trainer._replay_resume_planner_state = None
    trainer._replay_active_planner_state = None
    trainer.replay_cfg = SimpleNamespace(
        buffer_size=1,
        refresh_every_n_microbatches=99,
    )
    trainer.replay_generator = torch.Generator(device="cpu")
    trainer.replay_generator.manual_seed(1)
    trainer.replay_buffer = ReplayBuffer(
        buffer_size=1,
        storage_dtype=torch.float32,
        generator=trainer.replay_generator,
        pin_memory=False,
    )
    trainer.replay_buffer.append(
        ReplayEntry(state=torch.tensor([[[999.0]]]), cursor=cursor)
    )

    pipeline = _ReplayPrefetchPipeline(
        trainer,
        start_batch_in_epoch=0,
        total_batches=1,
        max_lead_steps=3,
        refresh_every_n_microbatches=99,
    )
    iterator = iter(pipeline)
    prepared = next(iterator)
    input, label = prepared.data[0]
    pipeline.complete(prepared)
    pipeline.close()

    assert input.flatten().tolist() == [999.0, 1.0]
    assert label.flatten().tolist() == [30.0]


def test_domain_follower_prefetch_uses_metadata_without_reader_threads():
    cursor = ReplayCursor(
        dataset_index=0,
        source_index=0,
        lead_step=0,
        stride=1,
        temporal_stride=1,
    )
    request = ReplayBatchRequest(
        request_id=0,
        train_slots=(ReplayBatchSlot(replay_index=0, cursor=cursor),),
        seed_slots=(),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.dp_ctx = _FollowerDomainContext()
    trainer.num_workers = 6
    trainer._replay_active_planner = None
    trainer._replay_active_planner_state = None
    trainer.consume_replay_planner_resume_state = lambda: None
    trainer.replay_prefetch_horizon = lambda: 1
    trainer.plan_replay_batch = lambda **kwargs: (
        request if kwargs["leader_can_plan"] else None
    )

    pipeline = _ReplayPrefetchPipeline(
        trainer,
        start_batch_in_epoch=0,
        total_batches=1,
        max_lead_steps=1,
        refresh_every_n_microbatches=99,
    )

    assert pipeline._worker_threads == []
    assert pipeline._raw_cache[0].request == request
    assert pipeline._raw_cache[0].train_transitions == []
    pipeline.close()


@pytest.mark.parametrize("is_leader", [True, False])
def test_domain_replay_planning_broadcasts_blocked_window(is_leader):
    message = "No unreserved replay slots are available"

    class BlockedContext:
        is_domain_leader = is_leader

        def broadcast_from_leader(self, value):
            if self.is_domain_leader:
                assert value == {"status": "blocked", "message": message}
                return value
            assert value is None
            return {"status": "blocked", "message": message}

    trainer = Trainer.__new__(Trainer)
    trainer.dp_ctx = BlockedContext()

    def blocked_plan(**_kwargs):
        raise RuntimeError(message)

    trainer._plan_replay_batch_local = blocked_plan

    with pytest.raises(RuntimeError, match=message):
        trainer.plan_replay_batch(
            global_batch_index=0,
            max_lead_steps=4,
            refresh_every_n_microbatches=8,
            exclude_reserved=set(),
        )


@pytest.mark.parametrize("is_leader", [True, False])
def test_domain_replay_planning_leader_closes_fill_window(is_leader):
    class CompleteContext:
        is_domain_leader = is_leader

        def broadcast_from_leader(self, value):
            if self.is_domain_leader:
                assert value == {"status": "complete"}
                return value
            assert value is None
            return {"status": "complete"}

    trainer = Trainer.__new__(Trainer)
    trainer.dp_ctx = CompleteContext()

    def unexpected_plan(**_kwargs):
        raise AssertionError("The leader must not plan beyond its window")

    trainer._plan_replay_batch_local = unexpected_plan

    assert trainer.plan_replay_batch(
        global_batch_index=31,
        max_lead_steps=4,
        refresh_every_n_microbatches=8,
        exclude_reserved=set(),
        leader_can_plan=False,
    ) is None


def test_replay_refresh_schedule_resolves_by_epoch():
    trainer = Trainer.__new__(Trainer)
    trainer.start_epoch = 1
    trainer.replay_cfg = SimpleNamespace(
        refresh_every_n_microbatches=[4, 8, 16],
        refresh_every_n_microbatches_transition=[3, 5],
    )

    assert trainer.get_current_replay_refresh_every_n_microbatches(1) == 4
    assert trainer.get_current_replay_refresh_every_n_microbatches(2) == 4
    assert trainer.get_current_replay_refresh_every_n_microbatches(3) == 8
    assert trainer.get_current_replay_refresh_every_n_microbatches(4) == 8
    assert trainer.get_current_replay_refresh_every_n_microbatches(5) == 16


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


def test_replay_buffer_sampling_excludes_reserved_indices():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    buffer = ReplayBuffer(
        buffer_size=4,
        storage_dtype=torch.float32,
        generator=generator,
        pin_memory=False,
    )
    for index in range(4):
        cursor = ReplayCursor(
            dataset_index=0,
            source_index=index,
            lead_step=0,
            stride=1,
            temporal_stride=1,
        )
        buffer.append(ReplayEntry(state=torch.zeros(1, 1, 1), cursor=cursor))

    sampled = buffer.sample_indices(
        batch_size=8,
        max_lead_steps=1,
        exclude_reserved={1, 2, 3},
    )
    refreshed = buffer.random_indices(count=8, exclude_reserved={0, 1, 2})

    assert set(sampled) == {0}
    assert set(refreshed) == {3}
