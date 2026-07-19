import pytest
import torch

from ocean_emulators.datasets import TrainData
from ocean_emulators.shardtensor import DomainParallelConfig, validate_shardable
from ocean_emulators.train import Trainer, _DomainFollowerLoader


class _FakeLeaderContext:
    is_domain_leader = True

    def scatter_spatial(self, tensor, *, ndim):
        assert ndim == 4
        assert tensor is not None
        return tensor.clone()


def test_domain_parallel_config_validates_cluster_shape():
    config = DomainParallelConfig(cluster_shape=(2, 2))
    assert config.cluster_size == 4
    assert config.model_dump()["cluster_shape"] == [2, 2]
    with pytest.raises(ValueError, match="two positive integers"):
        DomainParallelConfig(cluster_shape=(0, 2))


def test_shardable_validation_rejects_deep_tiles_smaller_than_halo():
    with pytest.raises(ValueError, match="maximum convolution halo"):
        validate_shardable(128, 128, (2, 2), num_downsamples=4, max_halo=8)
    validate_shardable(320, 320, (2, 2), num_downsamples=4, max_halo=8)


def test_shardable_validation_requires_each_shard_to_divide_through_unet():
    validate_shardable(704, 704, (2, 2), num_downsamples=4, max_halo=8)
    with pytest.raises(ValueError, match=r"Per-shard tile \(360x360\)"):
        validate_shardable(720, 720, (2, 2), num_downsamples=4, max_halo=8)


def test_shardable_validation_rejects_non_divisible_global_unet_shape():
    with pytest.raises(ValueError, match=r"Per-shard tile \(722x722\)"):
        validate_shardable(722, 722, (1, 1), num_downsamples=4)


def test_domain_follower_loader_matches_leader_batch_count():
    loader = _DomainFollowerLoader(num_batches=3, num_prognostic_channels=5)
    batches = list(loader)
    assert len(loader) == 3
    assert len(batches) == 3
    assert all(batch.num_prognostic_channels == 5 for batch in batches)
    assert all(len(batch) == 0 for batch in batches)
    assert len(loader.with_offset(2)) == 1


def test_scatter_domain_batch_preserves_all_curriculum_steps():
    trainer = Trainer.__new__(Trainer)
    trainer.dp_ctx = _FakeLeaderContext()
    trainer.num_out = 2

    dense = TrainData(num_prognostic_channels=2)
    for step in range(3):
        dense.append(
            torch.full((1, 4, 32, 32), float(step)),
            torch.full((1, 2, 32, 32), float(step + 10)),
        )

    sharded = trainer._scatter_domain_batch(dense, expected_steps=3)

    assert len(sharded) == 3
    for step in range(3):
        assert torch.equal(sharded.get_input(step), dense.get_input(step))
        assert torch.equal(sharded.get_label(step), dense.get_label(step))


def test_merge_prognostic_and_boundary_uses_dispatchable_cat():
    data = TrainData(num_prognostic_channels=2)
    original = torch.randn(1, 5, 8, 8)
    data.append(original, torch.randn(1, 2, 8, 8))
    prognostic = torch.randn(1, 2, 8, 8)

    merged = data.merge_prognostic_and_boundary(prognostic, step=0)

    assert torch.equal(merged[:, :2], prognostic)
    assert torch.equal(merged[:, 2:], original[:, 2:])
    assert torch.equal(data.get_input(0), original)


def test_wandb_model_watch_is_skipped_for_domain_parallel_parameters():
    class FakeWandbLogger:
        enabled = False

        def __init__(self):
            self.calls = []

        def watch(self, model, **kwargs):
            self.calls.append((model, kwargs))

    trainer = Trainer.__new__(Trainer)
    trainer.model = object()
    trainer.wandb_logger = FakeWandbLogger()
    trainer.dp_ctx = object()

    trainer._configure_wandb_model_watch()
    assert trainer.wandb_logger.calls == []

    trainer.dp_ctx = None
    trainer._configure_wandb_model_watch()
    assert trainer.wandb_logger.calls == [(trainer.model, {"log": "all"})]
