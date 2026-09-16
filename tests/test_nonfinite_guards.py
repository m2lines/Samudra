"""Guards that keep a diverged run recoverable.

Wide-model replay runs intermittently blow up: the replay feedback loop drives
the training loss up exponentially until it overflows float32. Two unguarded
spots used to turn that transient excursion into permanent, silent corruption --
these tests pin both guards down.
"""

import logging
from types import SimpleNamespace

import torch

from ocean_emulators.train import Trainer


def test_clip_grad_norm_spreads_a_single_nonfinite_gradient():
    """The behaviour the optimizer-step guard exists to defend against.

    `clip_grad_norm_` scales *every* gradient by `max_norm / (total_norm + eps)`.
    One non-finite gradient makes `total_norm` non-finite, so that factor is NaN
    and the scaling smears NaN across all parameters. Stepping on that wipes the
    model and every Adam moment at once.
    """
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 3, padding=1),
        torch.nn.Conv2d(8, 4, 3, padding=1),
    )
    model(torch.randn(1, 4, 8, 8)).sum().backward()

    params = list(model.parameters())
    params[0].grad[0, 0, 0, 0] = float("nan")
    assert sum(not torch.isfinite(p.grad).all() for p in params) == 1

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    assert not torch.isfinite(grad_norm)
    # Every parameter is now poisoned, which is why the guard must skip the step.
    assert all(not torch.isfinite(p.grad).all() for p in params)


def test_guarded_step_leaves_weights_and_optimizer_state_finite():
    """Skipping the step on a non-finite grad norm keeps the run recoverable."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 3, padding=1),
        torch.nn.Conv2d(8, 4, 3, padding=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # A healthy step first, so Adam has state to lose.
    model(torch.randn(1, 4, 8, 8)).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    model(torch.randn(1, 4, 8, 8)).sum().backward()
    list(model.parameters())[0].grad[0, 0, 0, 0] = float("inf")

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(grad_norm):  # pragma: no cover - guard under test
        optimizer.step()
    optimizer.zero_grad()

    assert all(torch.isfinite(p).all() for p in model.parameters())
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                assert torch.isfinite(value).all()


def test_save_checkpoint_refuses_nonfinite_weights(tmp_path, caplog):
    """A NaN checkpoint is how divergence survives a resume -- never write one."""
    target = tmp_path / "ckpt_emergency.pt"
    target.write_bytes(b"previous-checkpoint")

    poisoned = {
        "conv.weight": torch.full((2, 2), float("nan")),
        "conv.bias": torch.zeros(2),
    }
    trainer = SimpleNamespace(
        _model_state_dict_for_save=lambda: poisoned,
        _ema_context=None,
    )

    with caplog.at_level(logging.ERROR):
        result = Trainer.save_checkpoint(trainer, epoch=3, checkpoint_path=target)

    assert result is None
    # The good checkpoint already on disk is left untouched.
    assert target.read_bytes() == b"previous-checkpoint"
    assert "non-finite" in caplog.text


def test_save_checkpoint_still_writes_finite_weights(tmp_path):
    """The guard must not block the normal path."""
    target = tmp_path / "ckpt.pt"
    healthy = {"conv.weight": torch.ones(2, 2), "conv.bias": torch.zeros(2)}
    trainer = SimpleNamespace(
        _model_state_dict_for_save=lambda: healthy,
        _ema_context=None,
        optimizer=SimpleNamespace(state_dict=lambda: {}),
        _ema=SimpleNamespace(get_state=lambda include_ema_params: {}),
        dp_ctx=None,
        best_val_loss=0.5,
        best_inf_loss=0.5,
        num_batches_seen=10,
        wandb_id=None,
        wandb_name=None,
        loss_fn=SimpleNamespace(),
        scheduler=None,
    )

    Trainer.save_checkpoint(trainer, epoch=3, checkpoint_path=target)

    written = torch.load(target, map_location="cpu", weights_only=False)
    assert torch.isfinite(written["model"]["conv.weight"]).all()
    assert written["epoch"] == 3


def _trainer_with_sigma(max_state_sigma):
    return SimpleNamespace(
        replay_cfg=SimpleNamespace(max_state_sigma=max_state_sigma)
    )


def test_replay_state_diverged_always_rejects_nonfinite():
    """Non-finite states are rejected even with the range check disabled."""
    trainer = _trainer_with_sigma(0.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        state = torch.ones(2, 2)
        state[0, 0] = bad
        assert Trainer._replay_state_diverged(trainer, state)


def test_replay_state_diverged_range_check_is_opt_in():
    """max_state_sigma=0 keeps the established write-back behaviour."""
    trainer = _trainer_with_sigma(0.0)
    assert not Trainer._replay_state_diverged(trainer, torch.full((2, 2), 1e6))


def test_replay_state_diverged_rejects_runaway_magnitudes():
    trainer = _trainer_with_sigma(50.0)
    assert not Trainer._replay_state_diverged(trainer, torch.full((2, 2), 3.0))
    assert not Trainer._replay_state_diverged(trainer, torch.full((2, 2), -49.0))
    assert Trainer._replay_state_diverged(trainer, torch.full((2, 2), 51.0))
    assert Trainer._replay_state_diverged(trainer, torch.full((2, 2), -1e4))


def test_replay_state_diverged_catches_a_single_bad_cell():
    """One runaway grid point is enough to poison the slot."""
    trainer = _trainer_with_sigma(50.0)
    state = torch.zeros(4, 4)
    state[2, 3] = 1e3
    assert Trainer._replay_state_diverged(trainer, state)
