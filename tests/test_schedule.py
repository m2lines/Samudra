import torch

from ocean_emulators.utils.schedule import CosineWithTailSchedulerConfig


def test_cosine_with_tail_holds_constant_lr():
    """Test that LR is held constant for tail_epochs after cosine schedule completes.

    The scheduler should:
    1. Run cosine annealing from initial_lr to tail_lr for (total_epochs - tail_epochs)
    2. Hold LR constant at tail_lr for the final tail_epochs
    """
    initial_lr = 0.01
    tail_lr = 0.001
    tail_epochs = 10
    total_epochs = 50

    optimizer = torch.optim.SGD([torch.zeros(1)], lr=initial_lr)

    scheduler_config = CosineWithTailSchedulerConfig(
        tail_lr=tail_lr, tail_epochs=tail_epochs
    )
    scheduler = scheduler_config.build(optimizer, epochs=total_epochs)

    lr_history = []
    # Need to call step after optimizer.step() to avoid warnings
    # For testing, we'll simulate optimizer steps
    for epoch in range(total_epochs):
        lr_history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()  # Simulate optimizer step
        scheduler.step()

    cosine_epochs = total_epochs - tail_epochs

    tail_lrs = lr_history[cosine_epochs:]
    # The last 10 epochs should all have the same LR value
    for i, lr in enumerate(tail_lrs):
        assert abs(lr - tail_lr) < 1e-7, (
            f"LR at tail epoch {i} ({lr}) doesn't match first tail LR ({tail_lr})"
        )

    # cosine_epochs should start at initial_lr and end at tail_lr, be monotonically decreasing
    assert lr_history[0] == initial_lr, (
        f"LR at start of cosine schedule ({lr_history[0]}) doesn't match initial LR ({initial_lr})"
    )
    assert lr_history[-1] == tail_lr, (
        f"LR at end of cosine schedule ({lr_history[-1]}) doesn't match tail LR ({tail_lr})"
    )
    assert all(lr_history[i] > lr_history[i + 1] for i in range(cosine_epochs - 1)), (
        "LRs are not monotonically decreasing"
    )
