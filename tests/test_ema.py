import torch
import pytest

from ocean_emulators.utils.ema import EMATracker


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


def test_restore_state_with_weights():
    model = SimpleModel()
    ema = EMATracker(model, decay=0.5, faster_decay_at_start=False)
    # update EMA with some model updates
    for _ in range(3):
        model.linear.weight.data.add_(1.0)
        ema(model)

    state = ema.get_state()
    new_model = SimpleModel()
    restored = EMATracker.from_state(state, new_model)

    for key in ema._ema_params:
        assert torch.allclose(ema._ema_params[key], restored._ema_params[key])


def test_restore_state_without_weights_fails():
    model = SimpleModel()
    ema = EMATracker(model, decay=0.5, faster_decay_at_start=False)
    for _ in range(3):
        model.linear.weight.data.add_(1.0)
        ema(model)

    state = ema.get_state()
    state.pop("ema_params")

    new_model = SimpleModel()
    restored = EMATracker.from_state(state, new_model)

    mismatched = False
    for key in ema._ema_params:
        if not torch.allclose(ema._ema_params[key], restored._ema_params[key]):
            mismatched = True
            break

    assert mismatched


def test_remove_module_prefix_nested():
    from collections import OrderedDict

    def remove_module_prefix(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.removeprefix("module.")
            if isinstance(v, dict):
                new_state_dict[name] = {
                    key.removeprefix("module."): val for key, val in v.items()
                }
            else:
                new_state_dict[name] = v
        return new_state_dict

    state = OrderedDict(
        {
            "module.weight": torch.tensor(1.0),
            "module.ema": {"module.weight": torch.tensor(2.0)},
        }
    )

    new_state = remove_module_prefix(state)
    assert "weight" in new_state
    assert "ema" in new_state
    assert "weight" in new_state["ema"]
