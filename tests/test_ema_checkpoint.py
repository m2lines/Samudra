"""Regression tests for how EMA weights reach a saved checkpoint.

``save_checkpoint(for_inference=True)`` swaps the EMA weights into the model
inside ``_ema_context`` and reads ``state_dict()``. Because ``state_dict()``
returns tensors aliasing live parameter storage, the ``restore()`` on context
exit used to overwrite the values before ``torch.save`` ran, so ``ema_ckpt.pt``
silently contained the raw weights.
"""

import torch
from torch import nn

from ocean_emulators.train import Trainer
from ocean_emulators.utils.ema import EMATracker


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("count", torch.tensor(3.0))

    def forward(self, x):
        return self.lin(x)


def _diverged_tracker(model):
    """An EMA tracker whose shadow weights differ from the live weights."""
    ema = EMATracker(model, decay=0.5, faster_decay_at_start=False)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema(model=model)  # shadow moves halfway, so it matches neither old nor new
    return ema


def test_clone_state_dict_survives_ema_restore():
    model = _TinyModel()
    ema = _diverged_tracker(model)
    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Mirror save_checkpoint(for_inference=True).
    ema.store(parameters=model.parameters())
    ema.copy_to(model=model)
    try:
        saved = Trainer._clone_state_dict(model.state_dict())
    finally:
        ema.restore(parameters=model.parameters())

    assert not torch.allclose(saved["lin.weight"], raw["lin.weight"]), (
        "saved weights match the raw weights; the EMA snapshot was clobbered "
        "by restore()"
    )
    assert torch.allclose(saved["lin.weight"], ema._ema_params["linweight"])
    # The live model is back to its raw weights, so training is unaffected.
    assert torch.allclose(model.lin.weight, raw["lin.weight"])


def test_clone_state_dict_does_not_alias_parameters():
    model = _TinyModel()
    cloned = Trainer._clone_state_dict(model.state_dict())
    with torch.no_grad():
        model.lin.weight.fill_(42.0)
    assert not torch.allclose(cloned["lin.weight"], model.lin.weight)


def test_clone_state_dict_preserves_non_parameter_entries():
    model = _TinyModel()
    state = model.state_dict()
    cloned = Trainer._clone_state_dict(state)
    assert list(cloned.keys()) == list(state.keys())
    assert torch.equal(cloned["count"], state["count"])
    if hasattr(state, "_metadata"):
        assert cloned._metadata == state._metadata
