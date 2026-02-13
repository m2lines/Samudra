import builtins

import pytest
from perceiver_pytorch import Perceiver as NaivePerceiver

from ocean_emulators.config import PerceiverConfig


def _block_flash_perceiver_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force flash_perceiver imports to fail, independent of environment state."""
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_perceiver":
            raise ModuleNotFoundError("No module named 'flash_perceiver'")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_perceiver_auto_falls_back_to_naive_when_flash_unavailable(monkeypatch):
    _block_flash_perceiver_import(monkeypatch)
    monkeypatch.setattr(
        "ocean_emulators.config.torch.cuda.is_available", lambda: True
    )

    cfg = PerceiverConfig(implementation="auto", depth=1, latent_dim=8, num_latents=16)
    with pytest.warns(UserWarning, match="Falling back to `naive`"):
        perceiver = cfg.build(in_channels=24, out_channels=2, max_patch_size=(6, 10))

    assert isinstance(perceiver, NaivePerceiver)


def test_perceiver_flash_raises_when_flash_unavailable(monkeypatch):
    _block_flash_perceiver_import(monkeypatch)
    monkeypatch.setattr(
        "ocean_emulators.config.torch.cuda.is_available", lambda: True
    )

    cfg = PerceiverConfig(implementation="flash", depth=1, latent_dim=8, num_latents=16)
    with pytest.raises(ValueError, match="implementation==flash"):
        cfg.build(in_channels=24, out_channels=2, max_patch_size=(6, 10))
