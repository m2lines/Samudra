# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Prevent a requested tracked evaluation from silently running without W&B."""

from types import SimpleNamespace

import pytest

from scripts import onedeg_tracked_eval as tracked


@pytest.mark.parametrize("mode", ["disabled", "online"])
def test_refuses_untracked_rollout(monkeypatch, tmp_path, mode):
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            wandb=SimpleNamespace(mode=mode), output_dir=tmp_path
        ),
        prepare_output_dirs=lambda: None,
        debug=False,
    )
    calls = []
    evaluator = SimpleNamespace(
        wandb_logger=SimpleNamespace(run=None), run=lambda: calls.append("rollout")
    )
    monkeypatch.setattr(tracked.EvalConfig, "from_yaml_and_cli", lambda: cfg)
    monkeypatch.setattr(tracked, "handle_logging", lambda *args: None)
    monkeypatch.setattr(tracked, "handle_warnings", lambda: None)
    monkeypatch.setattr(tracked, "Eval", lambda cfg: evaluator)
    with pytest.raises((ValueError, RuntimeError), match="W&B"):
        tracked.main()
    assert not calls
