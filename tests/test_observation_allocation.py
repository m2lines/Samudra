# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise allocation sequencing with lightweight subprocesses, without GPUs."""

import json
import sys

import pytest

from samudra.experiments import observation_allocation as allocation

CHILD = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index('--output') + 1])
root, name = output.parent, output.name
if name != 'fitting':
    assert (root / 'fitting/QUALIFIED.json').exists()
if name == 'scratch':
    assert (root / 'scratch-fitting/QUALIFIED.json').exists()
if name.endswith('-evaluation'):
    assert (root / name.removesuffix('-evaluation') / 'TRAIN_COMPLETE.json').exists()
if name == os.environ.get('FAIL_ARM'):
    sys.exit(4)
output.mkdir(parents=True, exist_ok=True)
marker = 'COMPLETE.json' if name.endswith('-evaluation') else (
    'QUALIFIED.json' if 'fitting' in name else 'TRAIN_COMPLETE.json'
)
(output / marker).write_text(json.dumps({'device': os.environ['CUDA_VISIBLE_DEVICES']}))
"""


@pytest.mark.parametrize("fail_adapter", [False, True])
def test_allocation_gates_devices_and_independent_failure(
    tmp_path, monkeypatch, fail_adapter
):
    original_popen = allocation.subprocess.Popen
    original_sleep = allocation.time.sleep
    launched = []

    def lightweight_process(command, **kwargs):
        launched.append(command)
        return original_popen([sys.executable, "-c", CHILD, *command[4:]], **kwargs)

    monkeypatch.setattr(allocation.subprocess, "Popen", lightweight_process)
    monkeypatch.setattr(allocation.time, "sleep", lambda _: original_sleep(0.01))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "allocation",
            "--root",
            str(tmp_path),
            "--data",
            "unused",
            "--checkpoint",
            "unused",
            "--source-contract",
            "unused",
        ],
    )
    if fail_adapter:
        monkeypatch.setenv("FAIL_ARM", "adapter-only")
        with pytest.raises(RuntimeError, match="adapter-only"):
            allocation.main()
        assert not (tmp_path / "adapter-only-evaluation/COMPLETE.json").exists()
    else:
        allocation.main()
    for path, device in [
        ("primary/TRAIN_COMPLETE.json", "GPU-a"),
        ("scratch/TRAIN_COMPLETE.json", "GPU-c"),
        ("scratch-evaluation/COMPLETE.json", "GPU-d"),
    ]:
        assert json.loads((tmp_path / path).read_text())["device"] == device
    if not fail_adapter:
        count = len(launched)
        allocation.main()
        assert len(launched) == count
