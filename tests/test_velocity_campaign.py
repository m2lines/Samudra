# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from scripts import velocity_campaign


@pytest.fixture
def controller():
    return vars(velocity_campaign)


def test_allocation_accounting_counts_requeues_and_requires_gpu_data(
    controller, monkeypatch
):
    def run(*a, **kw):
        return SimpleNamespace(
            stdout="1|COMPLETED|3600|cpu=32,gres/gpu=4|\n1|PREEMPTED|1800|gres/gpu=4|\n2|COMPLETED|900|gres/gpu=2|\n"
        )

    monkeypatch.setattr(controller["subprocess"], "run", run)
    assert controller["allocation_hours"](["1", "2"]) == pytest.approx(6.5)
    with pytest.raises(RuntimeError, match="incomplete"):
        controller["allocation_hours"](["1", "2", "3"])


def test_screen_waits_for_pilots_and_limits_parallel_training_to_two_lanes(
    controller, tmp_path
):
    for variant in ("D0", "D3"):
        path = tmp_path / ("pilot-" + variant + "-s15")
        path.mkdir()
        (path / "best.pt").touch()
        rows = [
            {"peak_rss_gib_rank0": 2, "peak_cuda_gib_rank0": 5},
            {"validation/vector_rmse_10d": 0.1},
            {"complete": True},
        ]
        (path / "history.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    class FakeCampaign(velocity_campaign.Campaign):
        def __init__(self):
            self.root = tmp_path
            self.manifest = {"budget_gpu_hours": 1344}
            self.launched = []
            self.advanced = None

        def save(self):
            pass

        def spent(self):
            return 2

        def launch_run(self, stage, variant, seed, gpu_hours, predecessor=None):
            self.launched.append((stage, variant, seed, gpu_hours, predecessor))
            return str(len(self.launched))

        def next_stage(self, stage, dependencies):
            self.advanced = (stage, dependencies)

    campaign = FakeCampaign()
    campaign.screen()
    assert [row[-1] for row in campaign.launched] == [None, None, "1", "2", "3", "4"]
    assert sum(row[3] for row in campaign.launched) == 288
    assert campaign.advanced == ("confirm", ["1", "2", "3", "4", "5", "6"])
    (tmp_path / "pilot-D3-s15/history.jsonl").write_text('{"complete":false}\n')
    with pytest.raises(RuntimeError, match="did not complete"):
        FakeCampaign().screen()
