# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.diffusion_native_controls import velocity_statistics


def test_velocity_calibration_is_physical_and_preserves_member_structure():
    # Opposite spatial patterns have a smooth ensemble mean. Nonzero training
    # mean checks that the zero reference is physical zero, not standardized 0.
    physical = torch.tensor([[[[[0.0, 2.0]]]], [[[[2.0, 0.0]]]]])
    truth = torch.ones(1, 1, 1, 2)
    mean, std = torch.tensor([3.0]), torch.tensor([2.0])
    result = velocity_statistics(
        (physical - 3) / 2,
        (truth - 3) / 2,
        mean,
        std,
        torch.ones(1, 1, 2),
        torch.ones(1, 1, 2),
    )
    assert result["ensemble"]["mean_squared_error"].item() == 0
    assert result["ensemble"]["fair_crps"].item() == 0
    assert result["ensemble"]["ensemble_variance"].item() == 4
    assert result["zero_velocity"]["absolute_error"].item() == 2
    assert result["zero_velocity"]["mean_squared_error"].item() == 2
    assert result["mean_structure"]["variance"].item() == 0
    torch.testing.assert_close(
        result["members_structure"]["variance"], torch.ones(2, 1, 1)
    )
    torch.testing.assert_close(
        result["ensemble"]["rank_weights"].sum(0), result["ensemble"]["weight"]
    )


def test_native_summary_requires_verified_complete_cohort(tmp_path):
    import json

    import pytest

    from samudra.experiments.diffusion_native_summary import summarize
    from samudra.experiments.diffusion_report import json_statistics
    from samudra.experiments.observation_pilot import digest

    members = torch.tensor([[[[[0.0, 2.0]]]], [[[[2.0, 0.0]]]]])
    stats = json_statistics(
        velocity_statistics(
            members,
            torch.ones(1, 1, 1, 2),
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1, 1, 2),
            torch.ones(1, 1, 2),
        )
    )
    origins = [str(i) for i in range(24)]
    protocol = dict(origins=origins)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "metrics.csv").write_text("header\n")
    hashes = {}
    for origin in origins:
        path = tmp_path / f"velocity-{origin}.json"
        path.write_text(
            json.dumps(
                dict(
                    origin=origin,
                    channels=["uo_0"],
                    units="m/s",
                    records=[
                        dict(region=region, lead_days=lead, statistics=stats)
                        for region in (
                            "global",
                            "scored_latitudes",
                            "outside_scored_latitudes",
                        )
                        for lead in (0, 5, 15, 30)
                    ],
                )
            )
        )
        hashes[path.name] = digest(path)
    (tmp_path / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                **protocol,
                velocity_sha256=hashes,
                metrics_sha256=digest(tmp_path / "metrics.csv"),
            )
        )
    )
    result = summarize(tmp_path)
    assert len(result["results"]) == 12
    assert result["results"][0]["ensemble"]["scores"]["mean_squared_error"] == 0
    assert result["results"][0]["zero_velocity"]["scores"]["absolute_error"] == 1
    (tmp_path / "velocity-0.json").write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        summarize(tmp_path)
