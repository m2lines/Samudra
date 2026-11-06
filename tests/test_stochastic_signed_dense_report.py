# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reporting must preserve comparison identity and aggregate squared errors first."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "signed_dense_report",
    Path(__file__).parents[1] / "scripts/report_stochastic_signed_dense.py",
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evaluation(tmp_path):
    root = tmp_path / "evaluation"
    write(root / "manifest.json", [dict(variant=report.VARIANT, step=500)])
    write(
        root / "COMPLETE.json",
        dict(
            complete=True,
            source=report.SOURCE,
            members=8,
            weights=["ema", "raw"],
            initializer_steps=[500],
            forecast_step=500,
            checkpoints={"500": "checkpoint-500"},
        ),
    )
    for weights in ("ema", "raw"):
        folder = root / f"initializer-{weights}"
        write(
            folder / "COMPLETE.json",
            dict(
                complete=True,
                manifest_sha256=report.digest(root / "manifest.json"),
                script_sha256="same-code",
                results=[f"{report.VARIANT}-00500-{m}" for m in report.MONTHS],
            ),
        )
        for month in report.MONTHS:
            write(
                folder / f"input-{month}.json",
                dict(
                    weights=weights,
                    members=8,
                    month=month,
                    seed=904003,
                    sample_sha256=month,
                    grid_sha256="same-grid",
                ),
            )
            write(
                folder / f"{report.VARIANT}-00500-{month}.json",
                dict(
                    variant=report.VARIANT,
                    step=500,
                    weights=weights,
                    month=month,
                    checkpoint_sha256="checkpoint-500",
                    metrics={
                        key: [value] * 77
                        for key, value in dict(
                            mean_mse=1.0,
                            member_mse=2.0,
                            gold_power=4.0,
                            member_power=3.0,
                            correlation=0.25,
                            fair_crps=0.6,
                        ).items()
                    },
                ),
            )
        write(
            root / f"forecast-{weights}/COMPLETE.json",
            dict(
                step=500,
                weights=weights,
                checkpoint_sha256="checkpoint-500",
                qualification=False,
                metrics=dict(
                    members=8,
                    origins=[f"origin-{i}" for i in range(9)],
                    metrics=dict.fromkeys(report.COMPONENTS, 2.0),
                ),
                climatology=dict(metrics=dict.fromkeys(report.COMPONENTS, 4.0)),
                ratios=dict.fromkeys(report.COMPONENTS, 0.5),
                score=0.5,
            ),
        )
    return root


def mutate(path, fn):
    value = json.loads(path.read_text())
    fn(value)
    write(path, value)


def test_realistic_inventory_and_preview_gate(evaluation):
    rows, forecasts, receipt, bundles = report.collect_evaluations(
        [evaluation], preview=True
    )
    assert len(rows) == 77 * 3 * 2
    assert len(forecasts) == 10
    assert receipt["checkpoints"] == {"500": "checkpoint-500"}
    assert bundles == [(evaluation, [500])]
    with pytest.raises(ValueError, match="Missing required"):
        report.collect_evaluations([evaluation])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d.update(checkpoint_sha256="wrong"), "checkpoint"),
        (lambda d: d.update(month="2014-02"), "checkpoint"),
        (lambda d: d["metrics"]["mean_mse"].__setitem__(4, -1.0), "Negative"),
        (lambda d: d["metrics"]["correlation"].__setitem__(4, 2.0), "correlation"),
    ],
)
def test_reject_changed_identity_or_invalid_metrics(evaluation, mutation, match):
    path = evaluation / "initializer-raw" / f"{report.VARIANT}-00500-2014-01.json"
    mutate(path, mutation)
    with pytest.raises(ValueError, match=match):
        report.collect_evaluations([evaluation], preview=True)


def test_input_seed_and_support_are_paired(evaluation):
    path = evaluation / "initializer-raw/input-2014-01.json"
    mutate(path, lambda d: d.update(grid_sha256="changed-wet-mask"))
    with pytest.raises(ValueError, match="support changed"):
        report.collect_evaluations([evaluation], preview=True)


def test_forecast_blend_recomputed_not_trusted(evaluation):
    mutate(evaluation / "forecast-ema/COMPLETE.json", lambda d: d.update(score=0.1))
    with pytest.raises(AssertionError):
        report.collect_evaluations([evaluation], preview=True)


def test_root_and_power_ratio_after_aggregation():
    frame = pd.DataFrame(
        [
            dict(
                weights="ema",
                step=1000,
                channel="uo_0",
                variable="uo",
                mean_mse=mse,
                member_mse=2 * mse,
                member_power=power,
                gold_power=gold,
            )
            for mse, power, gold in [(1.0, 1.0, 1.0), (9.0, 9.0, 3.0)]
        ]
    )
    for summary in (report.initializer_summary(frame), report.family_summary(frame)):
        assert summary.iloc[0].rmse == pytest.approx(np.sqrt(5))
        assert summary.iloc[0].member_power_ratio == pytest.approx(2.5)
        assert summary.iloc[0].rmse != 2.0


def test_conditioning_changes_aggregate_as_squared_errors():
    frame = pd.DataFrame(
        [
            dict(
                source="om4",
                lead="0",
                intervention="recipient",
                variable="uo",
                mean_mse=mse,
                climatology_mse=10.0,
                fair_crps=1.0,
                climatology_mae=2.0,
                paired_member_change_rmse=rms,
                paired_mean_change_rmse=rms,
            )
            for mse, rms in [(1.0, 1.0), (9.0, 3.0)]
        ]
    )
    row = report.conditioning_summary(frame).iloc[0]
    assert row.rmse == pytest.approx(np.sqrt(5))
    assert row.paired_member_change_rmse == pytest.approx(np.sqrt(5))
    assert row.mse_skill == pytest.approx(0.5)
    assert row.crps_skill == pytest.approx(0.5)


def test_conditioning_reference_cannot_change_between_swaps():
    frame = pd.DataFrame(
        [
            dict(
                case="om4-2014-01",
                lead="0",
                channel="uo_0",
                intervention=mode,
                climatology_mse=1.0,
                climatology_mae=1.0,
                support_weight=30.0,
            )
            for mode in report.MODES
        ]
    )
    report.validate_conditioning_support(frame)
    frame.loc[2, "support_weight"] = 29.0
    with pytest.raises(ValueError, match="support changed"):
        report.validate_conditioning_support(frame)
    with pytest.raises(ValueError, match="omits a mode"):
        report.validate_conditioning_support(frame.iloc[:3])


def test_actual_cost_includes_failed_allocation_and_gpu_count(tmp_path):
    ledger = tmp_path / "ledger.json"
    rows = [
        dict(job_id="1", gpus=1, elapsed_seconds=3600, state="COMPLETED"),
        dict(job_id="2", gpus=4, elapsed_seconds=900, state="FAILED"),
    ]
    write(ledger, dict(fixed_completed_gpu_hours=6.5, allocations=rows))
    assert report.cost_summary(ledger)["actual_gpu_hours"] == 8.5
    rows[1]["state"] = "RUNNING"
    write(ledger, dict(allocations=rows))
    with pytest.raises(ValueError, match="terminal"):
        report.cost_summary(ledger)
    rows[1]["state"] = "FAILED"
    rows[1]["job_id"] = "1"
    write(ledger, dict(allocations=rows))
    with pytest.raises(ValueError, match="duplicate"):
        report.cost_summary(ledger)


def test_training_resume_deduplicates_exact_events_and_rejects_conflict(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(report, "save_figure", lambda fig, path: report.plt.close(fig))
    events = [
        dict(step=i, source="om4" if i % 2 == 0 else "observation", loss=1.0)
        for i in range(1, 5)
    ]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(x) for x in events[:3]))
    b.write_text("\n".join(json.dumps(x) for x in events[2:]))
    assert report.training_curve([a, b], tmp_path, preview=True) == dict(
        last_step=4, events=4
    )
    events[2]["loss"] = 2.0
    b.write_text("\n".join(json.dumps(x) for x in events[2:]))
    with pytest.raises(ValueError, match="Conflicting"):
        report.training_curve([a, b], tmp_path, preview=True)


def test_training_history_rejects_missing_first_reference_updates(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(dict(step=23, source="observation", loss=1.0)))
    with pytest.raises(ValueError, match="continuous"):
        report.training_curve([path], tmp_path, preview=True)


def test_final_conditioning_receipts_cover_every_channel_lead_and_swap(tmp_path):
    root = tmp_path / "conditioning"
    cases = [{"name": f"om4-{month}", "source": "om4"} for month in report.MONTHS]
    cases += [{"name": f"obs-{i}", "source": "observation"} for i in range(9)]
    prepared = tmp_path / "PREPARED.json"
    write(
        prepared,
        dict(
            schema="stochastic-conditioning-v2",
            grid_sha256="grid",
            script_sha256="diagnostic",
            cases=cases,
        ),
    )
    rows: list[dict[str, object]] = []
    perturb: list[dict[str, object]] = []
    for case in cases:
        keys = [
            (str(lead), channel)
            for lead in (0, 5, 15, 30)
            for channel in (
                report.NAMES if case["source"] == "om4" else ["thetao_0", "zos"]
            )
        ]
        if case["source"] == "observation":
            keys += [
                ("calendar-month", report.NAMES[i])
                for i in [*range(38, 52), *range(57, 71)]
            ]
            keys += [("calendar-month-OHC", s) for s in ("ohc_0_700", "ohc_700_2000")]
        for mode in report.MODES:
            rows.extend(
                dict(
                    case=case["name"],
                    source=case["source"],
                    intervention=mode,
                    lead=lead,
                    channel=channel,
                    climatology_mse=1.0,
                    climatology_mae=1.0,
                    support_weight=30.0,
                )
                for lead, channel in keys
            )
            perturb.extend(
                dict(
                    case=case["name"],
                    intervention=mode,
                    history_slot=slot,
                    channel=channel,
                    paired_member_change_rmse=0.0,
                    paired_mean_change_rmse=0.0,
                )
                for slot in range(2)
                for channel in report.NAMES
            )
    complete = dict(
        complete=True,
        checkpoint_sha256="final",
        prepared_sha256=report.digest(prepared),
        contract=dict(producer=report.SOURCE),
        members=8,
        weights="ema",
        seed=904003,
        modes=report.MODES,
        script_sha256="diagnostic",
        receipts=[dict(case=c["name"]) for c in cases],
        rows=rows,
        initial_perturbations=perturb,
    )

    def release():
        write(root / "COMPLETE.json", complete)
        write(
            root / "RELEASE_COMPLETE.json",
            dict(
                complete=True,
                checkpoint_sha256="final",
                diagnostic_complete_sha256=report.digest(root / "COMPLETE.json"),
            ),
        )

    release()
    frame, states, _ = report.collect_conditioning(
        root, prepared, {"8000": "final"}, "grid"
    )
    assert len(frame) == 5064 and len(states) == 7392
    complete["rows"] = [
        r
        for r in rows
        if not (
            r["case"] == cases[0]["name"]
            and r["channel"] == "vo_8"
            and r["lead"] == "30"
        )
    ]
    release()
    with pytest.raises(ValueError, match="inventory"):
        report.collect_conditioning(root, prepared, {"8000": "final"}, "grid")
