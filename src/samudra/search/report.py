# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Human- and agent-readable reports for architecture searches."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from samudra.search.successive_halving import SuccessiveHalving


def _markdown(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_search_report(search: SuccessiveHalving, state: dict[str, Any]) -> Path:
    """Atomically write a compact account of outcomes and rung progression."""
    rows = search.result_rows(state)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = str(row["candidate"])
        if candidate not in latest or int(row["rung"]) > int(latest[candidate]["rung"]):
            latest[candidate] = row
    ascending = search.config.objective.mode == "min"
    eligible = [row for row in latest.values() if row.get("eligible")]
    eligible.sort(
        key=lambda row: float(row[search.config.objective.metric]),
        reverse=not ascending,
    )
    failures = [row for row in latest.values() if not row.get("eligible")]
    final_rung = len(search.rungs) - 1
    finalists = [row for row in eligible if int(row["rung"]) == final_rung]
    winner = finalists[0] if state.get("status") == "complete" and finalists else None

    lines = [
        f"# {search.config.name}",
        "",
        f"- Search run: `{search.run_id}`",
        f"- Status: **{_markdown(state.get('status'))}**",
        f"- Objective: `{search.config.objective.metric}` "
        f"({search.config.objective.mode})",
        f"- Code commit: `{_markdown(state.get('provenance', {}).get('commit'))}`",
        f"- Created: {_markdown(state.get('created_at'))}",
        "",
    ]
    if winner is not None:
        lines.extend(
            [
                "## Outcome",
                "",
                f"**Winner: `{_markdown(winner['candidate'])}`** at "
                f"{_markdown(winner.get('epochs'))} epochs with "
                f"`{search.config.objective.metric}="
                f"{_markdown(winner[search.config.objective.metric])}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Latest eligible result per candidate",
            "",
            "| Rank | Candidate | Rung | Epochs | Objective | Train loss | "
            "Best validation loss | Optimizer steps | W&B ID |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(eligible, start=1):
        values = [
            rank,
            f"`{_markdown(row['candidate'])}`",
            row["rung"],
            row.get("epochs"),
            row.get(search.config.objective.metric),
            row.get("train_loss"),
            row.get("best_validation_loss"),
            row.get("optimizer_steps"),
            row.get("wandb_id"),
        ]
        lines.append("| " + " | ".join(_markdown(value) for value in values) + " |")
    if not eligible:
        lines.append("| — | No eligible results yet | — | — | — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Rung progression",
            "",
            "| Rung | Epoch budget | Candidates | Eligible | Promoted |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for rung in state["rungs"]:
        promoted = ", ".join(f"`{_markdown(name)}`" for name in rung["promoted"])
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown(rung["index"]),
                    _markdown(rung["epochs"]),
                    _markdown(len(rung["candidates"])),
                    _markdown(
                        sum(bool(row.get("eligible")) for row in rung["results"])
                    ),
                    promoted or "—",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Ineligible or failed results", ""])
    if failures:
        lines.extend(
            [
                "| Candidate | Rung | Epochs | Reason |",
                "|---|---:|---:|---|",
            ]
        )
        for row in sorted(failures, key=lambda item: str(item["candidate"])):
            values = [
                f"`{_markdown(row['candidate'])}`",
                row["rung"],
                row.get("epochs"),
                row.get("error", "unknown"),
            ]
            lines.append("| " + " | ".join(_markdown(value) for value in values) + " |")
    else:
        lines.append("None recorded.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Promotion uses only the configured objective at each cumulative "
            "epoch budget. Use [`results.parquet`](../results.parquet) for "
            "rung-level comparisons and [`epochs.parquet`](../epochs.parquet) "
            "for learning curves, timing, and metric breakdowns. Short-budget "
            "rankings are screening evidence, not a substitute for matched "
            "rollout evaluation of the finalists.",
            "",
        ]
    )
    report = search.search_dir / "analysis/report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=report.parent,
        prefix=f".{report.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, report)
    return report
