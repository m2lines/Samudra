# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Glue: score a rollout, compare with the champion, write the scorecard directory.

The directory is self-contained and is exactly what the registry stores for a
champion:

- ``scorecard.parquet``: the frame (`card.COLUMNS`), the contract;
- ``meta.json``: what was scored (track, label, window, data, git hash);
- ``series_*.parquet``: the tables the frame was built from, so it can be
  rebuilt or redrawn without the fields;
- ``comparison.parquet`` (with a champion), ``scorecard.md`` and the figures.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pandas as pd
import xarray as xr

from samudra.config import ScorecardConfig
from samudra.constants import DataLayout
from samudra.metrics.scorecard import card, gate, render, series
from samudra.utils.wandb import MetricsDict

logger = logging.getLogger(__name__)

SCORECARD_FILE = "scorecard.parquet"
META_FILE = "meta.json"
COMPARISON_FILE = "comparison.parquet"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Scored:
    frame: pd.DataFrame
    meta: dict
    checks: list[gate.Check]
    comparison: pd.DataFrame | None

    @property
    def passed(self) -> bool:
        return gate.passed(self.checks)


def layout_depths(layout: DataLayout, channels: list[str]):
    """Depth and thickness of each channel's level, from the data layout."""
    depth, dz = {}, {}
    for c in channels:
        _, level = series.channel_parts(c)
        depth[c] = float(layout.depth_levels[level])
        dz[c] = float(layout.depth_thickness[level])
    return depth, dz


def policy(cfg: ScorecardConfig) -> gate.GatePolicy:
    rankable = {m for m, rule in card.BETTER.items() if rule != card.BOUNDED}
    unknown = set(cfg.gated_metrics) - rankable
    if unknown:
        raise ValueError(
            f"Unknown gated metrics {sorted(unknown)}; choose from {sorted(rankable)}"
        )
    return gate.GatePolicy(
        max_sst_above_truth=cfg.max_sst_above_truth,
        min_temperature_below_truth=cfg.min_temperature_below_truth,
        gated_metrics=tuple(cfg.gated_metrics),
        regression_level=cfg.regression_level,
    )


def read_scorecard(location: str) -> tuple[pd.DataFrame, dict]:
    """A scorecard directory, local or remote (any fsspec URL)."""
    base = location.rstrip("/")
    with fsspec.open(f"{base}/{META_FILE}", "r") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{location} has scorecard schema {meta.get('schema_version')}, "
            f"this code reads {SCHEMA_VERSION}. Rescore the champion."
        )
    with fsspec.open(f"{base}/{SCORECARD_FILE}", "rb") as handle:
        frame = pd.read_parquet(handle)
    return frame, meta


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def score_series(
    tables: series.Series,
    cfg: ScorecardConfig,
    layout: DataLayout,
    meta: dict,
) -> Scored:
    """Everything after the pass over the fields. Cheap; reruns freely."""
    channels = list(dict.fromkeys(tables.sums["channel"]))
    depth, dz = layout_depths(layout, channels)
    frame = card.build(
        tables,
        depth=depth,
        dz=dz,
        years=cfg.year_list,
        bootstrap_samples=cfg.bootstrap_samples,
    )
    champion = None
    comparison = None
    if cfg.champion is not None:
        champion, champ_meta = read_scorecard(cfg.champion)
        if champ_meta["track"] != cfg.track:
            raise ValueError(
                f"The champion is on track {champ_meta['track']!r}, the candidate "
                f"on {cfg.track!r}. Tracks are never compared."
            )
        meta = {**meta, "champion_label": champ_meta["label"], "champion": cfg.champion}
        comparison = gate.compare(
            frame, champion, bootstrap_samples=cfg.bootstrap_samples
        )
    checks = gate.checks(frame, policy(cfg), champion, cfg.bootstrap_samples)
    meta = {**meta, "passed": gate.passed(checks)}
    return Scored(frame, meta, checks, comparison)


def score(
    predictions: xr.Dataset,
    truth: xr.Dataset,
    climatology: xr.Dataset,
    cfg: ScorecardConfig,
    layout: DataLayout,
    *,
    label: str,
    truth_name: str,
) -> tuple[series.Series, Scored]:
    """The full scorecard for one rollout: the pass over the fields, then the rest."""
    channels = list(layout.prognostic_var_names)
    tables = series.compute(predictions, truth, climatology, channels)
    first = str(predictions["time"].values[0])
    meta = {
        "schema_version": SCHEMA_VERSION,
        "track": cfg.track,
        "label": label,
        "window": f"{cfg.years[0]}-{cfg.years[1]}",
        "truth": truth_name,
        "first_forecast": first,
        "n_steps": int(predictions.sizes["time"]),
        "climatology_period": climatology.attrs.get("climatology_period", "unknown"),
        "rollout_attrs": {k: str(v) for k, v in predictions.attrs.items()},
        "git_hash": git_hash(),
    }
    return tables, score_series(tables, cfg, layout, meta)


def headline(scored: Scored) -> MetricsDict:
    """A few scalars for W&B: the gate, and window skill against climatology per layer.

    Kept short on purpose (W&B charges for storage); the full frame is on disk.
    """
    f = scored.frame
    rows = f[
        (f.scope == "layer")
        & (f.period_kind == "window")
        & (f.metric == "skill_vs_climatology")
    ]
    out: MetricsDict = {"scorecard/passed": float(scored.passed)}
    for var, layer, value in zip(rows.variable, rows.layer, rows.value, strict=True):
        out[f"scorecard/skill_vs_climatology/{var}/{layer}"] = float(value)
    return out


def write(out: Path, tables: series.Series, scored: Scored) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tables.write(out)
    scored.frame.to_parquet(out / SCORECARD_FILE, index=False)
    (out / META_FILE).write_text(json.dumps(scored.meta, indent=2, default=str))
    champion = None
    if scored.comparison is not None:
        scored.comparison.to_parquet(out / COMPARISON_FILE, index=False)
        champion, _ = read_scorecard(scored.meta["champion"])
    (out / "scorecard.md").write_text(
        render.markdown(scored.frame, scored.meta, scored.checks, scored.comparison)
    )
    render.figures(scored.frame, tables, out, champion)
    logger.info(
        "Wrote the scorecard to %s (%s)", out, "PASS" if scored.passed else "FAIL"
    )
