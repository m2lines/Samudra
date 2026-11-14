# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The scorecard as something to read: a Markdown summary and four figures.

Every number drawn or printed here comes from the scorecard frame (or, for the
time series, from the same series tables the frame was built from), so a figure
cannot disagree with the table beside it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from samudra.metrics.scorecard import card, gate
from samudra.metrics.scorecard.series import Series

MODEL_COLOR = "#1f6fb8"
TRUTH_COLOR = "#c4520f"
GRAY = "#6b6b6b"
VARIABLES = ("thetao", "so", "uo", "vo", "zos")
LAYER_ORDER = ("surface", "upper", "intermediate", "deep", card.FULL_DEPTH)


def _fmt(v: float, ci: tuple[float, float] | None = None, digits: int = 3) -> str:
    if not np.isfinite(v):
        return "-"
    s = f"{v:.{digits}f}"
    if ci is not None and all(np.isfinite(ci)):
        s += f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
    return s


def markdown(
    frame: pd.DataFrame,
    meta: dict,
    checks: list[gate.Check],
    comparison: pd.DataFrame | None,
) -> str:
    """A self-explaining summary: what was scored, the gate, and the layer table."""
    w = frame[(frame.scope == "layer") & (frame.period_kind == "window")]
    lead = frame[(frame.scope == "layer") & (frame.period_kind == "lead")]
    verdicts: dict[tuple[str, str, str], str] = {}
    if comparison is not None:
        for r in comparison[comparison.scope == "layer"].itertuples():
            verdicts[(str(r.metric), str(r.variable), str(r.layer))] = str(r.verdict)

    def get(df, metric, var, layer, period=None):
        sel = df[(df.metric == metric) & (df.variable == var) & (df.layer == layer)]
        if period is not None:
            sel = sel[sel.period == period]
        return sel.iloc[0] if len(sel) else None

    lines = [
        f"# Scorecard: {meta['label']}",
        "",
        f"Track `{meta['track']}` · window {meta['window']} · "
        f"truth `{meta['truth']}` · rollout from {meta['first_forecast']}"
        + (
            f" · champion `{meta['champion_label']}`"
            if meta.get("champion_label")
            else ""
        ),
        "",
        "## Gate",
        "",
        f"**{'PASS' if gate.passed(checks) else 'FAIL'}**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        *(
            f"| {c.name} | {'pass' if c.passed else '**fail**'} | {c.detail} |"
            for c in checks
        ),
        "",
        "## Physical plausibility, over the window",
        "",
        "| Variable | Max (model / truth) | Min (model / truth) | Unphysical share of wet area | Non-finite share |",
        "|---|---|---|---|---|",
        *_plausibility_rows(frame),
        "",
        "## Skill by variable and layer",
        "",
        "Skill is 1 − MSE_model / MSE_reference: 1 is perfect, 0 matches the "
        "reference, negative is worse. Window values weight each calendar year "
        "equally, with a 95% calendar-year block-bootstrap interval. "
        "Persistence holds the rollout's initial state fixed, so its skill is "
        "meaningful only at short leads. Bias is model minus truth, "
        "volume-weighted. A variance ratio below 1 means the rollout keeps less "
        "deseasonalized variability than the truth."
        + (" Verdicts compare with the champion, paired by year." if verdicts else ""),
        "",
        "| Variable | Layer | Skill vs climatology | Skill vs persistence, step 1 "
        "| Skill vs persistence, to 1 mo | Bias | Anomaly variance ratio"
        + (" | vs champion (skill, bias)" if verdicts else "")
        + " |",
        "|---|---|---|---|---|---|---" + ("|---" if verdicts else "") + "|",
    ]
    for var in VARIABLES:
        for layer in LAYER_ORDER:
            sk = get(w, "skill_vs_climatology", var, layer)
            if sk is None:
                continue
            b = get(w, "bias", var, layer)
            av = get(w, "anomaly_variance_ratio", var, layer)
            p1 = get(lead, "skill_vs_persistence", var, layer, card.LEAD_BINS[0][0])
            pm = get(lead, "skill_vs_persistence", var, layer, card.LEAD_BINS[1][0])
            cells = [
                f"`{var}`",
                layer,
                _fmt(sk.value, (sk.ci_low, sk.ci_high)),
                _fmt(p1.value) if p1 is not None else "-",
                _fmt(pm.value) if pm is not None else "-",
                _fmt(b.value, (b.ci_low, b.ci_high), 4) + f" {b.units}",
                _fmt(av.value, digits=2),
            ]
            if verdicts:
                cells.append(
                    f"{verdicts.get(('skill_vs_climatology', var, layer), '-')}, "
                    f"{verdicts.get(('bias', var, layer), '-')}"
                )
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Figures",
        "",
        "- `skill-vs-climatology.png`, `skill-vs-persistence.png`: skill by depth "
        "level (rows) and lead bin or calendar year (columns). Red is better than "
        "the reference, blue worse.",
        "- `drift.png`: volume-mean temperature and salinity bias by layer and "
        "year, and the warmest surface temperature per step against the truth's.",
        "- `variance.png`: deseasonalized variance ratio and spatial-spread ratio "
        "by depth. 1 is right; below 1 is variance collapse, above 1 excess noise.",
    ]
    if comparison is not None:
        lines.append(
            "- `vs-champion.png`: window skill against climatology by variable and "
            "layer, candidate beside champion."
        )
    lines += [
        "",
        "Every row behind these is in `scorecard.parquet` (channel and layer "
        "scopes; window, year and lead periods).",
        "",
    ]
    return "\n".join(lines)


def _plausibility_rows(frame: pd.DataFrame) -> list[str]:
    h = frame[frame.scope == "health"].set_index(["variable", "metric"])
    rows = []
    for var in VARIABLES:
        if (var, "max") not in h.index:
            continue

        def pair(metric: str, var: str = var) -> str:
            model = float(np.asarray(h.loc[(var, metric), "value"]))
            truth = float(np.asarray(h.loc[(var, metric), "truth_value"]))
            return f"{model:.3g} / {truth:.3g}"

        unphys = (
            pair("unphysical_area_fraction")
            if (var, "unphysical_area_fraction") in h.index
            else "-"
        )
        rows.append(
            f"| `{var}` | {pair('max')} | {pair('min')} | {unphys} | "
            f"{float(np.asarray(h.loc[(var, 'nonfinite_area_fraction'), 'value'])):.3g} |"
        )
    return rows


def _heatmap(frame: pd.DataFrame, metric: str, path: Path) -> None:
    chan = frame[(frame.scope == "channel") & (frame.metric == metric)]
    periods = [label for label, _, _ in card.LEAD_BINS] + sorted(
        set(chan.loc[chan.period_kind == "year", "period"])
    )
    vars3d = [v for v in VARIABLES if v != "zos" and v in set(chan.variable)]
    fig, axes = plt.subplots(
        1,
        len(vars3d),
        figsize=(4.2 * len(vars3d), 6.5),
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    im = None
    for ax, var in zip(axes, vars3d, strict=True):
        sub = chan[(chan.variable == var) & chan.period_kind.isin(["lead", "year"])]
        table = sub.pivot_table(index="depth_m", columns="period", values="value")
        table = table.reindex(columns=[p for p in periods if p in table.columns])
        im = ax.imshow(
            np.clip(table.to_numpy(), -1, 1),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(var)
        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels([f"{d:g}" for d in table.index], fontsize=8)
        ax.axvline(len(card.LEAD_BINS) - 0.5, color="black", lw=1)
    axes[0].set_ylabel("Depth (m)")
    ref = metric.removeprefix("skill_vs_")
    fig.suptitle(
        f"Skill against {ref} (1 − MSE ratio), clipped to ±1 · left of the line: "
        "lead from the rollout's start · right: calendar years"
    )
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label=f"skill vs {ref}")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _drift(frame: pd.DataFrame, series: Series, path: Path) -> None:
    years = frame[(frame.scope == "layer") & (frame.metric == "bias")]
    years = years[years.period_kind == "year"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for ax, var in zip(axes[:2], ("thetao", "so"), strict=True):
        for layer, style in zip(
            ("upper", "intermediate", "deep", card.FULL_DEPTH),
            ("-", "--", ":", "-"),
            strict=True,
        ):
            s = years[(years.variable == var) & (years.layer == layer)]
            if s.empty:
                continue
            ax.plot(
                s.period.astype(int),
                s.value,
                style,
                color="black" if layer == card.FULL_DEPTH else MODEL_COLOR,
                lw=2,
                label=layer,
            )
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_title(f"{var}: volume-mean bias by year ({card.UNITS[var]})")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    h = series.health[series.health.quantity == "thetao_max"]
    ax = axes[2]
    ax.plot(h.time, h.model, color=MODEL_COLOR, lw=1.5, label="model")
    ax.plot(h.time, h.truth, color=TRUTH_COLOR, lw=1.5, label="truth (OM4)")
    ax.set_title("Warmest temperature anywhere, per step (degC)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _variance(frame: pd.DataFrame, path: Path) -> None:
    w = frame[(frame.scope == "channel") & (frame.period_kind == "window")]
    vars3d = [v for v in VARIABLES if v != "zos" and v in set(w.variable)]
    fig, axes = plt.subplots(
        1, len(vars3d), figsize=(4 * len(vars3d), 5), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, vars3d, strict=True):
        for metric, color, label in (
            ("anomaly_variance_ratio", MODEL_COLOR, "deseasonalized variance"),
            ("spatial_std_ratio", TRUTH_COLOR, "spatial spread"),
        ):
            s = w[(w.variable == var) & (w.metric == metric)].sort_values("depth_m")
            ax.plot(
                s.value, s.depth_m, color=color, lw=2, marker="o", ms=4, label=label
            )
        ax.axvline(1, color=GRAY, lw=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale("log")
        ax.set_title(var)
        ax.grid(alpha=0.2)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model ÷ truth, by depth: 1 is right, < 1 collapse, > 1 excess")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _vs_champion(frame: pd.DataFrame, champion: pd.DataFrame, path: Path) -> None:
    def rows(df):
        return df[
            (df.scope == "layer")
            & (df.period_kind == "window")
            & (df.metric == "skill_vs_climatology")
        ].set_index(["variable", "layer"])

    cand, champ = rows(frame), rows(champion)
    keys = [k for k in cand.index if k in champ.index]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(keys) + 1.5), layout="constrained")
    for i, k in enumerate(keys):
        for df, color, dy, label in (
            (champ, GRAY, 0.15, "champion"),
            (cand, MODEL_COLOR, -0.15, "candidate"),
        ):
            r = df.loc[k]
            ax.errorbar(
                r.value,
                i + dy,
                xerr=[[r.value - r.ci_low], [r.ci_high - r.value]]
                if np.isfinite(r.ci_low)
                else None,
                fmt="o",
                color=color,
                ms=6,
                label=label if i == 0 else None,
            )
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{v} · {layer}" for v, layer in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("skill vs climatology, window (95% CI)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, axis="x")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def figures(
    frame: pd.DataFrame,
    series: Series,
    out: Path,
    champion: pd.DataFrame | None = None,
) -> list[Path]:
    paths = {
        "skill-vs-climatology.png": lambda p: _heatmap(
            frame, "skill_vs_climatology", p
        ),
        "skill-vs-persistence.png": lambda p: _heatmap(
            frame, "skill_vs_persistence", p
        ),
        "drift.png": lambda p: _drift(frame, series, p),
        "variance.png": lambda p: _variance(frame, p),
    }
    if champion is not None:
        paths["vs-champion.png"] = lambda p: _vs_champion(frame, champion, p)
    written = []
    for name, draw in paths.items():
        draw(out / name)
        written.append(out / name)
    return written
