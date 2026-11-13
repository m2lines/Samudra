# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Compare two runs channel by channel, e.g. with and without forcing.

    python compare.py BASE/results.json NEW/results.json OUT.md [--label-base X --label-new Y]

Both runs must share channels, bin tolerance and test rows (same cache, eps and
seed), so per-channel differences are paired. Writes a markdown table: per
variable, bits per wet value (levels weighted by wet area) for persistence and
mix, bits saved by NEW over BASE, RMSE skill of mix in each, and the estimated
surface joint-coding gap in each. Then the per-channel table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARS = ["thetao", "so", "uo", "vo", "zos"]


def var_of(c: str) -> str:
    return "zos" if c == "zos" else c.rsplit("_", 1)[0]


def load(p: Path) -> tuple[dict, dict]:
    R = json.loads(p.read_text())
    return R, {c["channel"]: c for c in R["channels"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label-base", default="base")
    ap.add_argument("--label-new", default="new")
    a = ap.parse_args()
    Rb, B = load(a.base)
    Rn, N = load(a.new)
    if set(B) != set(N):
        raise SystemExit("runs have different channels")
    for c in B:
        if not np.isclose(B[c]["bin_width"], N[c]["bin_width"], rtol=1e-12):
            raise SystemExit(f"{c}: bin widths differ, differences would not be paired")
        if B[c]["n_eval_rows"] != N[c]["n_eval_rows"]:
            raise SystemExit(f"{c}: different numbers of test rows")
        # Persistence ignores the features, so identical test rows give identical
        # persistence bits. This checks that the comparison is truly paired.
        pb, pn = B[c]["bits"]["persistence"], N[c]["bits"]["persistence"]
        if not np.isclose(pb, pn, rtol=0, atol=1e-9):
            raise SystemExit(f"{c}: persistence bits differ ({pb} vs {pn}): not paired")
    wet = Rb["meta"]["wet_area"]
    lb, ln = a.label_base, a.label_new

    def skill(ch, c):
        return 1 - ch[c]["rmse"]["mix"] / ch[c]["rmse"]["persistence"]

    lines = [
        f"Per variable: bits per wet value (levels weighted by wet area), surface in brackets. "
        f"Saved = {lb} mix bits - {ln} mix bits.",
        "",
        f"| variable | {lb} mix bits | {ln} mix bits | saved by {ln} | RMSE skill of mix, {lb} | RMSE skill of mix, {ln} | surface gap, {lb} | surface gap, {ln} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARS:
        cs = sorted([c for c in B if var_of(c) == v], key=lambda c: B[c]["depth"])
        if not cs:
            continue
        w = [wet[c] for c in cs]
        s = cs[0]

        def agg(f, nd=2, cs=cs, w=w, s=s):
            return f"{np.average([f(c) for c in cs], weights=w):.{nd}f} [{f(s):.{nd}f}]"

        lines.append(
            f"| {v} | {agg(lambda c: B[c]['bits']['mix'])} | {agg(lambda c: N[c]['bits']['mix'])} "
            f"| {agg(lambda c: B[c]['bits']['mix'] - N[c]['bits']['mix'])} "
            f"| {agg(lambda c: skill(B, c), 3)} | {agg(lambda c: skill(N, c), 3)} "
            f"| {B[s]['joint_coding_gap_bits_est']:.2f} | {N[s]['joint_coding_gap_bits_est']:.2f} |"
        )
    lines += [
        "",
        f"| channel | depth (m) | {lb} persistence | {lb} lpc | {ln} lpc | {lb} mix | {ln} mix | mix saved by {ln} | RMSE skill {lb} | RMSE skill {ln} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(B, key=lambda c: (var_of(c), B[c]["depth"])):
        lines.append(
            f"| {c} | {B[c]['depth']:g} | {B[c]['bits']['persistence']:.2f} "
            f"| {B[c]['bits']['lpc']:.2f} | {N[c]['bits']['lpc']:.2f} "
            f"| {B[c]['bits']['mix']:.2f} | {N[c]['bits']['mix']:.2f} "
            f"| {B[c]['bits']['mix'] - N[c]['bits']['mix']:.3f} "
            f"| {skill(B, c):.3f} | {skill(N, c):.3f} |"
        )
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
