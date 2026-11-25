# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Pick the bin tolerance eps (bin width / sigma_delta) from an eps sweep.

    python select_eps.py SWEEP_DIR   # SWEEP_DIR/eps_<value>/results.json, one per eps

Rule (fixed before looking at OM4 numbers): the largest eps at which, on every
surface channel in the sweep, every rung's bits saved over persistence changes
by less than TOL bits when eps is halved. Below that width, model differences
no longer depend on the bins. Deeper channels are reported, not used: at a
per-variable tolerance they may stay in the coarse regime, which is the point.
Writes SWEEP_DIR/eps_selection.json and prints the chosen eps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOL = 0.01


def main(sweep: Path) -> float:
    runs = {}
    for f in sweep.glob("eps_*/results.json"):
        R = json.loads(f.read_text())
        runs[float(R["args"]["eps"])] = {c["channel"]: c for c in R["channels"]}
    if len(runs) < 2:
        raise SystemExit(f"need at least two eps runs in {sweep}, found {len(runs)}")
    eps_desc = sorted(runs, reverse=True)
    for e, half in zip(eps_desc, eps_desc[1:]):
        if abs(half - e / 2) > 1e-12:
            raise SystemExit(f"sweep must halve eps each step; got {e} then {half}")

    def saved(e):
        return {
            c: {m: r["bits"]["persistence"] - r["bits"][m] for m in r["bits"]}
            for c, r in runs[e].items()
        }

    top = min(r["depth"] for r in runs[eps_desc[0]].values())
    surface = [c for c, r in runs[eps_desc[0]].items() if r["depth"] == top]
    table, chosen = [], None
    for e, half in zip(eps_desc, eps_desc[1:]):
        s0, s1 = saved(e), saved(half)
        change = {
            c: max(abs(s0[c][m] - s1[c][m]) for m in s0[c] if m != "persistence")
            for c in s0
        }
        ok = all(change[c] < TOL for c in surface)
        table.append(
            {"eps": e, "max_change_on_halving": change, "surface_converged": ok}
        )
        if ok and chosen is None:
            chosen = e
    out = {
        "rule": f"largest eps with max |change| < {TOL} bits on halving, surface channels",
        "surface_channels": surface,
        "chosen_eps": chosen,
        "sweep": table,
        "persistence_bits": {
            e: {c: r["bits"]["persistence"] for c, r in runs[e].items()}
            for e in eps_desc
        },
    }
    (sweep / "eps_selection.json").write_text(json.dumps(out, indent=1))
    if chosen is None:
        raise SystemExit("no eps in the sweep converged on the surface channels")
    return chosen


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
