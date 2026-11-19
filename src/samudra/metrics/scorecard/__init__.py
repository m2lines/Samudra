# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The scorecard: is this model better than the last one, and better than doing nothing?

A model's errors only mean something next to a reference. The scorecard scores
a finished rollout against the OM4 truth at its own resolution, and reports
every error as *skill* relative to two reference forecasts computed from the
truth (`samudra.metrics.skill`): persistence of the initial state, and the
pentad-of-year climatology of the training years. It also checks the rollout
stays physical, and compares everything with the *champion*, the best-known
model on the same track, paired by calendar year.

The pieces, in the order they run:

- `series`: the one pass over the fields. A rollout and its truth become three
  small tables of additive sums, extremes and variances.
- `card`: those tables become the scorecard frame, one row per metric, scope
  (a channel, or a variable over a depth layer) and period (the window, a
  calendar year, or a lead bin), with block-bootstrap intervals.
- `gate`: the paired comparison with the champion, and the hard checks that
  decide whether the candidate passes.
- `render`: a Markdown summary and figures, drawn from the frame.
- `climatology`: builds a track's climatology once, from its training split.

Run it on an existing rollout with ``python -m samudra.metrics.scorecard``, or
add a ``scorecard`` block to an eval config to run it after the rollout. The
design and its decisions are in #986.
"""
