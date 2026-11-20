# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Input-only geometric coverage proposals with exact finite-pool correction."""

import numpy as np
import torch


def geometric_groups(features, depths, clusters_per_depth):
    """Farthest-point Voronoi cells, separately within each query depth.

    This uses geometric spread, not balanced quantiles: a rare, distant input
    state can form a small cell. No future target or model prediction is used.
    """
    labels = np.empty(len(features), dtype=np.int64)
    next_label = 0
    for depth in np.unique(depths):
        indices = np.flatnonzero(depths == depth)
        x = features[indices].astype(np.float64)
        center = int(np.argmin(((x - x.mean(0)) ** 2).sum(1)))
        distance = np.full(len(x), np.inf)
        assigned = np.zeros(len(x), dtype=np.int64)
        for cluster in range(min(clusters_per_depth, len(x))):
            candidate = ((x - x[center]) ** 2).sum(1)
            improve = candidate < distance
            distance[improve] = candidate[improve]
            assigned[improve] = cluster
            center = int(distance.argmax())
            if distance[center] == 0:
                break
        # Equal feature vectors and ties can leave some seeds without a cell.
        _, compact = np.unique(assigned, return_inverse=True)
        labels[indices] = compact + next_label
        next_label += int(compact.max()) + 1
    return labels


def proposal_probabilities(labels, base_fraction=0.5):
    if not 0 < base_fraction <= 1 or not len(labels):
        raise ValueError("Need a nonempty pool and a positive exploration fraction")
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n = len(labels)
    proposal = base_fraction / n + (1 - base_fraction) / (len(counts) * counts[inverse])
    np.testing.assert_allclose(proposal.sum(), 1, rtol=1e-12)
    return proposal


def draw_locations(source, rng, size):
    depths = rng.integers(0, 19, size)
    uniform = rng.random(size)
    flat = np.empty(size, dtype=np.int64)
    for depth in np.unique(depths):
        selected = depths == depth
        ids = np.searchsorted(
            source.location_cdfs[depth], uniform[selected], side="right"
        )
        flat[selected] = source.locations[depth][ids]
    return depths, flat // source.width, flat % source.width


def input_features(source, frame, depths, rows, cols):
    known = frame["input"]
    current = known[1, 38 + depths, rows, cols]
    return np.stack(
        (
            current,
            known[1, 38, rows, cols],
            known[1, 44, rows, cols],
            known[1, 49, rows, cols],
            known[1, 57 + depths, rows, cols],
            known[1, depths, rows, cols],
            known[1, 19 + depths, rows, cols],
            current - known[0, 38 + depths, rows, cols],
            source.wet[:, rows, cols].mean(0),
            source.wet[6, rows, cols],
            source.wet[11, rows, cols],
            source.xyz[rows, cols, 0],
            source.xyz[rows, cols, 1],
            source.xyz[rows, cols, 2],
        ),
        -1,
    ).astype(np.float32)


def sample_coverage(source, frame, rng, size, crop, config):
    """A fresh i.i.d. base pool per minibatch; sample occurrences with replacement.

    Conditional on the pool, p_i=1/M and w_i=p_i/q_i gives its exact mean
    loss/gradient in expectation. Averaging over base pools recovers the
    original distribution conditional on the loaded date. Duplicate physical
    points in the pool remain separate occurrences, as required by this proof.
    """
    n = int(config["candidate_pool"])
    if n < size or config["clusters_per_depth"] < 1:
        raise ValueError("Invalid coverage pool or cluster count")
    depths, rows, cols = draw_locations(source, rng, n)
    features = input_features(source, frame, depths, rows, cols)
    if not np.isfinite(features).all():
        raise FloatingPointError("Nonfinite input-only coverage features")
    groups = geometric_groups(features, depths, config["clusters_per_depth"])
    q = proposal_probabilities(groups, config["base_fraction"])
    selected = rng.choice(n, size=size, replace=True, p=q)
    batch = source.sample(
        frame,
        rng,
        size,
        crop,
        depths[selected],
        list(zip(rows[selected], cols[selected], strict=True)),
    )
    batch["importance"] = torch.from_numpy((1 / (n * q[selected])).astype(np.float32))
    batch["proposal_probability"] = torch.from_numpy(q[selected].astype(np.float32))
    batch["coverage_group"] = torch.from_numpy(groups[selected])
    return batch


def corrected_mean(losses, importance):
    """No clipping or self-normalization: preserve the original expected loss."""
    if losses.shape != importance.shape:
        raise ValueError("Need one sampling weight per example")
    return (losses * importance.detach()).mean()
