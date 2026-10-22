# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact native-calendar model windows for source-matched annual diagnostics."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np

from samudra.experiments.monthly_annual import month_layout
from samudra.experiments.monthly_model_data import calendar_context


def annual_indices(bounds, origin):
    matches = [i for i, (start, _) in enumerate(bounds) if str(start)[:10] == origin]
    if len(matches) != 1:
        raise ValueError("No unique native model origin")
    first = matches[0]
    if first < 19 or first + 73 > len(bounds):
        raise ValueError("Incomplete model annual coverage")
    selected = bounds[first - 19 : first + 73]
    for i, (start, end) in enumerate(selected):
        if end - start != timedelta(days=5):
            raise ValueError("Model intervals are not exact five-day means")
        if i and start != selected[i - 1, 1]:
            raise ValueError(f"Model time gap: {selected[i - 1, 1]} → {start}")
    return np.arange(first - 19, first + 73)


def model_annual_payload(reader, data, origin):
    """Read exact OM4 intervals; targets never enter the initializer or processor."""
    indices = annual_indices(reader.bounds, origin)
    np.testing.assert_array_equal(reader.mask, data.grid["mask"])
    interval = slice(int(indices[0]), int(indices[-1]) + 1)
    names = reader.names + ["tauuo", "tauvo", "hfds"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        arrays = list(pool.map(lambda name: reader.store[name][interval], names))
    values = np.stack(arrays[:77], 1)
    forcing = np.stack(arrays[77:], 1)
    del arrays
    if not np.isfinite(values[:, reader.mask]).all():
        raise ValueError("Nonfinite OM4 physical data on wet support")
    if not np.isfinite(forcing[:, :, reader.mask[38]]).all():
        raise ValueError("Nonfinite OM4 native forcing")
    history = np.where(
        reader.mask,
        (values[:19] - data.grid["mean"][:, None, None])
        / data.grid["std"][:, None, None],
        0,
    ).astype(np.float32)
    normalized_forcing = np.where(
        reader.mask[38],
        (forcing - reader.forcing_mean[:, None, None])
        / reader.forcing_std[:, None, None],
        0,
    ).astype(np.float32)
    starts = np.array([str(reader.bounds[i, 0]) for i in indices])
    midpoints = np.array([str(reader.midpoints[i]) for i in indices])
    temperature, ohc, months = [], [], []
    for month in range(1, 13):
        label = origin[:4] + f"-{month:02d}"
        positions, weights, _, _ = month_layout(starts[19:], label)
        physical = (values[19 + positions] * weights[:, None, None, None]).sum(0)
        normalized = np.where(
            reader.mask,
            (physical - data.grid["mean"][:, None, None])
            / data.grid["std"][:, None, None],
            0,
        ).astype(np.float32)
        # Return the physical reference represented by these normalization
        # scales so its T profiles and independently integrated OHC agree.
        recovered = (
            data.physical(data.tensor(normalized)[None, None])[0, 0].cpu().numpy()
        )
        temperature.append(recovered[38:52])
        ohc.append(data.ohc(data.tensor(normalized)[None])[0])
        months.append(label)
    slots = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(14)])
    return dict(
        fields=history,
        forcing=normalized_forcing,
        contexts=calendar_context(
            reader.lat, reader.lon, [reader.midpoints[i] for i in indices]
        ),
        start=starts,
        midpoint=midpoints,
        months=np.array(months),
        reference_surface=values[19:, [38, 76, 0, 19]],
        reference_initial=values[18, slots],
        reference_temperature=np.stack(temperature),
        reference_ohc=np.stack(ohc),
        source_indices=indices,
    )


def model_batch(data, prepared):
    support = np.zeros_like(data.grid["mask"])
    support[[38, 76]] = data.grid["mask"][[38, 76]]
    return dict(
        fields=data.tensor(prepared["fields"])[None],
        available=data.tensor(support).bool()[None, None].expand(1, 19, -1, -1, -1),
        forcing=data.tensor(prepared["forcing"])[None],
        contexts=data.tensor(prepared["contexts"])[None],
    )
