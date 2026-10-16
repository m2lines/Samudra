# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reduce paired forecasts and draw common-scale RMSE and signed-difference maps."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("control", "aligned", "seasonal")
LEADS = (5, 30, 90, 180, 360)


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regions(lat, lon):
    lat, lon = np.broadcast_arrays(lat[:, None], lon[None, :] % 360)
    masks = dict(
        gulf_stream=(lat >= 25) & (lat <= 50) & (lon >= 280) & (lon <= 320),
        kuroshio=(lat >= 25) & (lat <= 45) & (lon >= 125) & (lon <= 180),
        acc=(lat >= -65) & (lat <= -40),
        tropics=(lat >= -20) & (lat <= 20),
    )
    masks["rest"] = ~np.logical_or.reduce(list(masks.values()))
    return {"global": np.ones_like(lat, dtype=bool), **masks}


def normalized_weights(wet, area, masks):
    weights = np.stack([wet * area * mask for mask in masks.values()]).astype("float64")
    sums = weights.sum((-2, -1), keepdims=True)
    return np.divide(weights, sums, out=np.zeros_like(weights), where=sums != 0)


def reduce_run(path, output, grid, weights):
    errors = np.load(path / "quarterly_errors.npy", mmap_mode="r")
    assert errors.shape == (28, 5, *grid["wet"].shape)
    maps_sq = np.zeros(errors.shape[1:], dtype="float64")
    maps_bias = np.zeros_like(maps_sq)
    mse = np.empty((28, 5, len(weights), len(grid["channels"])))
    for start in range(28):
        for lead in range(5):
            error = errors[start, lead].astype("float64")
            if not np.isfinite(error).all():
                raise ValueError(f"Non-finite saved errors in {path}")
            squared = error**2
            mse[start, lead] = np.einsum(
                "cyx,rcyx->rc", squared, weights, optimize=True
            )
            maps_sq[lead] += squared / 28
            maps_bias[lead] += error / 28
    np.savez_compressed(
        output / f"{path.name}-maps.npz",
        mse=maps_sq.astype("float32"),
        bias=maps_bias.astype("float32"),
    )
    np.save(output / f"{path.name}-mse.npy", mse)
    return mse


def compare(base, treatment, indices):
    b, t = base.mean(), treatment.mean()
    bboot = base.reshape(7, 4).mean(1)[indices].mean(1)
    tboot = treatment.reshape(7, 4).mean(1)[indices].mean(1)
    changes = 100 * (np.sqrt(tboot / bboot) - 1)
    low, high = np.quantile(changes, [0.025, 0.975])
    return dict(
        control_rmse=np.sqrt(b),
        treatment_rmse=np.sqrt(t),
        change_percent=100 * (np.sqrt(t / b) - 1),
        ci_low=low,
        ci_high=high,
    )


def map_canvas(rows, shape, pixels_per_cell):
    """Give every map an integer-aligned raster of square grid cells."""
    if pixels_per_cell < 1:
        raise ValueError("Map pixels per cell must be positive")
    height, width = np.array(shape) * pixels_per_cell
    left, right, top, bottom, gap_x, gap_y = 80, 100, 100, 65, 100, 90
    canvas_width = left + 5 * width + 4 * gap_x + right
    canvas_height = top + rows * height + (rows - 1) * gap_y + bottom
    fig = plt.figure(figsize=(canvas_width / 100, canvas_height / 100), dpi=100)
    axes, bars = [], []
    for row in range(rows):
        y = canvas_height - top - height - row * (height + gap_y)
        map_row, bar_row = [], []
        for col in range(5):
            x = left + col * (width + gap_x)
            map_row.append(
                fig.add_axes(
                    (
                        x / canvas_width,
                        y / canvas_height,
                        width / canvas_width,
                        height / canvas_height,
                    )
                )
            )
            bar_row.append(
                fig.add_axes(
                    (
                        (x + width + 10) / canvas_width,
                        (y + 20) / canvas_height,
                        10 / canvas_width,
                        (height - 40) / canvas_height,
                    )
                )
            )
        axes.append(map_row)
        bars.append(bar_row)
    return fig, axes, bars


def verify_map_pixels(fig, axes, shape, pixels_per_cell):
    """Check the rendered bitmap, including uniform pixels within each cell."""
    fig.canvas.draw()
    bitmap = np.asarray(fig.canvas.buffer_rgba())
    ny, nx = shape
    for row in axes:
        for ax in row:
            bounds = np.array(ax.get_window_extent().extents)
            np.testing.assert_allclose(bounds, np.round(bounds), atol=1e-7, rtol=0)
            x0, y0, x1, y1 = np.round(bounds).astype(int)
            assert (x1 - x0, y1 - y0) == (nx * pixels_per_cell, ny * pixels_per_cell)
            pixels = bitmap[len(bitmap) - y1 : len(bitmap) - y0, x0:x1]
            blocks = pixels.reshape(ny, pixels_per_cell, nx, pixels_per_cell, 4)
            np.testing.assert_array_equal(
                blocks, np.broadcast_to(blocks[:, :1, :, :1], blocks.shape)
            )


def draw_maps(
    output,
    grid,
    channels,
    channel_indices,
    lead_index,
    *,
    bias=False,
    pixels_per_cell=2,
):
    # Common physical-unit range per variable/depth, symmetric common delta range.
    loaded = {arm: dict(np.load(output / f"{arm}-ema-maps.npz")) for arm in ARMS}
    fields = {}
    for arm in ARMS:
        values = loaded[arm]["bias" if bias else "mse"][lead_index]
        fields[arm] = (values if bias else np.sqrt(values)) * grid["std"][:, None, None]
    kind = "bias" if bias else "rmse"
    draw_fields(
        output,
        grid,
        channels,
        channel_indices,
        fields,
        signed=bias,
        title=f"{LEADS[lead_index]}-day nominal {kind}, final EMA; 28 starts",
        filename_tag=f"{kind}-{LEADS[lead_index]}d",
        pixels_per_cell=pixels_per_cell,
    )


def draw_fields(
    output,
    grid,
    channels,
    channel_indices,
    fields,
    *,
    signed,
    title,
    filename_tag,
    pixels_per_cell=2,
):
    """Draw physical-unit error fields with shared scales and exact cell pixels."""
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    lon = (grid["lon"] + 180) % 360 - 180
    order = np.argsort(lon)
    visible = np.flatnonzero(np.abs(grid["lat"]) <= 80)
    lat = grid["lat"][visible]
    shape = (len(visible), len(lon))
    units = {"thetao": "°C", "so": "psu", "uo": "m/s", "vo": "m/s", "zos": "m"}
    for variable in ("thetao", "so", "uo", "vo", "zos"):
        selected = [i for i in channel_indices if channels[i].split("_")[0] == variable]
        if not selected:
            continue
        fig, axes, bars = map_canvas(len(selected), shape, pixels_per_cell)
        for row, channel in enumerate(selected):
            arrays = {}
            for arm in ARMS:
                arrays[arm] = np.where(
                    grid["wet"][channel], fields[arm][channel], np.nan
                )[:, order]
            scale = np.nanpercentile(
                np.concatenate([v.ravel() for v in arrays.values()]), 98
            )
            if signed:
                scale = np.nanpercentile(
                    np.abs(np.concatenate([v.ravel() for v in arrays.values()])), 98
                )
            diffs = [arrays[arm] - arrays["control"] for arm in ARMS[1:]]
            delta_scale = max(
                np.nanpercentile(
                    np.abs(np.concatenate([v.ravel() for v in diffs])), 98
                ),
                1e-10,
            )
            for col, values in enumerate([*arrays.values(), *diffs]):
                ax = axes[row][col]
                delta = col >= 3
                vmax = delta_scale if delta else max(scale, 1e-10)
                image = ax.imshow(
                    values[visible],
                    origin="lower",
                    interpolation="nearest",
                    resample=False,
                    aspect="equal",
                    cmap="RdBu_r" if delta or signed else "viridis",
                    vmin=-vmax if delta or signed else 0,
                    vmax=vmax,
                )
                ax.set_facecolor("#dddddd")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(length=0)
                ax.set_xticks(
                    np.linspace(-0.5, shape[1] - 0.5, 5), [-180, -90, 0, 90, 180]
                )
                lat_ticks = np.array([-60, -30, 0, 30, 60])
                ax.set_yticks(np.interp(lat_ticks, lat, np.arange(len(lat))), lat_ticks)
                ax.set_xlabel("Longitude")
                if col == 0:
                    level = (
                        "surface"
                        if variable == "zos"
                        else f"{float(grid['depths'][int(channels[channel].split('_')[-1])]):g} m"
                    )
                    ax.set_ylabel(level + "\nLatitude")
                if row == 0:
                    ax.set_title(
                        [
                            "Control",
                            "Aligned KE",
                            "Seasonal KE",
                            "Aligned − control",
                            "Seasonal − control",
                        ][col]
                    )
                fig.colorbar(image, cax=bars[row][col], label=units[variable])
        fig.suptitle(
            f"{variable}: {title}\n"
            + (
                "Signed bias differences; gray = land"
                if signed
                else "Blue differences = lower RMSE; gray = land; limits clipped at pooled 98th percentile"
            )
        )
        verify_map_pixels(fig, axes, shape, pixels_per_cell)
        fig.savefig(
            figures / f"{variable}-{filename_tag}.png",
            dpi=100,
            pil_kwargs={"optimize": True},
        )
        plt.close(fig)


def selected_depth_channels(grid, channels):
    indices: list[int] = []
    for target in (0, 100, 700, 2000):
        depth_index = int(np.argmin(abs(grid["depths"] - target)))
        indices.extend(
            i for i, c in enumerate(channels) if c.endswith(f"_{depth_index}")
        )
    indices.append(channels.index("zos"))
    indices = list(dict.fromkeys(indices))
    return indices


def draw_atlas(output, grid, channels, pixels_per_cell):
    indices = selected_depth_channels(grid, channels)
    for lead_index in (1, 3, 4):
        draw_maps(
            output, grid, channels, indices, lead_index, pixels_per_cell=pixels_per_cell
        )
    draw_maps(
        output, grid, channels, indices, 4, bias=True, pixels_per_cell=pixels_per_cell
    )
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maps-only", action="store_true", help="Render cached map arrays"
    )
    parser.add_argument(
        "--map-pixels", type=int, default=2, help="Square pixels per cell"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(args.input / "control-ema/grid.npz"))
    channels = grid["channels"].tolist()
    if args.maps_only:
        draw_atlas(args.output, grid, channels, args.map_pixels)
        return
    masks = regions(grid["lat"], grid["lon"])
    weights = normalized_weights(grid["wet"], grid["area"], masks)
    reference = json.loads((args.input / "control-ema/COMPLETE.json").read_text())
    all_mse = {}
    receipts = {}
    for kind in ("ema", "raw"):
        for arm in ARMS:
            name = f"{arm}-{kind}"
            path = args.input / name
            receipt = json.loads((path / "COMPLETE.json").read_text())
            assert receipt["initializations"] == reference["initializations"]
            assert (
                receipt["evaluation_script_sha256"]
                == reference["evaluation_script_sha256"]
            )
            assert receipt["smoke"] is False
            other = np.load(path / "grid.npz")
            for key in grid:
                np.testing.assert_array_equal(grid[key], other[key])
            receipts[name] = receipt
            all_mse[name] = reduce_run(path, args.output, grid, weights)
            print("Reduced", name, flush=True)
    bootstrap = np.random.default_rng(1563).integers(0, 7, (2000, 7))
    groups = {"all": np.arange(len(channels))}
    groups.update(
        {
            var: np.array([i for i, c in enumerate(channels) if c.split("_")[0] == var])
            for var in ("thetao", "so", "uo", "vo", "zos")
        }
    )
    summaries = []
    detailed = []
    for kind in ("ema", "raw"):
        for arm in ARMS[1:]:
            base = all_mse[f"control-{kind}"]
            treatment = all_mse[f"{arm}-{kind}"]
            for lead_index, lead in enumerate(LEADS):
                for region_index, region in enumerate(masks):
                    for group, which in groups.items():
                        which = which[weights[region_index, which].sum((-2, -1)) > 0]
                        if not len(which):
                            continue
                        summaries.append(
                            dict(
                                checkpoint=kind,
                                treatment=arm,
                                lead=lead,
                                region=region,
                                group=group,
                                **compare(
                                    base[:, lead_index, region_index][:, which].mean(1),
                                    treatment[:, lead_index, region_index][
                                        :, which
                                    ].mean(1),
                                    bootstrap,
                                ),
                            )
                        )
                    for channel, name in enumerate(channels):
                        if weights[region_index, channel].sum() == 0:
                            continue
                        stats = compare(
                            base[:, lead_index, region_index, channel],
                            treatment[:, lead_index, region_index, channel],
                            bootstrap,
                        )
                        stats["control_rmse"] *= float(grid["std"].reshape(-1)[channel])
                        stats["treatment_rmse"] *= float(
                            grid["std"].reshape(-1)[channel]
                        )
                        detailed.append(
                            dict(
                                checkpoint=kind,
                                treatment=arm,
                                lead=lead,
                                region=region,
                                channel=name,
                                **stats,
                            )
                        )
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "channels.csv", detailed)
    continuous = []
    reference_times = None
    for name in all_mse:
        data = np.load(args.input / name / "continuous/scores.npz")
        if reference_times is None:
            reference_times = data["times"]
        np.testing.assert_array_equal(reference_times, data["times"])
        normalized_mse = data["mse"]
        map_mse = np.einsum("cyx,cyx->c", data["mean_squared_error"], weights[0])
        np.testing.assert_allclose(map_mse, normalized_mse.mean(0), rtol=2e-6)
        mean_bias_mse = np.einsum(
            "cyx,cyx->c", data["mean_error"].astype("float64") ** 2, weights[0]
        )
        continuous.append(
            dict(
                run=name,
                trajectory_normalized_rmse=np.sqrt(normalized_mse.mean()),
                time_mean_normalized_rmse=np.sqrt(mean_bias_mse.mean()),
                n_dates=len(reference_times),
            )
        )
    write_csv(args.output / "continuous.csv", continuous)
    indices = draw_atlas(args.output, grid, channels, args.map_pixels)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for arm, color in [("aligned", "#2468b4"), ("seasonal", "#c66327")]:
        for kind, style in [("ema", "-"), ("raw", "--")]:
            rows = [
                r
                for r in summaries
                if r["treatment"] == arm
                and r["checkpoint"] == kind
                and r["region"] == "global"
                and r["group"] == "all"
            ]
            ax.plot(
                LEADS,
                [r["change_percent"] for r in rows],
                style,
                marker="o",
                color=color,
                label=f"{arm}, {kind}",
            )
            if kind == "ema":
                ax.fill_between(
                    LEADS,
                    [r["ci_low"] for r in rows],
                    [r["ci_high"] for r in rows],
                    color=color,
                    alpha=0.15,
                )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Nominal forecast lead (days)",
        ylabel="Normalized RMSE change vs control (%)",
        title="1° Samudra-current + auxiliary subcell KE\n28 matched starts; shading: paired year-block 95% interval",
    )
    ax.legend()
    fig.savefig(args.output / "figures/lead-summary.png", dpi=170)
    plt.close(fig)
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                receipts=receipts,
                regions={k: int(v.sum()) for k, v in masks.items()},
                bootstrap=dict(
                    seed=1563,
                    samples=2000,
                    block="initialization year; four quarters",
                    years=7,
                ),
                selected_channels=[channels[i] for i in indices],
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            [
                r
                for r in summaries
                if r["checkpoint"] == "ema"
                and r["region"] == "global"
                and r["group"] == "all"
            ],
            indent=2,
        ),
        flush=True,
    )
    print(json.dumps(continuous, indent=2), flush=True)


if __name__ == "__main__":
    main()
