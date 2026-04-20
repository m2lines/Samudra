import gc
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from wandb.data_types import WBValue

from ocean_emulators.utils.wandb import WandBLogger


def get_cmap_limits(data: np.ndarray, diverging=False) -> tuple[float, float]:
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    return vmin, vmax


def plot_imshow(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Colormap | None = None,
    flip_lat: bool = True,
    use_colorbar: bool = True,
    nan_padding: bool = True,
) -> Figure:
    """Plot a 2D array using imshow, ensuring figure size is same as array size."""
    min_ = np.nanmin(data) if vmin is None else vmin
    max_ = np.nanmax(data) if vmax is None else vmax

    if flip_lat:
        lat_dim = -2
        data = np.flip(data, axis=lat_dim)

    if use_colorbar:
        height, width = data.shape
        colorbar_width = max(1, int(0.025 * width))
        range_ = np.linspace(min_, max_, height)
        range_ = np.repeat(range_[:, np.newaxis], repeats=colorbar_width, axis=1)
        range_ = np.flipud(range_)  # wandb images start from top (and left)
        padding = np.zeros((height, colorbar_width))
        if nan_padding:
            padding = padding + np.nan  # Set when using non-diverging map
        data = np.concatenate((data, padding, range_), axis=1)

    # make figure size (in pixels) be the same as array size
    figsize = np.array(data.T.shape) / plt.rcParams["figure.dpi"]
    fig = Figure(figsize=figsize)  # create directly for cleanup when it leaves scope
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    return fig


def plot_paneled_data(
    data: list[list[np.ndarray]],
    diverging: bool,
    caption: str | None = None,
):
    """Plot a list of 2D data arrays in a paneled plot."""
    if diverging:
        cmap = plt.colormaps.get_cmap("RdBu_r")
        cmap.set_bad(color=(0.7, 0.7, 0.7))
    else:
        cmap = plt.colormaps.get_cmap("viridis")
        cmap.set_bad(color="white")
    vmin = np.inf
    vmax = -np.inf
    for row in data:
        for arr in row:
            vmin = min(vmin, np.nanmin(arr))
            vmax = max(vmax, np.nanmax(arr))
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    if caption is not None:
        caption += " "
    else:
        caption = ""

    caption += f"vmin={vmin:.4g}, vmax={vmax:.4g}."

    if diverging:
        fill_value = 0.5 * (vmin + vmax)
    else:
        fill_value = vmin
    all_data = _stitch_data_panels(data, fill_value=fill_value)

    fig = plot_imshow(
        all_data, vmin=vmin, vmax=vmax, cmap=cmap, nan_padding=not diverging
    )
    wandb = WandBLogger.get_instance()
    wandb_image = wandb.Image(fig, caption=caption)
    plt.close(fig)

    # necessary to avoid CUDA error in some contexts
    # see https://github.com/ai2cm/full-model/issues/740#issuecomment-2086546187
    gc.collect()

    return wandb_image


def _stitch_data_panels(data: list[list[np.ndarray]], fill_value) -> np.ndarray:
    for row in data:
        if len(row) != len(data[0]):
            raise ValueError("All rows must have the same number of panels.")

    n_rows = len(data)
    n_cols = len(data[0])
    for row in data:
        for arr in row:
            if arr.shape != data[0][0].shape:
                raise ValueError("All panels must have the same shape.")

    stitched_data = np.full(
        (
            n_rows * data[0][0].shape[0] + n_rows - 1,
            n_cols * data[0][0].shape[1] + n_cols - 1,
        ),
        fill_value=fill_value,
    )

    # iterate over rows backwards, as the image starts in the bottom left
    # and moves upwards
    for i, row in enumerate(reversed(data)):
        for j, arr in enumerate(row):
            start_row = i * (arr.shape[0] + 1)
            end_row = start_row + arr.shape[0]
            start_col = j * (arr.shape[1] + 1)
            end_col = start_col + arr.shape[1]
            stitched_data[start_row:end_row, start_col:end_col] = arr

    return stitched_data


def _downsample_for_display(data: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Average-pool a 2D array for display if it is too large."""
    height, width = data.shape
    if height <= max_size and width <= max_size:
        return data

    out_height = min(height, max_size)
    out_width = min(width, max_size)
    row_edges = np.linspace(0, height, out_height + 1, dtype=int)
    col_edges = np.linspace(0, width, out_width + 1, dtype=int)
    pooled = np.empty((out_height, out_width), dtype=np.float32)

    for i in range(out_height):
        for j in range(out_width):
            block = data[
                row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]
            ]
            pooled[i, j] = float(block.mean())

    return pooled


def plot_attention_map(
    attn_weights: np.ndarray,
    axis: Literal["height", "width", "full"],
    caption: str | None = None,
) -> WBValue:
    wandb_logger = WandBLogger.get_instance()
    display_weights = _downsample_for_display(attn_weights)

    if display_weights.shape != attn_weights.shape:
        size_note = (
            f"Displayed as {display_weights.shape[0]}x{display_weights.shape[1]} "
            f"from original {attn_weights.shape[0]}x{attn_weights.shape[1]}."
        )
        caption = f"{caption} {size_note}" if caption else size_note

    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(display_weights, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = {
        "height": (
            "Key (latitude index)",
            "Query (latitude index)",
            "Height-axis attention",
        ),
        "width": (
            "Key (longitude index)",
            "Query (longitude index)",
            "Width-axis attention",
        ),
        "full": ("Key token index", "Query token index", "Full 2D attention"),
    }
    xlabel, ylabel, title = labels[axis]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    image = wandb_logger.Image(fig, caption=caption)
    plt.close(fig)
    gc.collect()
    return image


def plot_attention_receptive_field(
    height_weights: np.ndarray,
    width_weights: np.ndarray,
    query_lat: int,
    query_lon: int,
    caption: str | None = None,
) -> WBValue:
    wandb_logger = WandBLogger.get_instance()

    h_row = height_weights[query_lat]
    w_row = width_weights[query_lon]
    receptive_field = np.outer(h_row, w_row)

    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(receptive_field, cmap="inferno", aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")
    ax.set_title(f"Receptive field at ({query_lat}, {query_lon})")
    fig.tight_layout()

    image = wandb_logger.Image(fig, caption=caption)
    plt.close(fig)
    gc.collect()
    return image


def plot_full_attention_receptive_field(
    attn_weights: np.ndarray,
    grid_shape: tuple[int, int],
    query_lat: int,
    query_lon: int,
    caption: str | None = None,
) -> WBValue:
    height, width = grid_shape
    if not (0 <= query_lat < height and 0 <= query_lon < width):
        raise IndexError(
            f"Query ({query_lat}, {query_lon}) is outside the grid shape {grid_shape}."
        )

    query_index = query_lat * width + query_lon
    receptive_field = attn_weights[query_index].reshape(height, width)

    wandb_logger = WandBLogger.get_instance()
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(receptive_field, cmap="inferno", aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")
    ax.set_title(f"Full-attention receptive field at ({query_lat}, {query_lon})")
    fig.tight_layout()

    image = wandb_logger.Image(fig, caption=caption)
    plt.close(fig)
    gc.collect()
    return image


def plot_attention_entropy_map(
    entropy: np.ndarray,
    grid_shape: tuple[int, int] | None = None,
    caption: str | None = None,
) -> WBValue:
    """Plot per-query attention entropy as a heatmap.

    Args:
        entropy: Per-query attention entropy. Either a 1D vector or a 2D map.
        grid_shape: Optional shape used to reshape 1D entropy into a 2D map.
        caption: Optional caption for the W&B image.
    """
    entropy_map = entropy
    if entropy.ndim == 1:
        if grid_shape is None:
            entropy_map = entropy[np.newaxis, :]
        else:
            entropy_map = entropy.reshape(grid_shape)

    wandb_logger = WandBLogger.get_instance()
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(entropy_map, cmap="magma", aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Query index" if entropy_map.shape[0] == 1 else "Longitude index")
    ax.set_ylabel("Entropy" if entropy_map.shape[0] == 1 else "Latitude index")
    ax.set_title("Attention entropy")
    fig.tight_layout()

    image = wandb_logger.Image(fig, caption=caption)
    plt.close(fig)
    gc.collect()
    return image
