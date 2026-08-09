"""Seam-fidelity diagnostics for tiled inference.

The measurements here all answer one question: does a stitched field carry a
signature at the seam that it does not carry elsewhere? Raw curves cannot answer
that on their own, because ocean energy varies across the domain -- a curve that
rises near the seam may simply be sampling a more energetic region. So every
diagnostic is computed at the real seam *and* at matched **pseudo-seams** drawn
through undisturbed interior, and the anomaly between them is the result.

Kept out of the notebooks so the binning geometry can be tested; the notebooks
are presentation only.
"""

import dataclasses

import numpy as np

__all__ = [
    "SeamAxis",
    "SeamWindows",
    "seam_windows",
    "profile_along_axis",
    "signed_offsets",
    "derivative_jump",
    "high_k_power_ratio",
    "seam_anomaly",
    "response_by_distance",
]

SeamAxis = str  # "i" (a vertical seam) or "j" (a horizontal seam)


@dataclasses.dataclass(frozen=True)
class SeamWindows:
    """One real seam plus the matched pseudo-seams it is compared against.

    ``centres`` are canonical indices along ``axis``; ``centres[0]`` is the real
    seam. Every window spans ``+/- half_width`` about its centre and they are
    guaranteed disjoint and clear of the domain edge, so the pseudo-seams sample
    interior cells that no tile boundary and no constant padding ever touched.
    """

    axis: SeamAxis
    centres: tuple[int, ...]
    half_width: int
    #: Cells to drop along the *orthogonal* axis so the other seam cannot leak in.
    orthogonal_exclusion: tuple[int, int] | None

    @property
    def real_centre(self) -> int:
        return self.centres[0]

    @property
    def pseudo_centres(self) -> tuple[int, ...]:
        return self.centres[1:]


def seam_windows(
    *,
    axis: SeamAxis,
    seam_centre: int,
    extent: int,
    half_width: int,
    orthogonal_seam_centre: int | None = None,
    orthogonal_guard: int | None = None,
) -> SeamWindows:
    """Place a real seam window and matched pseudo-seam windows.

    This is the answer to "how long should the pseudo-seam strip be?": it does
    not need to be a strip at all. Fix an analysis half-width and place the
    pseudo-seam centres so that ``+/- half_width`` about each of them clears both
    the real seam and the domain edge. The half-width does the work that
    trimming strip lengths would otherwise have to, and because every window is
    the same size the curves are matched by construction.

    With the live geometry -- a 720-cell axis, a seam at 360, and a 16-cell
    overlap -- ``half_width=64`` places pseudo-seams at 180 and 540, sampling
    [116, 244] and [476, 604]. Neither reaches the seam at [352, 368] nor the
    padded exterior, and both are 4x the overlap width wide.
    """
    if half_width < 1:
        raise ValueError("half_width must be >= 1")
    if not 0 <= seam_centre < extent:
        raise ValueError(f"seam_centre {seam_centre} is outside [0, {extent})")

    # Midpoints of the two undisturbed halves either side of the real seam.
    candidates = (seam_centre // 2, (seam_centre + extent) // 2)
    for centre in candidates:
        if centre - half_width < 0 or centre + half_width >= extent:
            raise ValueError(
                f"half_width {half_width} is too large: a pseudo-seam at "
                f"{centre} would run off a {extent}-cell axis. Reduce it."
            )
        if abs(centre - seam_centre) <= 2 * half_width:
            raise ValueError(
                f"half_width {half_width} is too large: the pseudo-seam at "
                f"{centre} would overlap the real seam window at {seam_centre}. "
                "Reduce it."
            )
    if seam_centre - half_width < 0 or seam_centre + half_width >= extent:
        raise ValueError(
            f"half_width {half_width} runs the real seam window off the axis"
        )

    exclusion = None
    if orthogonal_seam_centre is not None:
        guard = orthogonal_guard if orthogonal_guard is not None else half_width
        exclusion = (orthogonal_seam_centre - guard, orthogonal_seam_centre + guard + 1)

    return SeamWindows(
        axis=axis,
        centres=(seam_centre, *candidates),
        half_width=half_width,
        orthogonal_exclusion=exclusion,
    )


def signed_offsets(half_width: int) -> np.ndarray:
    """The x-axis every seam curve shares: signed distance in cells."""
    return np.arange(-half_width, half_width + 1)


def profile_along_axis(
    field: np.ndarray,
    *,
    axis: SeamAxis,
    centre: int,
    half_width: int,
    orthogonal_exclusion: tuple[int, int] | None = None,
    reduce: str = "rms",
) -> np.ndarray:
    """Collapse a ``(..., j, i)`` field to a curve vs signed distance from ``centre``.

    The orthogonal exclusion is what keeps the vertical-seam curve from being
    contaminated by the horizontal seam. It is applied identically to the real
    and pseudo windows, which is what makes them comparable.
    """
    if field.ndim < 2:
        raise ValueError("field must have at least (j, i) dimensions")
    j_axis, i_axis = field.ndim - 2, field.ndim - 1
    along_axis = i_axis if axis == "i" else j_axis
    other_axis = j_axis if axis == "i" else i_axis

    if orthogonal_exclusion is not None:
        lo, hi = orthogonal_exclusion
        keep = np.ones(field.shape[other_axis], dtype=bool)
        keep[max(0, lo) : max(0, hi)] = False
        if not keep.any():
            raise ValueError("orthogonal_exclusion removed every cell")
        field = np.take(field, np.flatnonzero(keep), axis=other_axis)

    lo, hi = centre - half_width, centre + half_width + 1
    window = np.take(field, np.arange(lo, hi), axis=along_axis)

    # Move the along-seam axis last, then reduce over everything before it.
    window = np.moveaxis(window, along_axis, -1)
    flat = window.reshape(-1, window.shape[-1])
    if reduce == "rms":
        return np.sqrt(np.nanmean(flat**2, axis=0))
    if reduce == "mean":
        return np.nanmean(flat, axis=0)
    if reduce == "absmean":
        return np.nanmean(np.abs(flat), axis=0)
    raise ValueError(f"Unknown reduce {reduce!r}")


def derivative_jump(field: np.ndarray, *, axis: SeamAxis) -> np.ndarray:
    """First difference across ``axis``, the quantity a hard stitch spikes.

    A kink at the seam is a delta function in the derivative, so this is the
    most direct seam signature there is -- more direct than RMSE, which mixes
    the seam in with ordinary forecast error.

    Two things follow from that and both matter when reducing the result:

    * The output is a forward difference padded on the right, so element ``k``
      describes the interface between ``k`` and ``k+1``. The signature therefore
      sits half a cell to the low side of the discontinuity.
    * A delta function must be reduced with a **max**, not a mean. Averaging a
      single spike over a window divides it by the window width, which makes a
      hard stitch look *better* than a smooth blend that spreads the same total
      variation over many cells -- the exact opposite of the truth.
    """
    j_axis, i_axis = field.ndim - 2, field.ndim - 1
    diff_axis = i_axis if axis == "i" else j_axis
    difference = np.diff(field, axis=diff_axis)
    # Pad back to the input width so offsets stay aligned with the field.
    pad = [(0, 0)] * field.ndim
    pad[diff_axis] = (0, 1)
    return np.pad(difference, pad, mode="edge")


def high_k_power_ratio(
    field: np.ndarray,
    reference: np.ndarray,
    *,
    axis: SeamAxis,
    high_k_fraction: float = 0.5,
) -> np.ndarray:
    """Per-line ratio of high-wavenumber power to a reference's.

    Averaging decorrelated small scales destroys them, so a blend that is too
    broad shows up as a *deficit* here: values below 1 near the seam mean the
    stitch has eaten variance the truth still has. Computed line by line across
    the seam so it can be binned by distance like everything else.
    """
    if not 0 < high_k_fraction < 1:
        raise ValueError("high_k_fraction must be in (0, 1)")
    j_axis, i_axis = field.ndim - 2, field.ndim - 1
    line_axis = j_axis if axis == "i" else i_axis

    def band_power(values: np.ndarray) -> np.ndarray:
        moved = np.moveaxis(values, line_axis, -1)
        detrended = moved - moved.mean(axis=-1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(detrended, axis=-1)) ** 2
        cut = int(spectrum.shape[-1] * (1.0 - high_k_fraction))
        return spectrum[..., cut:].sum(axis=-1)

    numerator = band_power(field)
    denominator = band_power(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator > 0, numerator / denominator, np.nan)
    # band_power dropped the line axis; put a length-1 axis back so the result
    # still broadcasts against (..., j, i) consumers.
    return np.expand_dims(ratio, axis=line_axis)


def seam_anomaly(
    field: np.ndarray,
    windows: SeamWindows,
    *,
    reduce: str = "rms",
) -> dict[str, np.ndarray]:
    """Real-seam curve, the mean pseudo-seam curve, and their difference.

    The anomaly is the actual result: it is what remains after the ordinary
    spatial variation that both windows share has been subtracted out.
    """
    real = profile_along_axis(
        field,
        axis=windows.axis,
        centre=windows.real_centre,
        half_width=windows.half_width,
        orthogonal_exclusion=windows.orthogonal_exclusion,
        reduce=reduce,
    )
    pseudo = np.stack(
        [
            profile_along_axis(
                field,
                axis=windows.axis,
                centre=centre,
                half_width=windows.half_width,
                orthogonal_exclusion=windows.orthogonal_exclusion,
                reduce=reduce,
            )
            for centre in windows.pseudo_centres
        ]
    )
    pseudo_mean = pseudo.mean(axis=0)
    return {
        "offsets": signed_offsets(windows.half_width),
        "seam": real,
        "pseudo": pseudo_mean,
        "pseudo_all": pseudo,
        "anomaly": real - pseudo_mean,
    }


def response_by_distance(
    response: np.ndarray,
    *,
    centre: tuple[int, int],
    num_bins: int = 32,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a perturbation response ``(..., j, i)`` by distance from ``centre``.

    This is the far-field test's readout. Receptive-field coupling decays with
    distance; GroupNorm couples every cell to the whole tile through shared
    spatial statistics, so it shows up instead as a distance-independent floor.
    The shape of this curve is what separates the two.
    """
    height, width = response.shape[-2:]
    j = np.arange(height)[:, None] - centre[0]
    i = np.arange(width)[None, :] - centre[1]
    distance = np.sqrt(j.astype(float) ** 2 + i.astype(float) ** 2)

    limit = float(distance.max()) if max_distance is None else float(max_distance)
    edges = np.linspace(0.0, limit, num_bins + 1)
    index = np.clip(np.digitize(distance.ravel(), edges) - 1, 0, num_bins - 1)

    flat = response.reshape(-1, height * width)
    binned = np.full((flat.shape[0], num_bins), np.nan)
    for b in range(num_bins):
        selected = index == b
        if selected.any():
            binned[:, b] = np.sqrt(np.nanmean(flat[:, selected] ** 2, axis=1))

    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, binned.reshape(*response.shape[:-2], num_bins)
