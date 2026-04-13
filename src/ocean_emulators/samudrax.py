from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Literal, Self

import numpy as np

type TensorKind = Literal["prognostic", "boundary"]

try:
    jnp: Any = importlib.import_module("jax.numpy")
except ImportError as exc:  # pragma: no cover - exercised when optional dep is absent
    raise RuntimeError(
        "The experimental Samudra JAX helpers require the optional dev dependency "
        "`jax`. Install it with `uv sync --dev`."
    ) from exc


@dataclasses.dataclass(frozen=True)
class DataSource:
    prognostic_mean: np.ndarray
    prognostic_std: np.ndarray
    boundary_mean: np.ndarray
    boundary_std: np.ndarray
    prognostic_mask: np.ndarray
    boundary_mask: np.ndarray
    normalize_before_mask: bool
    masked_fill_value: np.float32

    @classmethod
    def from_train_dataset(cls, dataset: Any) -> Self:
        return cls(
            prognostic_mean=_flatten_xarray_dataset(dataset.prognostic_srcs[0].means),
            prognostic_std=_flatten_xarray_dataset(dataset.prognostic_srcs[0].stds),
            boundary_mean=_flatten_xarray_dataset(dataset.boundary_src.means),
            boundary_std=_flatten_xarray_dataset(dataset.boundary_src.stds),
            prognostic_mask=np.asarray(
                dataset.wet_prognostic[0].cpu().numpy(), dtype=np.bool_
            ),
            boundary_mask=np.asarray(dataset.wet_surface.cpu().numpy(), dtype=np.bool_),
            normalize_before_mask=bool(dataset.normalize_before_mask),
            masked_fill_value=np.float32(dataset.masked_fill_value),
        )

    def crop_spatial(
        self, lat_start: int, lat_end: int, lon_start: int, lon_end: int
    ) -> Self:
        return dataclasses.replace(
            self,
            prognostic_mask=self.prognostic_mask[
                :, lat_start:lat_end, lon_start:lon_end
            ],
            boundary_mask=self.boundary_mask[lat_start:lat_end, lon_start:lon_end],
        )


def normalize(tensor: Any, source: DataSource, kind: TensorKind) -> Any:
    mean, std = _normalization_for_kind(source, kind)
    channel_count = int(tensor.shape[1])
    if channel_count % mean.shape[0] != 0:
        raise ValueError(
            f"{kind} channel count {channel_count} must be divisible by "
            f"statistics length {mean.shape[0]}"
        )
    stat_index = jnp.arange(channel_count) % mean.shape[0]
    mean_jax = jnp.asarray(mean, dtype=tensor.dtype)[stat_index].reshape(
        (1, channel_count, 1, 1)
    )
    std_jax = jnp.asarray(std, dtype=tensor.dtype)[stat_index].reshape(
        (1, channel_count, 1, 1)
    )
    normalized = (tensor - mean_jax) / std_jax
    return jnp.where(jnp.isnan(normalized), jnp.asarray(0.0, tensor.dtype), normalized)


def apply_mask(tensor: Any, source: DataSource, kind: TensorKind) -> Any:
    channel_count = int(tensor.shape[1])
    if kind == "prognostic":
        mask = np.asarray(source.prognostic_mask, dtype=np.bool_)
        if channel_count % mask.shape[0] != 0:
            raise ValueError(
                f"prognostic channel count {channel_count} must be divisible by "
                f"mask channels {mask.shape[0]}"
            )
        mask_index = jnp.arange(channel_count) % mask.shape[0]
        mask_jax = jnp.asarray(mask)[mask_index].reshape(
            (1, channel_count, mask.shape[1], mask.shape[2])
        )
    elif kind == "boundary":
        mask = np.asarray(source.boundary_mask, dtype=np.bool_)
        mask_jax = jnp.asarray(mask).reshape((1, 1, mask.shape[0], mask.shape[1]))
    else:
        raise AssertionError(f"unknown tensor kind {kind!r}")

    fill = jnp.asarray(source.masked_fill_value, dtype=tensor.dtype)
    return jnp.where(mask_jax, tensor, fill)


def normalize_and_mask(tensor: Any, source: DataSource, kind: TensorKind) -> Any:
    if source.normalize_before_mask:
        return apply_mask(normalize(tensor, source, kind), source, kind)
    return normalize(apply_mask(tensor, source, kind), source, kind)


def _flatten_xarray_dataset(dataset: Any) -> np.ndarray:
    return dataset.to_array().to_numpy().reshape(-1).astype(np.float32, copy=False)


def _normalization_for_kind(
    source: DataSource, kind: TensorKind
) -> tuple[np.ndarray, np.ndarray]:
    match kind:
        case "prognostic":
            return (
                np.asarray(source.prognostic_mean, dtype=np.float32),
                np.asarray(source.prognostic_std, dtype=np.float32),
            )
        case "boundary":
            return (
                np.asarray(source.boundary_mean, dtype=np.float32),
                np.asarray(source.boundary_std, dtype=np.float32),
            )
        case _:
            raise AssertionError(f"unknown tensor kind {kind!r}")
