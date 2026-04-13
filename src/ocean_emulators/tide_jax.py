from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import numpy as np
import torch

type TideLeafKind = Literal[
    "raw_step0_prognostic",
    "raw_step0_boundary",
    "raw_step0_label",
    "raw_boundary",
    "raw_label",
]
type TideTensorKind = Literal["prognostic", "boundary"]
type TensorPlacement = Literal["cpu", "torch_device"]
type JaxBlobPlacement = Literal["cpu", "device"]
type JaxBlobPlacementPolicy = Literal["cpu", "device", "auto"]

try:
    jax: Any = importlib.import_module("jax")
    jnp: Any = importlib.import_module("jax.numpy")
    jax_core: Any = importlib.import_module("jax.extend.core")
except ImportError as exc:  # pragma: no cover - exercised when optional dep is absent
    raise RuntimeError(
        "The experimental Tide JAX frontend requires the optional dev dependency "
        "`jax`. Install it with `uv sync --dev`."
    ) from exc


@dataclasses.dataclass(frozen=True)
class TideJaxShapeSpec:
    step0_prognostic_shape: tuple[int, ...]
    step0_boundary_shape: tuple[int, ...]
    step0_label_shape: tuple[int, ...]
    boundary_shape: tuple[int, ...]
    label_shape: tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float32)


@dataclasses.dataclass(frozen=True)
class TideJaxStats:
    prognostic_mean: np.ndarray
    prognostic_std: np.ndarray
    boundary_mean: np.ndarray
    boundary_std: np.ndarray
    prognostic_mask: np.ndarray
    boundary_mask: np.ndarray
    normalize_before_mask: bool
    masked_fill_value: np.float32


@dataclasses.dataclass(frozen=True)
class TideLeaf:
    kind: TideLeafKind
    step: int
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclasses.dataclass(frozen=True)
class TideJaxBlob:
    index: int
    primitive_names: tuple[str, ...]
    input_count: int
    output_count: int


@dataclasses.dataclass(frozen=True)
class _DeferredTideLeaf:
    leaf: TideLeaf


class TideMaterializer(Protocol):
    def materialize(
        self,
        leaf: TideLeaf,
        *,
        tensor_placement: TensorPlacement = "cpu",
        jax_device: Any | None = None,
    ) -> Any: ...


@dataclasses.dataclass(frozen=True)
class TideJaxProgram:
    closed_jaxpr: Any
    out_tree: Any
    leaves: tuple[TideLeaf, ...]

    def eval(self, materializer: TideMaterializer, *args: Any) -> Any:
        env: dict[Any, Any] = {}
        jaxpr = self.closed_jaxpr.jaxpr

        def read(var: Any) -> Any:
            if type(var).__name__ == "Literal":
                return var.val
            return env[var]

        def write(var: Any, val: Any) -> None:
            if type(var).__name__ != "DropVar":
                env[var] = val

        for var, val in zip(jaxpr.constvars, self.closed_jaxpr.consts, strict=True):
            write(var, val)
        for var, val in zip(jaxpr.invars, args, strict=True):
            write(var, val)

        eqns = tuple(jaxpr.eqns)
        index = 0
        blob_index = 0
        while index < len(eqns):
            eqn = eqns[index]
            if eqn.primitive is _tide_leaf_p:
                write(eqn.outvars[0], _DeferredTideLeaf(_leaf_from_params(eqn.params)))
                index += 1
                continue

            blob_eqns = []
            while index < len(eqns) and eqns[index].primitive is not _tide_leaf_p:
                blob_eqns.append(eqns[index])
                index += 1

            blob = _blob_from_eqns(blob_index, tuple(blob_eqns))
            blob_index += 1
            _eval_jax_blob(
                blob,
                tuple(blob_eqns),
                materializer=materializer,
                read=read,
                write=write,
            )

        flat_outputs = [
            _place_output(read(var), materializer=materializer) for var in jaxpr.outvars
        ]
        return jax.tree_util.tree_unflatten(self.out_tree, flat_outputs)


class RustTrainBatchMaterializer:
    """Materialize raw Tide leaves as JAX arrays.

    `tensor_placement` controls where the Python Tide shim places torch tensors
    before conversion. `jax_device` controls the target JAX device; GPU stays
    zero-copy only when torch CUDA tensors and a compatible CUDA jaxlib are both
    present.
    """

    def __init__(
        self,
        batch: Any,
        *,
        tensor_placement: TensorPlacement | None = None,
        jax_device: Any | None = None,
        jax_blob_placement: JaxBlobPlacementPolicy | None = None,
        output_placement: JaxBlobPlacementPolicy | None = None,
    ) -> None:
        self._batch = batch
        self._jax_device = jax_device
        if jax_blob_placement is None:
            jax_blob_placement = (
                "device" if tensor_placement == "torch_device" else "cpu"
            )
        self._jax_blob_placement = jax_blob_placement
        self._output_placement = output_placement or jax_blob_placement

    def placement_for_blob(self, _blob: TideJaxBlob) -> JaxBlobPlacement:
        return _resolve_blob_placement(self._jax_blob_placement)

    def jax_device_for_blob(
        self, _blob: TideJaxBlob, placement: JaxBlobPlacement
    ) -> Any:
        return self._device_for_placement(placement)

    def output_device(self) -> Any | None:
        return self._device_for_placement(
            _resolve_blob_placement(self._output_placement)
        )

    def materialize(
        self,
        leaf: TideLeaf,
        *,
        tensor_placement: TensorPlacement = "cpu",
        jax_device: Any | None = None,
    ) -> Any:
        match leaf.kind:
            case "raw_step0_prognostic":
                tensor = self._batch.get_raw_step0_prognostic(tensor_placement)
            case "raw_step0_boundary":
                tensor = self._batch.get_raw_step0_boundary(tensor_placement)
            case "raw_step0_label":
                tensor = self._batch.get_raw_step0_label(tensor_placement)
            case "raw_boundary":
                tensor = self._batch.get_raw_boundary(leaf.step, tensor_placement)
            case "raw_label":
                tensor = self._batch.get_raw_label(leaf.step, tensor_placement)
            case _:
                raise AssertionError(f"unknown Tide leaf kind {leaf.kind!r}")
        return _torch_tensor_to_jax_array(tensor, jax_device=jax_device)

    def _device_for_placement(self, placement: JaxBlobPlacement) -> Any:
        if placement == "cpu":
            return jax.devices("cpu")[0]
        return self._jax_device or _default_device()


def trace_tide_jax(fn: Callable[..., Any], *args: Any) -> TideJaxProgram:
    closed_jaxpr, out_shape = jax.make_jaxpr(fn, return_shape=True)(*args)
    _validate_jaxpr(closed_jaxpr.jaxpr)
    return TideJaxProgram(
        closed_jaxpr=closed_jaxpr,
        out_tree=jax.tree_util.tree_structure(out_shape),
        leaves=_collect_leaves(closed_jaxpr.jaxpr.eqns),
    )


def raw_step0_prognostic(spec: TideJaxShapeSpec) -> Any:
    return _bind_leaf(
        "raw_step0_prognostic", 0, spec.step0_prognostic_shape, spec.dtype
    )


def raw_step0_boundary(spec: TideJaxShapeSpec) -> Any:
    return _bind_leaf("raw_step0_boundary", 0, spec.step0_boundary_shape, spec.dtype)


def raw_step0_label(spec: TideJaxShapeSpec) -> Any:
    return _bind_leaf("raw_step0_label", 0, spec.step0_label_shape, spec.dtype)


def raw_boundary(spec: TideJaxShapeSpec, step: int) -> Any:
    if step <= 0:
        raise ValueError("raw_boundary(step) only supports rollout steps > 0")
    return _bind_leaf("raw_boundary", step, spec.boundary_shape, spec.dtype)


def raw_label(spec: TideJaxShapeSpec, step: int) -> Any:
    if step <= 0:
        raise ValueError("raw_label(step) only supports rollout steps > 0")
    return _bind_leaf("raw_label", step, spec.label_shape, spec.dtype)


def normalize(tensor: Any, stats: TideJaxStats, kind: TideTensorKind) -> Any:
    mean, std = _stats_for_kind(stats, kind)
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


def apply_mask(tensor: Any, stats: TideJaxStats, kind: TideTensorKind) -> Any:
    channel_count = int(tensor.shape[1])
    if kind == "prognostic":
        mask = np.asarray(stats.prognostic_mask, dtype=np.bool_)
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
        mask = np.asarray(stats.boundary_mask, dtype=np.bool_)
        mask_jax = jnp.asarray(mask).reshape((1, 1, mask.shape[0], mask.shape[1]))
    else:
        raise AssertionError(f"unknown tensor kind {kind!r}")

    fill = jnp.asarray(stats.masked_fill_value, dtype=tensor.dtype)
    return jnp.where(mask_jax, tensor, fill)


def normalize_and_mask(tensor: Any, stats: TideJaxStats, kind: TideTensorKind) -> Any:
    if stats.normalize_before_mask:
        return apply_mask(normalize(tensor, stats, kind), stats, kind)
    return normalize(apply_mask(tensor, stats, kind), stats, kind)


def shape_spec_from_batch(batch: Any, *, later_step: int = 1) -> TideJaxShapeSpec:
    del later_step
    step0_prognostic_tensor, step0_boundary_tensor, step0_label_tensor = (
        batch.get_raw_step0_parts()
    )

    return TideJaxShapeSpec(
        step0_prognostic_shape=tuple(step0_prognostic_tensor.shape),
        step0_boundary_shape=tuple(step0_boundary_tensor.shape),
        step0_label_shape=tuple(step0_label_tensor.shape),
        boundary_shape=tuple(step0_boundary_tensor.shape),
        label_shape=tuple(step0_label_tensor.shape),
        dtype=np.dtype(np.float32),
    )


def stats_from_batch(batch: Any) -> TideJaxStats:
    return TideJaxStats(
        prognostic_mean=np.asarray(batch.prognostic_mean, dtype=np.float32),
        prognostic_std=np.asarray(batch.prognostic_std, dtype=np.float32),
        boundary_mean=np.asarray(batch.boundary_mean, dtype=np.float32),
        boundary_std=np.asarray(batch.boundary_std, dtype=np.float32),
        prognostic_mask=np.asarray(batch.prognostic_mask, dtype=np.bool_),
        boundary_mask=np.asarray(batch.boundary_mask, dtype=np.bool_),
        normalize_before_mask=bool(batch.normalize_before_mask),
        masked_fill_value=np.float32(batch.masked_fill_value),
    )


def _bind_leaf(
    kind: TideLeafKind, step: int, shape: Sequence[int], dtype: np.dtype
) -> Any:
    return _tide_leaf_p.bind(
        kind=kind,
        step=int(step),
        shape=tuple(int(dim) for dim in shape),
        dtype=np.dtype(dtype),
    )


def _leaf_from_params(params: dict[str, Any]) -> TideLeaf:
    return TideLeaf(
        kind=params["kind"],
        step=params["step"],
        shape=params["shape"],
        dtype=np.dtype(params["dtype"]),
    )


def _collect_leaves(eqns: Sequence[Any]) -> tuple[TideLeaf, ...]:
    leaves: list[TideLeaf] = []
    seen: set[TideLeaf] = set()
    for eqn in eqns:
        if eqn.primitive is not _tide_leaf_p:
            continue
        leaf = _leaf_from_params(eqn.params)
        if leaf not in seen:
            leaves.append(leaf)
            seen.add(leaf)
    return tuple(leaves)


def _blob_from_eqns(index: int, eqns: tuple[Any, ...]) -> TideJaxBlob:
    inputs = {
        var
        for eqn in eqns
        for var in eqn.invars
        if type(var).__name__ not in {"Literal", "DropVar"}
    }
    outputs = {
        var for eqn in eqns for var in eqn.outvars if type(var).__name__ != "DropVar"
    }
    return TideJaxBlob(
        index=index,
        primitive_names=tuple(eqn.primitive.name for eqn in eqns),
        input_count=len(inputs - outputs),
        output_count=len(outputs),
    )


def _eval_jax_blob(
    blob: TideJaxBlob,
    eqns: tuple[Any, ...],
    *,
    materializer: TideMaterializer,
    read: Callable[[Any], Any],
    write: Callable[[Any, Any], None],
) -> None:
    placement = _placement_for_blob(materializer, blob)
    device = _device_for_blob(materializer, blob, placement)
    tensor_placement = _tensor_placement_for_blob(placement)

    def read_blob(var: Any) -> Any:
        value = read(var)
        if isinstance(value, _DeferredTideLeaf):
            value = materializer.materialize(
                value.leaf,
                tensor_placement=tensor_placement,
                jax_device=device,
            )
            write(var, value)
        return _place_jax_value(value, device)

    for eqn in eqns:
        invals = [read_blob(var) for var in eqn.invars]
        result = eqn.primitive.bind(*invals, **eqn.params)
        outvals = list(result) if eqn.primitive.multiple_results else [result]
        for var, val in zip(eqn.outvars, outvals, strict=True):
            write(var, val)


def _placement_for_blob(
    materializer: TideMaterializer, blob: TideJaxBlob
) -> JaxBlobPlacement:
    fn = getattr(materializer, "placement_for_blob", None)
    if fn is None:
        return "cpu"
    return _resolve_blob_placement(fn(blob))


def _device_for_blob(
    materializer: TideMaterializer, blob: TideJaxBlob, placement: JaxBlobPlacement
) -> Any:
    fn = getattr(materializer, "jax_device_for_blob", None)
    if fn is None:
        return jax.devices("cpu")[0] if placement == "cpu" else _default_device()
    return fn(blob, placement)


def _place_output(value: Any, *, materializer: TideMaterializer) -> Any:
    if isinstance(value, _DeferredTideLeaf):
        value = materializer.materialize(value.leaf)
    fn = getattr(materializer, "output_device", None)
    if fn is None:
        return value
    return _place_jax_value(value, fn())


def _place_jax_value(value: Any, device: Any | None) -> Any:
    if device is None:
        return value
    try:
        return jax.device_put(value, device)
    except TypeError:
        return value


def _tensor_placement_for_blob(placement: JaxBlobPlacement) -> TensorPlacement:
    return "torch_device" if placement == "device" else "cpu"


def _resolve_blob_placement(policy: JaxBlobPlacementPolicy) -> JaxBlobPlacement:
    if policy in {"cpu", "device"}:
        return policy
    device = _default_device()
    platform = getattr(device, "platform", jax.default_backend())
    return "device" if platform in {"cuda", "gpu"} else "cpu"


def _default_device() -> Any:
    for backend in ("gpu", "cuda"):
        try:
            devices = jax.devices(backend)
        except RuntimeError:
            continue
        if devices:
            return devices[0]
    return jax.devices()[0]


_UNSUPPORTED_PRIMITIVES = frozenset(
    {
        "cond",
        "scan",
        "while",
        "xla_call",
        "custom_jvp_call",
        "custom_vjp_call",
    }
)


def _validate_jaxpr(jaxpr: Any) -> None:
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name in _UNSUPPORTED_PRIMITIVES:
            raise NotImplementedError(
                f"Tide JAX frontend does not support `{name}` in v0"
            )


def _stats_for_kind(
    stats: TideJaxStats, kind: TideTensorKind
) -> tuple[np.ndarray, np.ndarray]:
    match kind:
        case "prognostic":
            return (
                np.asarray(stats.prognostic_mean, dtype=np.float32),
                np.asarray(stats.prognostic_std, dtype=np.float32),
            )
        case "boundary":
            return (
                np.asarray(stats.boundary_mean, dtype=np.float32),
                np.asarray(stats.boundary_std, dtype=np.float32),
            )
        case _:
            raise AssertionError(f"unknown tensor kind {kind!r}")


def _torch_tensor_to_jax_array(
    tensor: torch.Tensor, *, jax_device: Any | None = None
) -> Any:
    array = tensor.detach()
    if _can_use_dlpack(array, jax_device):
        jax_array = jax.dlpack.from_dlpack(array)
        if jax_device is not None:
            return jax.device_put(jax_array, jax_device)
        return jax_array

    if array.device.type != "cpu":
        array = array.cpu()
    jax_array = jnp.asarray(array.numpy())
    if jax_device is not None:
        return jax.device_put(jax_array, jax_device)
    return jax_array


def _can_use_dlpack(tensor: torch.Tensor, jax_device: Any | None) -> bool:
    if tensor.device.type != "cuda":
        return False
    platform = getattr(jax_device, "platform", None)
    if platform is None:
        platform = jax.default_backend()
    return platform in {"cuda", "gpu"}


def _tide_leaf_impl(**_params: Any) -> Any:
    raise RuntimeError(
        "Tide JAX primitives are symbolic. Trace with trace_tide_jax(...) and "
        "evaluate the resulting TideJaxProgram."
    )


def _tide_leaf_abstract_eval(
    *, kind: TideLeafKind, step: int, shape: tuple[int, ...], dtype: np.dtype
) -> Any:
    del kind, step
    return jax.core.ShapedArray(shape, dtype)


_tide_leaf_p = jax_core.Primitive("tide_leaf")
_tide_leaf_p.def_impl(_tide_leaf_impl)
_tide_leaf_p.def_abstract_eval(_tide_leaf_abstract_eval)
