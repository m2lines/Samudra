from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Literal

import torch

ActivationLayout = Literal["lon", "data_lon"]

try:
    from physicsnemo.distributed.manager import DistributedManager
    from physicsnemo.distributed.shard_tensor import (
        DeviceMesh,
        Replicate,
        Shard,
        ShardTensor,
    )
except Exception:  # pragma: no cover - optional dependency may be missing locally
    DistributedManager = None
    DeviceMesh = None
    Replicate = None
    Shard = None
    ShardTensor = None


def _require_physicsnemo() -> None:
    """Fail with a clear error when the optional physicsnemo dependency is absent."""
    if DistributedManager is None or DeviceMesh is None:
        raise ImportError(
            "physicsnemo is required for model sharding. "
            "Install the physicsnemo extra to enable distributed sharding."
        )


def create_device_mesh(mesh_shape: dict[str, int]) -> DeviceMesh:
    """Create (or reuse) a global DeviceMesh from the PhysicsNeMo DistributedManager."""
    _require_physicsnemo()
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    manager = DistributedManager()
    mesh_dim_names: tuple[str, ...]
    mesh_dims: tuple[int, ...]
    if isinstance(mesh_shape, dict):
        ordered: OrderedDict[str, int] = OrderedDict(mesh_shape)
        mesh_dim_names = tuple(ordered.keys())
        mesh_dims = tuple(ordered.values())
    else:
        raise TypeError("mesh_shape must be a dict[str, int]")

    manager.initialize_mesh(mesh_shape=mesh_dims, mesh_dim_names=mesh_dim_names)
    return manager.global_mesh


def make_activation_placements(
    mesh: DeviceMesh, layout: ActivationLayout
) -> tuple[Shard | Replicate, ...]:
    _require_physicsnemo()
    placements: list[Shard | Replicate] = [Replicate() for _ in range(mesh.ndim)]
    axis_names: Iterable[str] = getattr(mesh, "axis_names", range(mesh.ndim))
    for axis_idx, axis_name in enumerate(axis_names):
        if layout == "data_lon" and axis_name == "data":
            placements[axis_idx] = Shard(0)
        if axis_name == "lon":
            placements[axis_idx] = Shard(3)
    return tuple(placements)


def shard_activations(
    tensor: torch.Tensor, mesh: DeviceMesh, layout: ActivationLayout
) -> ShardTensor:
    _require_physicsnemo()
    placements = make_activation_placements(mesh, layout)
    return ShardTensor.from_local(tensor, device_mesh=mesh, placements=placements)


def to_replicated(tensor: torch.Tensor) -> torch.Tensor:
    if ShardTensor is not None and isinstance(tensor, ShardTensor):
        return tensor.full_tensor()
    return tensor


def shard_pad(
    tensor: torch.Tensor, n_pad: int, lon_mode: str, constant: float = 0.0
) -> torch.Tensor:
    """Pad latitude with constants and longitude with circular wrap for ShardTensor."""
    if n_pad == 0:
        return tensor

    # Latitude (dim=2) constant pad
    tensor = torch.nn.functional.pad(
        tensor, (0, 0, n_pad, n_pad), mode="constant", value=constant
    )

    # Longitude (dim=3) circular pad
    if lon_mode == "circular":
        left = tensor[..., -n_pad:]
        right = tensor[..., :n_pad]
        tensor = torch.cat([left, tensor, right], dim=3)
    else:
        tensor = torch.nn.functional.pad(
            tensor, (n_pad, n_pad, 0, 0), mode=lon_mode, value=constant
        )
    return tensor
