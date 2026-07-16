"""Domain-parallel (ShardTensor) support for Ocean_Emulator.

DDP-like philosophy: training code builds the model and data as usual, then
calls into a DomainParallelContext to (a) build the device mesh, (b) shard
inputs, (c) distribute the model, and (d) gather outputs for logging.

v1 scope (quick 2x2 test):
  - pure domain parallelism, params REPLICATED across the domain mesh
    (no FSDP; the UNet is activation-dominated with modest params)
  - a single cluster (num_replicas == 1); leader-scatter for inputs
  - GroupNorm swapped for a shard-safe implementation (fused kernel does not
    support >1 sharded spatial dim; the functional reshape path does)

Deferred (see plan): FSDP over a replica axis, multi-cluster scatter,
per-rank shard reads, variable cluster sizes, replay integration.
"""

from __future__ import annotations

import dataclasses
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import guard: fail loud if enabled without physicsnemo.
# ---------------------------------------------------------------------------
def _import_physicsnemo():
    try:
        from physicsnemo.distributed import DistributedManager, scatter_tensor
    except ImportError:  # pragma: no cover
        try:
            from physicsnemo.distributed import DistributedManager
            from physicsnemo.domain_parallel import scatter_tensor
        except Exception as exc:
            raise RuntimeError(
                "domain_parallel.enabled=true but PhysicsNeMo (ShardTensor) is not "
                "importable. Install a recent NVIDIA/physicsnemo build (post-PR #1535)."
            ) from exc
    from physicsnemo.domain_parallel import ShardTensor
    from torch.distributed.tensor import distribute_module
    from torch.distributed.tensor.placement_types import Shard, Replicate
    return (DistributedManager, scatter_tensor, ShardTensor,
            distribute_module, Shard, Replicate)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class DomainParallelConfig:
    enabled: bool = False
    # cluster_shape is the GPU grid a single sample is sharded across.
    # (2, 2) -> 2D spatial sharding (H x W). (1, 2) or (2, 1) -> 1D.
    cluster_shape: tuple[int, int] = (2, 2)
    use_fsdp: bool = False          # v1: keep False (params replicated)
    leader_scatter: bool = True     # v1: leader loads full sample, scatters
    strict_equivalence: bool = False  # assert-heavy mode for exactness tests

    @property
    def cluster_size(self) -> int:
        return self.cluster_shape[0] * self.cluster_shape[1]

    @property
    def is_2d(self) -> bool:
        return self.cluster_shape[0] > 1 and self.cluster_shape[1] > 1


# ---------------------------------------------------------------------------
# Shard-safe GroupNorm (drop-in for nn.GroupNorm)
# ---------------------------------------------------------------------------
def group_norm_manual(x, num_groups, weight, bias, eps: float = 1e-5):
    """Functional GroupNorm from reshape + var_mean + affine.

    Works on ShardTensor sharded on H and/or W: the reshape splits only the
    (unsharded) channel dim, and var_mean over the sharded spatial dims rides
    the same all-reduce path MSE uses. Matches nn.GroupNorm (biased variance).
    """
    N, C, H, W = x.shape
    G = num_groups
    x_g = x.reshape(N, G, C // G, H, W)
    var, mean = torch.var_mean(x_g, dim=(2, 3, 4), keepdim=True, unbiased=False)
    x_n = (x_g - mean) * torch.rsqrt(var + eps)
    x_n = x_n.reshape(N, C, H, W)
    return x_n * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)


class ShardSafeGroupNorm(nn.Module):
    """GroupNorm that uses the fused kernel for plain tensors and the manual
    reshape path for ShardTensors (where the fused kernel refuses >1 sharded dim).
    Parameter layout is identical to nn.GroupNorm so state_dicts are compatible.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                 affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # Lazy import so non-DP runs never import physicsnemo.
        try:
            from physicsnemo.domain_parallel import ShardTensor
            is_shard = isinstance(x, ShardTensor)
        except Exception:
            is_shard = False

        w = self.weight if self.weight is not None else torch.ones(
            self.num_channels, device=x.device, dtype=x.dtype)
        b = self.bias if self.bias is not None else torch.zeros(
            self.num_channels, device=x.device, dtype=x.dtype)

        if is_shard:
            return group_norm_manual(x, self.num_groups, w, b, self.eps)
        return F.group_norm(x, self.num_groups, w, b, self.eps)


def convert_groupnorm_(model: nn.Module) -> nn.Module:
    """In-place swap every nn.GroupNorm for ShardSafeGroupNorm, copying params."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.GroupNorm):
            repl = ShardSafeGroupNorm(
                child.num_groups, child.num_channels, child.eps,
                affine=child.affine,
            )
            if child.affine:
                repl = repl.to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    repl.weight.copy_(child.weight)
                    repl.bias.copy_(child.bias)
            setattr(model, name, repl)
        else:
            convert_groupnorm_(child)
    return model


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
class DomainParallelContext:
    """Owns the device mesh, placements, and scatter/gather/model helpers."""

    def __init__(self, config: DomainParallelConfig, dm, device: torch.device):
        self.config = config
        self.device = device
        self._dm = dm

        (_, self._scatter_tensor, self._ShardTensor,
         self._distribute_module, self._Shard, self._Replicate) = _import_physicsnemo()

        world_size = torch.distributed.get_world_size()
        cs = config.cluster_size
        if world_size % cs != 0:
            raise ValueError(
                f"world_size={world_size} not divisible by cluster_size={cs} "
                f"(cluster_shape={config.cluster_shape})."
            )
        self.num_replicas = world_size // cs
        if self.num_replicas != 1 and not config.use_fsdp:
            raise ValueError(
                "v1 supports a single cluster (num_replicas=1) without FSDP. "
                f"Got num_replicas={self.num_replicas}. Enable FSDP / multi-cluster "
                "in a later phase."
            )

        ch, cw = config.cluster_shape
        self.mesh = dm.initialize_mesh(
            mesh_shape=(self.num_replicas, ch, cw),
            mesh_dim_names=["replica", "domain_h", "domain_w"],
        )
        self.domain_mesh = self.mesh["domain_h", "domain_w"]
        self.replica_mesh = self.mesh["replica"]

        # Placements: mesh dim 0 (domain_h) shards tensor dim 2 (H);
        #             mesh dim 1 (domain_w) shards tensor dim 3 (W).
        self.input_placements = (self._Shard(2), self._Shard(3))
        # Fully-replicated placement (for scalars / per-channel things).
        self.replicate_placements = (self._Replicate(), self._Replicate())

        # Global rank at this replica's domain coordinate (0, 0) — the leader.
        self._domain_leader_global_rank = int(self.domain_mesh.mesh.flatten()[0].item())

        logger.info(
            "DomainParallelContext: cluster_shape=%s num_replicas=%s "
            "domain_leader_global_rank=%s use_fsdp=%s",
            config.cluster_shape, self.num_replicas,
            self._domain_leader_global_rank, config.use_fsdp,
        )

    # -- topology --------------------------------------------------------
    @property
    def is_domain_leader(self) -> bool:
        return torch.distributed.get_rank() == self._domain_leader_global_rank

    # -- data ------------------------------------------------------------
    def scatter(
        self,
        tensor: torch.Tensor | None,
        *,
        placements=None,
        global_shape: torch.Size | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
    ):
        """Leader-scatter a full-domain tensor to a ShardTensor.

        Only the domain leader supplies ``tensor``; other ranks pass ``None``.
        ``global_shape`` and ``dtype`` are inferred from the leader tensor when
        omitted. For real per-rank reads use ``from_local_shard`` instead.
        """
        pl = placements or self.input_placements
        if self.is_domain_leader and tensor is None:
            raise ValueError("The domain leader must provide a tensor to scatter.")

        # PhysicsNeMo's scatter_tensor chooses its metadata-broadcast path from
        # local arguments. Broadcast it here so every rank follows the same
        # collective sequence when only the leader owns the full tensor.
        mesh_group = self._dm.get_mesh_group(self.domain_mesh)
        if self.is_domain_leader:
            assert tensor is not None
            metadata = [
                (
                    global_shape if global_shape is not None else tensor.shape,
                    dtype if dtype is not None else tensor.dtype,
                )
            ]
        else:
            metadata = [None]
        torch.distributed.broadcast_object_list(
            metadata,
            src=self._domain_leader_global_rank,
            group=mesh_group,
        )
        global_shape, dtype = metadata[0]
        st = self._scatter_tensor(
            tensor, self._domain_leader_global_rank, self.domain_mesh,
            placements=pl, global_shape=global_shape, dtype=dtype,
            requires_grad=requires_grad,
        )
        return st

    def from_local_shard(self, local_tensor: torch.Tensor, *, placements=None):
        """Build a ShardTensor from each rank's own local shard (no scatter).
        Use for real training where each rank reads only its tile."""
        pl = placements or self.input_placements
        return self._ShardTensor.from_local(
            local_tensor, self.domain_mesh, pl, sharding_shapes="infer",
        )

    def gather(self, st) -> torch.Tensor:
        """Gather a ShardTensor to a full replicated torch.Tensor (logging/val)."""
        if isinstance(st, self._ShardTensor):
            return st.full_tensor()
        return st

    # -- model -----------------------------------------------------------
    @staticmethod
    def _validate_v1_model(model: nn.Module) -> None:
        """Reject model features whose global semantics are not yet implemented."""
        unsupported: list[str] = []
        if getattr(model, "pad", None) != "constant":
            unsupported.append("pad must be 'constant'")
        if getattr(model, "positional_params", None) is not None:
            unsupported.append("learned positional channels")
        if getattr(model, "add_3d_coordinates", None) is not None:
            unsupported.append("3-D coordinate channels")
        if getattr(model, "corrector", None) is not None:
            unsupported.append("post-processing correctors")
        if getattr(model, "rollout_noise_injector", None) is not None:
            unsupported.append("rollout noise")

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                unsupported.append(type(module).__name__)
            if isinstance(module, nn.ConvTranspose2d):
                unsupported.append("ConvTranspose2d")

        if unsupported:
            raise ValueError(
                "Domain-parallel v1 only supports the exactness-gate surface; "
                "remove: " + ", ".join(sorted(set(unsupported)))
            )

    def distribute_model(self, model: nn.Module) -> nn.Module:
        """Swap GroupNorm, then replicate params as DTensors over the domain mesh.
        Gradients all-reduce across the domain automatically on backward."""
        if not getattr(model, "domain_parallel", False):
            raise ValueError(
                "Domain-parallel models must be constructed with "
                "domain_parallel=True so Conv2d owns interior halo exchange."
            )
        self._validate_v1_model(model)
        convert_groupnorm_(model)
        model = model.to(self.device)
        model = self._distribute_module(model, device_mesh=self.domain_mesh)
        if self.config.use_fsdp and self.num_replicas > 1:
            raise NotImplementedError(
                "FSDP over the replica axis is a later-phase feature."
            )
        return model


# ---------------------------------------------------------------------------
# Divisibility helper
# ---------------------------------------------------------------------------
def validate_shardable(
    global_h: int, global_w: int, cluster_shape: tuple[int, int],
    *, num_downsamples: int = 4,
) -> None:
    """Assert per-shard tiles stay integer-sized and UNet-divisible at every level."""
    ch, cw = cluster_shape
    if global_h % ch or global_w % cw:
        raise ValueError(
            f"Global ({global_h}x{global_w}) not divisible by cluster {cluster_shape}."
        )
    sh, sw = global_h // ch, global_w // cw
    m = 2 ** num_downsamples
    if sh % m or sw % m:
        raise ValueError(
            f"Per-shard tile ({sh}x{sw}) not divisible by 2^{num_downsamples}={m}; "
            f"UNet downsampling would produce non-integer sizes."
        )
    logger.info("Shardable: per-shard tile %sx%s (global %sx%s, cluster %s).",
                sh, sw, global_h, global_w, cluster_shape)
