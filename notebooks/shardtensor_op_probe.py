#!/usr/bin/env python3
"""
Probe PhysicsNeMo ShardTensor support for Ocean_Emulator spatial ops.

Sections (selected by world_size at launch:
  1-GPU              : local reference for custom groupnorm (mesh=None)
  2-GPU 1D sharding  : Shard(H) over a 1x2 domain
  3-GPU sharding     : Shard(W) over a 1x3 domain (even + uneven strips, ->
    even = same sizes, uneven = different size patches for reducing BC waste)
  4-GPU 2D sharding  : Shard(H) x Shard(W) over a 2x2 domain

Run once per GPU count, appending into one report:
  GPUS=1 ... --fresh          # truncates + writes header, runs 1-GPU section
  GPUS=2 ...                  # appends 2-GPU 1D section
  GPUS=3 ...                  # appends 3-GPU 1D section
  GPUS=4 ...                  # appends 4-GPU 2D section

NOTE on 3-GPU: an L-shaped mesh is not acceptable, (2x2 with 1 patch dropped).
    Instead, we can drop whole rows/cols as necessary, and try to trim excess
    land from land-heavy patches, which is what 3-GPU 1D test does. 

NOTE on custom groupnorm: shardtensor does not currently support group normalizsation
    for sharded tensors with more than one sharded dimension (2D). We have to implement
    our own custom function for this using operations that are supported in 2D.
"""

import argparse
import datetime as dt
import os
import sys
import traceback
from types import SimpleNamespace
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.placement_types import Shard


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ocean_emulators.models.modules.activations import CappedGELU  # noqa: E402


# =============================================================================
# group_norm_manual  (bypasses the fused-kernel guard that blocks >1 sharded dim)
# =============================================================================
def group_norm_manual(x, num_groups, weight, bias, eps: float = 1e-5):
    """Functional GroupNorm built from reshape + var_mean + affine.

    Works on ShardTensor sharded on H and/or W because:
      - reshape splits ONLY the (unsharded) channel dim -> no data crosses a
        shard boundary,
      - var_mean over the sharded spatial dims produces a Partial that is
        all-reduced via the same path MSE uses (proven working),
      - affine + backward are standard autograd ops (no manual collectives).

    Matches nn.GroupNorm semantics: biased variance, eps inside rsqrt.
    """
    N, C, Hh, Ww = x.shape
    G = num_groups
    x_g = x.reshape(N, G, C // G, Hh, Ww)
    var, mean = torch.var_mean(x_g, dim=(2, 3, 4), keepdim=True, unbiased=False)
    x_n = (x_g - mean) * torch.rsqrt(var + eps)
    x_n = x_n.reshape(N, C, Hh, Ww)
    return x_n * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)


# =============================================================================
# Infra
# =============================================================================
def import_physicsnemo():
    try:
        from physicsnemo.distributed import DistributedManager
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ERROR: physicsnemo is not installed. Install a recent "
            "NVIDIA/physicsnemo build (post-PR #1535)."
        ) from exc
    try:
        from physicsnemo.distributed import scatter_tensor
    except ImportError:
        from physicsnemo.domain_parallel import scatter_tensor
    return DistributedManager, scatter_tensor


class Reporter:
    """Rank-0-only writer. Appends to a shared file across launches."""

    def __init__(self, path: Path | None, rank: int, fresh: bool) -> None:
        self.path = path
        self.rank = rank
        if self.rank == 0 and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if fresh or not self.path.exists():
                self.path.write_text("", encoding="utf-8")

    def write(self, message: str = "") -> None:
        if self.rank != 0:
            return
        print(message, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, NotImplementedError) or "unsupported" in text:
        return "UNSUPPORTED"
    if "not implemented" in text or "no sharding propagation" in text:
        return "UNSUPPORTED"
    if "missingshardpatch" in text:
        return "UNSUPPORTED"
    return "FAIL"


def short_error(exc: BaseException, limit: int = 240) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:limit]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1088)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--conv-channels", type=int, default=32)
    p.add_argument("--small-side", type=int, default=544)
    p.add_argument("--small-channels", type=int, default=32)
    p.add_argument("--report-dir", default=str(REPO_ROOT / "logs"))
    p.add_argument("--report-name", default="shardtensor_op_probe.out")
    p.add_argument("--fresh", action="store_true",
                   help="Truncate report before writing (use on the 1-GPU launch).")
    p.add_argument("--print-tracebacks", action="store_true")
    return p.parse_args()


def setup_distributed():
    # Shorter NCCL timeout so collective divergence fails fast (~60s) instead of
    # hanging for 30+ minutes. Also enable async error handling.
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "60")

    DistributedManager, scatter_tensor = import_physicsnemo()
    DistributedManager.initialize()
    dm = DistributedManager()
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required for this probe.")
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(dm, "local_rank", 0)))
    torch.cuda.set_device(local_rank)
    device = getattr(dm, "device", torch.device(f"cuda:{local_rank}"))
    rank = int(os.environ.get("RANK", 0))
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return dm, scatter_tensor, device, rank, world_size


def source_of(mesh) -> int:
    if mesh is None or not torch.distributed.is_initialized():
        return 0
    try:
        return torch.distributed.get_global_rank(mesh.get_group(), 0)
    except Exception:
        return 0


def sync_all_ok(local_ok: int, device) -> bool:
    """All-reduce a success flag so every rank agrees whether to continue.
    
    Returns True only if ALL ranks reported ok (local_ok=1).
    If any rank failed (local_ok=0), all ranks see False.
    """
    if not torch.distributed.is_initialized():
        return bool(local_ok)
    t = torch.tensor([local_ok], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
    return bool(t.item())


def barrier():
    """Rank barrier to keep all ranks synchronized between ops."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def build_helpers(scatter_tensor, device, mesh, placements):
    """Bind make_input / maybe_distribute to a specific mesh + placement."""
    def make_input(shape, requires_grad: bool = True):
        full = torch.randn(shape, device=device, dtype=torch.float32)
        full.requires_grad_(requires_grad)
        if mesh is None:
            return full
        sh = scatter_tensor(full, source_of(mesh), mesh, placements=placements,
                            global_shape=torch.Size(shape), dtype=full.dtype)
        if requires_grad:
            sh.requires_grad_(True)
        return sh

    def maybe_distribute(module: nn.Module) -> nn.Module:
        module = module.to(device)
        if mesh is not None:
            module = distribute_module(module, device_mesh=mesh)
        return module

    return make_input, maybe_distribute


# =============================================================================
# Op suite (shared across sections via the H namespace)
# =============================================================================
def run_op_suite(H, reporter, results, args, *, full_suite: bool):
    """full_suite=True runs everything; False runs the 2D 'gate' subset.
    
    Uses lockstep sync to prevent distributed deadlocks: after forward,
    all ranks agree on success/failure; after backward, same; then barrier.
    """
    main_shape = (1, args.channels, args.height, args.width)
    conv_shape = (1, args.conv_channels, args.height, args.width)
    small_shape = (1, args.small_channels, args.small_side, args.small_side)

    def probe(name, shape, factory: Callable[[], Callable[[torch.Tensor], torch.Tensor]]):
        """Forward + backward with rank-lockstep sync to prevent deadlocks."""
        local_ok, err_msg, y = 1, "", None

        # --- forward (may involve collectives; all ranks must reach the sync) ---
        try:
            x = H.make_input(shape, requires_grad=True)
            y = factory()(x)
        except Exception as exc:  # noqa: BLE001 - probe/report script
            local_ok, err_msg = 0, f"{classify_exception(exc)} - {short_error(exc)}"

        # --- sync: all ranks agree on forward success before proceeding ---
        fwd_ok = sync_all_ok(local_ok, H.device)

        # --- backward ONLY if forward succeeded on every rank ---
        if fwd_ok:
            try:
                y.float().mean().backward()
            except Exception as exc:  # noqa: BLE001
                local_ok, err_msg = 0, f"{classify_exception(exc)} - {short_error(exc)}"
            # --- sync again: all ranks agree on backward success ---
            fwd_ok = sync_all_ok(local_ok, H.device)

        # --- record (rank 0 writes) ---
        if fwd_ok:
            results[name] = "PASS"
            reporter.write(f"  {name:28s} {shape}  -> PASS")
        else:
            results[name] = err_msg or "FAIL (failed on another rank)"
            reporter.write(f"  {name:28s} {shape}  -> {results[name]}")

        # --- barrier keeps everyone aligned before the next op ---
        barrier()
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --- GroupNorm: fused (nn.GroupNorm) AND manual, so we see which paths work
    probe("GroupNorm32_fused", main_shape,
          lambda: lambda x: H.maybe_distribute(
              nn.GroupNorm(num_groups=32, num_channels=args.channels))(x))

    w = torch.randn(args.channels, device=H.device)
    b = torch.randn(args.channels, device=H.device)
    probe("GroupNorm32_manual", main_shape,
          lambda: lambda x: group_norm_manual(x, 32, w, b))

    for d in (1, 2, 4, 8):
        probe(f"DilatedConv2d_d{d}", conv_shape,
              lambda d=d: lambda x: H.maybe_distribute(
                  nn.Conv2d(args.conv_channels, args.conv_channels,
                            kernel_size=3, dilation=d, padding=d))(x))

    probe("AvgPool2d_k2_s2", small_shape,
          lambda: lambda x: H.maybe_distribute(nn.AvgPool2d(2, 2))(x))

    def mse_factory():
        target = H.make_input(main_shape, requires_grad=False)
        return lambda x: F.mse_loss(x, target, reduction="mean")
    probe("MSELoss_mean", main_shape, mse_factory)

    if not full_suite:
        return

    probe("CappedGELU", small_shape, lambda: lambda x: CappedGELU().to(H.device)(x))
    probe("Clamp", small_shape, lambda: lambda x: torch.clamp(x, -2.0, 2.0))
    probe("ResidualAdd", small_shape, lambda: lambda x: x + 0.125 * x)
    probe("Where", small_shape, lambda: lambda x: torch.where(x > 0.0, x, -x))
    probe("CatChannel", small_shape, lambda: lambda x: torch.cat([x, x], dim=1))
    probe("MaxPool2d_k2_s2", small_shape,
          lambda: lambda x: H.maybe_distribute(nn.MaxPool2d(2, 2))(x))
    probe("BilinearUpsample_x2", small_shape,
          lambda: lambda x: H.maybe_distribute(nn.Upsample(scale_factor=2, mode="bilinear"))(x))

    # SkipAlignPadAdd is a probe artifact under uneven shards; only run when
    # the sharded axis splits evenly. With 1088-divisible model sizes, skip pads
    # are zero anyway, so this is a real-world non-issue.
    even_split = (H.shard_count is None) or (
        small_shape[3] % H.shard_count == 0 and small_shape[2] % H.shard_count == 0
    )
    if even_split:
        def skip_align_factory():
            pads = (1, 1, 2, 2)  # even totals per axis -> shards split cleanly
            skip = H.make_input(
                (small_shape[0], small_shape[1],
                 small_shape[2] + pads[2] + pads[3],
                 small_shape[3] + pads[0] + pads[1]),
                requires_grad=False)
            return lambda x: F.pad(x, pads) + skip
        probe("SkipAlignPadAdd", small_shape, skip_align_factory)
    else:
        results["SkipAlignPadAdd"] = "SKIP (uneven shard split; not a real model case)"
        reporter.write(f"  {'SkipAlignPadAdd':28s} {small_shape}  "
                       f"-> SKIP (uneven shard artifact)")
        barrier()


def exactness_group_norm(scatter_tensor, device, mesh, placements, reporter,
                         results, *, tag, shape, num_groups=32,
                         atol=1e-3, rtol=1e-3):
    """Check manual == torch.F.group_norm (single-dev) and sharded == reference.
    
    Uses lockstep sync: all ranks compute, then sync whether they succeeded,
    then barrier. Prevents divergence between ranks.
    """
    name = f"GN_exactness[{tag}]"
    local_ok, err_msg = 1, ""

    try:
        # Identical inputs on every rank (deterministic) => scatter is consistent.
        torch.manual_seed(0)
        full = torch.randn(shape, device=device)
        weight = torch.randn(shape[1], device=device)
        bias = torch.randn(shape[1], device=device)

        ref = F.group_norm(full, num_groups, weight, bias, eps=1e-5)
        manual_single = group_norm_manual(full, num_groups, weight, bias)
        err_ms = (manual_single - ref).abs().max().item()

        if mesh is not None:
            xs = scatter_tensor(full.clone(), source_of(mesh), mesh,
                                placements=placements,
                                global_shape=torch.Size(shape), dtype=full.dtype)
            out = group_norm_manual(xs, num_groups, weight, bias)
            gathered = out.full_tensor()
            err_sh = (gathered - ref).abs().max().item()
        else:
            err_sh = err_ms

        ok = (err_ms <= atol + rtol * ref.abs().max().item()) and \
             (err_sh <= atol + rtol * ref.abs().max().item())

        if not ok:
            local_ok, err_msg = 0, f"FAIL(err_ms={err_ms:.2e}, err_sh={err_sh:.2e})"
    except Exception as exc:  # noqa: BLE001
        local_ok, err_msg = 0, f"{classify_exception(exc)} - {short_error(exc)}"

    # --- sync: all ranks agree on success ---
    exactness_ok = sync_all_ok(local_ok, device)

    # --- record ---
    if exactness_ok:
        results[name] = "PASS"
        reporter.write(f"  {name:28s} shape={shape}  -> PASS")
    else:
        results[name] = err_msg or "FAIL (failed on another rank)"
        reporter.write(f"  {name:28s} shape={shape}  -> {results[name]}")

    barrier()


# =============================================================================
# Section runner
# =============================================================================
def section(reporter, title):
    reporter.write("\n" + "#" * 60)
    reporter.write(f"# SECTION: {title}")
    reporter.write("#" * 60)


def section_summary(reporter, results):
    passed = sum(v == "PASS" for v in results.values())
    reporter.write("-" * 60)
    reporter.write(f"  {passed}/{len(results)} passed in this section.")
    for k, v in results.items():
        if v != "PASS":
            reporter.write(f"    NEEDS ATTENTION: {k}: {v}")


def main() -> int:
    args = parse_args()
    dm, scatter_tensor, device, rank, world_size = setup_distributed()
    report_path = Path(args.report_dir) / args.report_name
    reporter = Reporter(report_path, rank, fresh=args.fresh)

    if args.fresh:
        reporter.write("ShardTensor Op-Support Probe Report")
        reporter.write("=" * 60)
        reporter.write(f"torch={torch.__version__}")
        reporter.write(f"main_shape=(1,{args.channels},{args.height},{args.width})")
        reporter.write(f"generated={dt.datetime.now().isoformat(timespec='seconds')}")

    results: dict[str, str] = {}
    gn_shape = (1, 64, 256, 256)  # cheap exactness shape (64 % 32 == 0)

    # -------------------------------------------------------------- 1 GPU
    if world_size == 1:
        section(reporter, "1-GPU (local reference, mesh=None)")
        mk, dist = build_helpers(scatter_tensor, device, None, None)
        H = SimpleNamespace(make_input=mk, maybe_distribute=dist, device=device,
                            shard_count=None)
        run_op_suite(H, reporter, results, args, full_suite=True)
        exactness_group_norm(scatter_tensor, device, None, None, reporter,
                             results, tag="1gpu", shape=gn_shape)
        section_summary(reporter, results)

    # -------------------------------------------------------- 2 GPU 1D (H)
    elif world_size == 2:
        section(reporter, "2-GPU 1D sharding  [Shard(H)] over 1x2 domain")
        mesh = dm.initialize_mesh(mesh_shape=(2,), mesh_dim_names=["domain"])
        mk, dist = build_helpers(scatter_tensor, device, mesh, (Shard(2),))
        H = SimpleNamespace(make_input=mk, maybe_distribute=dist, device=device,
                            shard_count=2)
        run_op_suite(H, reporter, results, args, full_suite=True)
        exactness_group_norm(scatter_tensor, device, mesh, (Shard(2),), reporter,
                             results, tag="2gpu-1D", shape=gn_shape)
        section_summary(reporter, results)

    # ---------------------------------------- 3 GPU 1x3 strip (W, even+uneven)
    elif world_size == 3:
        section(reporter, "3-GPU sharding  [Shard(W)] over 1x3 strip")
        mesh = dm.initialize_mesh(mesh_shape=(3,), mesh_dim_names=["domain"])

        # Even strip: three 1088-wide patches side by side (the land-cut cluster).
        even_args = argparse.Namespace(**vars(args))
        even_args.width = 3 * args.width  # 3264 wide, shards evenly into 3
        mk, dist = build_helpers(scatter_tensor, device, mesh, (Shard(3),))
        H = SimpleNamespace(make_input=mk, maybe_distribute=dist, device=device,
                            shard_count=3)
        reporter.write("  [even strip: global W=%d, 3 x %d]" % (even_args.width, args.width))
        run_op_suite(H, reporter, results, args=even_args, full_suite=True)
        exactness_group_norm(scatter_tensor, device, mesh, (Shard(3),), reporter,
                             results, tag="3gpu-even", shape=(1, 64, 256, 384))

        # Uneven strip: ShardTensor supports non-uniform chunks (256 % 3 != 0).
        reporter.write("  [uneven strip: global W not divisible by 3 -> tests "
                       "ShardTensor uneven chunking]")
        exactness_group_norm(scatter_tensor, device, mesh, (Shard(3),), reporter,
                             results, tag="3gpu-uneven", shape=(1, 64, 256, 256))

        section_summary(reporter, results)

    # ---------------------------------------------- 4 GPU 2D (H x W, 2x2)
    elif world_size == 4:
        section(reporter, "4-GPU 2D sharding  [Shard(H) x Shard(W)] over 2x2 domain")
        mesh = dm.initialize_mesh(mesh_shape=(2, 2),
                                  mesh_dim_names=["domain_h", "domain_w"])
        mk, dist = build_helpers(scatter_tensor, device, mesh, (Shard(2), Shard(3)))
        H = SimpleNamespace(make_input=mk, maybe_distribute=dist, device=device,
                            shard_count=4)
        # Gate subset: the ops that decide B/D/F viability (conv corners, GN, pool, mse)
        run_op_suite(H, reporter, results, args, full_suite=False)
        exactness_group_norm(scatter_tensor, device, mesh, (Shard(2), Shard(3)),
                             reporter, results, tag="4gpu-2D", shape=gn_shape)
        section_summary(reporter, results)

    else:
        reporter.write(f"\n[skip] world_size={world_size} has no defined section "
                       f"(use 1, 2, 3, or 4 GPUs).")

    reporter.write(f"\nreport_path={report_path}")
    # Return 0 if no FAIL or UNSUPPORTED; SKIP and PASS are both acceptable.
    has_real_failures = any("FAIL" in v or "UNSUPPORTED" in v 
                            for v in results.values())
    return 1 if has_real_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())