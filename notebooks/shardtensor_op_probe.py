#!/usr/bin/env python3
"""Probe PhysicsNeMo ShardTensor support for Ocean_Emulator spatial ops.

Runs two suites:
  1. Full 1D spatial sharding (Shard along H) — works with 2 or 4 GPUs.
  2. Gate-only 2D spatial sharding (Shard along H AND W, 2x2 mesh) — requires
     exactly 4 GPUs. This is the go/no-go test for row types B/D/F, which need
     square global domains and therefore 8-neighbor (corner) conv halo exchange.
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


def import_physicsnemo():
    try:
        from physicsnemo.distributed import DistributedManager
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ERROR: physicsnemo is not installed.\n"
            "Install a recent PhysicsNeMo build (post-PR #1535) — e.g. install "
            "NVIDIA/physicsnemo from a recent GitHub commit."
        ) from exc

    try:
        from physicsnemo.distributed import scatter_tensor
    except ImportError:
        from physicsnemo.domain_parallel import scatter_tensor

    return DistributedManager, scatter_tensor


class Reporter:
    def __init__(self, path: Path | None, rank: int) -> None:
        self.path = path
        self.rank = rank
        if self.rank == 0 and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        if self.rank != 0:
            return
        print(message, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if isinstance(exc, NotImplementedError) or "unsupported" in lowered:
        return "UNSUPPORTED"
    if "not implemented" in lowered or "no sharding propagation" in lowered:
        return "UNSUPPORTED"
    if "missingshardpatch" in lowered:
        return "UNSUPPORTED"
    return "FAIL"


def short_error(exc: BaseException, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--small-side", type=int, default=544)
    parser.add_argument("--small-channels", type=int, default=32)
    parser.add_argument("--shard-dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "logs"))
    parser.add_argument("--report-name", default="")
    parser.add_argument("--print-tracebacks", action="store_true")
    parser.add_argument(
        "--skip-2d", action="store_true",
        help="Skip the 4-GPU 2D spatial sharding gate suite even if 4 GPUs present.",
    )
    return parser.parse_args()


def setup_distributed():
    DistributedManager, scatter_tensor = import_physicsnemo()
    DistributedManager.initialize()
    dm = DistributedManager()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required for this ShardTensor probe.")

    local_rank = int(os.environ.get("LOCAL_RANK", getattr(dm, "local_rank", 0)))
    torch.cuda.set_device(local_rank)
    device = getattr(dm, "device", torch.device(f"cuda:{local_rank}"))
    rank = int(os.environ.get("RANK", 0))
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return dm, scatter_tensor, device, rank, world_size


def init_1d_domain_mesh(dm, world_size: int, reporter: Reporter):
    """1D spatial mesh: (batch, domain). Domain shards along one spatial axis."""
    domain_size = 2 if world_size >= 2 and world_size % 2 == 0 else 1
    batch_size = max(1, world_size // domain_size)
    try:
        mesh = dm.initialize_mesh(
            mesh_shape=(batch_size, domain_size),
            mesh_dim_names=["batch", "domain"],
        )
        domain_mesh = mesh["domain"]
        reporter.write(
            f"[1D] mesh: batch={batch_size}, domain={domain_size}, "
            f"world_size={world_size}"
        )
        return domain_mesh
    except Exception as exc:
        reporter.write(
            "[1D] WARNING: could not initialize device mesh; "
            f"falling back to local tensors only. {short_error(exc)}"
        )
        return None


def init_2d_domain_mesh(dm, world_size: int, reporter: Reporter):
    """2D spatial mesh: pure (domain_h, domain_w) 2x2 over all 4 GPUs.

    No batch/FSDP axis here — the point is to stress square-domain conv
    halo exchange (including corners). Requires exactly 4 GPUs.
    """
    if world_size != 4:
        reporter.write(
            f"[2D] SKIP: 2D spatial gate suite needs exactly 4 GPUs "
            f"(world_size={world_size}). Launch with --nproc_per_node=4."
        )
        return None
    try:
        mesh = dm.initialize_mesh(
            mesh_shape=(2, 2),
            mesh_dim_names=["domain_h", "domain_w"],
        )
        reporter.write("[2D] mesh: domain_h=2, domain_w=2 (2x2 over 4 GPUs)")
        return mesh
    except Exception as exc:
        reporter.write(
            f"[2D] WARNING: could not initialize 2x2 mesh; skipping. {short_error(exc)}"
        )
        return None


def build_helpers(scatter_tensor, device, mesh, placements, source_rank_fn):
    """Return (make_input, maybe_distribute) bound to a specific mesh/placement."""

    def make_input(shape, requires_grad: bool = True):
        full = torch.randn(shape, device=device, dtype=torch.float32)
        full.requires_grad_(requires_grad)
        if mesh is None:
            return full
        sharded = scatter_tensor(
            full,
            source_rank_fn(),
            mesh,
            placements=placements,
            global_shape=torch.Size(shape),
            dtype=full.dtype,
        )
        if requires_grad:
            sharded.requires_grad_(True)
        return sharded

    def maybe_distribute(module: nn.Module) -> nn.Module:
        module = module.to(device)
        if mesh is not None:
            module = distribute_module(module, device_mesh=mesh)
        return module

    return make_input, maybe_distribute


def main() -> int:
    args = parse_args()
    dm, scatter_tensor, device, rank, world_size = setup_distributed()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_name = args.report_name or f"shardtensor_op_probe-{stamp}.out"
    report_path = Path(args.report_dir) / report_name
    reporter = Reporter(report_path, rank)

    reporter.write("ShardTensor Op-Support Probe Report")
    reporter.write("=" * 50)
    reporter.write(f"torch={torch.__version__}")
    reporter.write(f"device={device}, rank={rank}, world_size={world_size}")
    reporter.write(
        f"main_shape=(1,{args.channels},{args.height},{args.width})"
    )

    main_shape = (1, args.channels, args.height, args.width)
    conv_shape = (1, args.conv_channels, args.height, args.width)
    small_shape = (1, args.small_channels, args.small_side, args.small_side)

    # Swappable helper namespace: op factories reference H.make_input /
    # H.maybe_distribute, so we can retarget them at a different mesh (1D -> 2D)
    # without duplicating op definitions.
    H = SimpleNamespace(make_input=None, maybe_distribute=None)

    results: dict[str, str] = {}

    def probe_op(name, shape, op_factory: Callable[[], Callable[[torch.Tensor], torch.Tensor]]):
        reporter.write(f"\n{name}: RUN shape={shape}")
        try:
            x = H.make_input(shape, requires_grad=True)
            op_fn = op_factory()
            y = op_fn(x)
            loss = y.float().mean()
            loss.backward()
            results[name] = "PASS"
            reporter.write(f"{name}: PASS")
        except Exception as exc:  # noqa: BLE001 - probe/report script.
            status = classify_exception(exc)
            results[name] = f"{status}: {short_error(exc)}"
            reporter.write(f"{name}: {status} - {short_error(exc)}")
            if args.print_tracebacks and rank == 0:
                reporter.write(traceback.format_exc())
        finally:
            try:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- Reusable op factories (reference H) --------------------------------
    def f_groupnorm():
        return lambda x: H.maybe_distribute(
            nn.GroupNorm(num_groups=32, num_channels=args.channels)
        )(x)

    def f_conv(dilation):
        return lambda x: H.maybe_distribute(
            nn.Conv2d(args.conv_channels, args.conv_channels,
                      kernel_size=3, dilation=dilation, padding=dilation)
        )(x)

    def f_avgpool_k2():
        return lambda x: H.maybe_distribute(nn.AvgPool2d(kernel_size=2, stride=2))(x)

    def f_mse():
        target = H.make_input(main_shape, requires_grad=False)
        return lambda x: F.mse_loss(x, target, reduction="mean")

    # =========================================================================
    # SUITE 1 — full 1D spatial sharding (Shard along H)
    # =========================================================================
    reporter.write("\n" + "#" * 50)
    reporter.write("# SUITE 1: 1D spatial sharding (Shard(dim=%d))" % args.shard_dim)
    reporter.write("#" * 50)

    domain_mesh_1d = init_1d_domain_mesh(dm, world_size, reporter)

    def source_rank_1d() -> int:
        if domain_mesh_1d is None or not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_global_rank(domain_mesh_1d.get_group(), 0)

    mk1, dist1 = build_helpers(
        scatter_tensor, device, domain_mesh_1d,
        placements=(Shard(args.shard_dim),),
        source_rank_fn=source_rank_1d,
    )
    H.make_input, H.maybe_distribute = mk1, dist1

    probe_op("GroupNorm32", main_shape, f_groupnorm)
    for d in (1, 2, 4, 8):
        probe_op(f"DilatedConv2d_d{d}", conv_shape, lambda d=d: f_conv(d))
    probe_op("CappedGELU", small_shape, lambda: lambda x: CappedGELU().to(device)(x))
    probe_op("Clamp", small_shape, lambda: lambda x: torch.clamp(x, -2.0, 2.0))
    probe_op("ResidualAdd", small_shape, lambda: lambda x: x + 0.125 * x)
    probe_op("Where", small_shape, lambda: lambda x: torch.where(x > 0.0, x, -x))
    probe_op("CatChannel", small_shape, lambda: lambda x: torch.cat([x, x], dim=1))
    probe_op("AvgPool2d_k2_s2", small_shape, f_avgpool_k2)
    probe_op("MaxPool2d_k2_s2", small_shape,
             lambda: lambda x: H.maybe_distribute(nn.MaxPool2d(kernel_size=2, stride=2))(x))
    probe_op("BilinearUpsample_x2", small_shape,
             lambda: lambda x: H.maybe_distribute(nn.Upsample(scale_factor=2, mode="bilinear"))(x))

    def skip_align_op():
        pads = (1, 1, 2, 2)  # even totals per axis so shards split cleanly
        skip = H.make_input(
            (small_shape[0], small_shape[1],
             small_shape[2] + pads[2] + pads[3],
             small_shape[3] + pads[0] + pads[1]),
            requires_grad=False,
        )
        return lambda x: F.pad(x, pads) + skip

    probe_op("SkipAlignPadAdd", small_shape, skip_align_op)
    probe_op("MSELoss_mean", main_shape, f_mse)

    # =========================================================================
    # SUITE 2 — gate-only 2D spatial sharding (Shard along H AND W, 2x2 mesh)
    # This is the go/no-go for row types B/D/F. Requires corner halo exchange.
    # =========================================================================
    domain_mesh_2d = None
    if not args.skip_2d:
        reporter.write("\n" + "#" * 50)
        reporter.write("# SUITE 2: 2D spatial sharding gate (Shard(2), Shard(3))")
        reporter.write("#" * 50)
        domain_mesh_2d = init_2d_domain_mesh(dm, world_size, reporter)

    if domain_mesh_2d is not None:
        mk2, dist2 = build_helpers(
            scatter_tensor, device, domain_mesh_2d,
            placements=(Shard(2), Shard(3)),  # H on domain_h, W on domain_w
            source_rank_fn=lambda: 0,          # (0,0) coord == global rank 0
        )
        H.make_input, H.maybe_distribute = mk2, dist2

        # Gate ops only. Dilated conv is the corner-halo stress test.
        probe_op("GroupNorm32_2D", main_shape, f_groupnorm)
        for d in (1, 2, 4, 8):
            probe_op(f"DilatedConv2d_d{d}_2D", conv_shape, lambda d=d: f_conv(d))
        probe_op("AvgPool2d_k2_s2_2D", small_shape, f_avgpool_k2)
        probe_op("MSELoss_mean_2D", main_shape, f_mse)

    # ---- Summary ------------------------------------------------------------
    reporter.write("\n" + "=" * 50)
    passed = sum(value == "PASS" for value in results.values())
    reporter.write(f"Summary: {passed}/{len(results)} ops pass.")
    for name, status in results.items():
        reporter.write(f"{name}: {status}")

    twod = {k: v for k, v in results.items() if k.endswith("_2D")}
    if twod:
        twod_pass = sum(v == "PASS" for v in twod.values())
        reporter.write("\n" + "-" * 50)
        reporter.write(
            f"2D spatial gate: {twod_pass}/{len(twod)} pass "
            f"-> {'ROW TYPES B/D/F VIABLE' if twod_pass == len(twod) else 'NEEDS ATTENTION'}"
        )

    reporter.write(f"report_path={report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())