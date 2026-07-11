#!/usr/bin/env python3
"""Probe PhysicsNeMo ShardTensor support for Ocean_Emulator spatial ops."""

import argparse
import datetime as dt
import os
import sys
import traceback
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
            "Install a recent PhysicsNeMo build that includes ShardTensor padding "
            "fixes after PR #1535, merged May 8 2026.\n"
            "Example:\n"
            "  pip install -U physicsnemo\n"
            "or install NVIDIA/physicsnemo from a post-2026-05-08 commit."
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
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return dm, scatter_tensor, device, rank, world_size


def init_domain_mesh(dm, world_size: int, reporter: Reporter):
    domain_size = 2 if world_size >= 2 and world_size % 2 == 0 else 1
    batch_size = max(1, world_size // domain_size)
    try:
        mesh = dm.initialize_mesh(mesh_shape=(batch_size, domain_size), mesh_dim_names=["batch", "domain"])
        domain_mesh = mesh["domain"]
        reporter.write(
            f"mesh: batch={batch_size}, domain={domain_size}, "
            f"world_size={world_size}"
        )
        return domain_mesh
    except Exception as exc:
        reporter.write(
            "WARNING: could not initialize PhysicsNeMo device mesh; "
            f"falling back to local tensors only. {short_error(exc)}"
        )
        return None


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
        f"main_shape=(1,{args.channels},{args.height},{args.width}), "
        f"shard_dim={args.shard_dim}"
    )

    domain_mesh = init_domain_mesh(dm, world_size, reporter)
    placement = (Shard(args.shard_dim),)

    def source_rank() -> int:
        if domain_mesh is None or not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_global_rank(domain_mesh.get_group(), 0)

    def make_input(shape: tuple[int, ...], requires_grad: bool = True):
        full = torch.randn(shape, device=device, dtype=torch.float32)
        full.requires_grad_(requires_grad)
        if domain_mesh is None:
            return full
        sharded = scatter_tensor(
            full,
            source_rank(),
            domain_mesh,
            placements=placement,
            global_shape=torch.Size(shape),
            dtype=full.dtype,
        )
        if requires_grad:
            sharded.requires_grad_(True)
        return sharded

    def maybe_distribute(module: nn.Module) -> nn.Module:
        module = module.to(device)
        if domain_mesh is not None:
            module = distribute_module(module, device_mesh=domain_mesh)
        return module

    results: dict[str, str] = {}

    def probe_op(name: str, shape: tuple[int, int, int, int], op_factory: Callable[[], Callable[[torch.Tensor], torch.Tensor]]) -> None:
        reporter.write(f"\n{name}: RUN shape={shape}")
        try:
            x = make_input(shape, requires_grad=True)
            op_fn = op_factory()
            y = op_fn(x)
            loss = y.float().mean()
            loss.backward()
            results[name] = "PASS"
            reporter.write(f"{name}: PASS")
        except Exception as exc:  # noqa: BLE001 - this is a probe/report script.
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

    main_shape = (1, args.channels, args.height, args.width)
    conv_shape = (1, args.conv_channels, args.height, args.width)
    small_shape = (1, args.small_channels, args.small_side, args.small_side)

    probe_op(
        "GroupNorm32",
        main_shape,
        lambda: lambda x: maybe_distribute(nn.GroupNorm(num_groups=32, num_channels=args.channels))(x),
    )

    for dilation in (1, 2, 4, 8):
        probe_op(
            f"DilatedConv2d_d{dilation}",
            conv_shape,
            lambda dilation=dilation: lambda x: maybe_distribute(
                nn.Conv2d(args.conv_channels, args.conv_channels, kernel_size=3, dilation=dilation, padding=dilation)
            )(x),
        )

    probe_op("CappedGELU", small_shape, lambda: lambda x: CappedGELU().to(device)(x))
    probe_op("Clamp", small_shape, lambda: lambda x: torch.clamp(x, -2.0, 2.0))
    probe_op("ResidualAdd", small_shape, lambda: lambda x: x + 0.125 * x)
    probe_op("Where", small_shape, lambda: lambda x: torch.where(x > 0.0, x, -x))
    probe_op("CatChannel", small_shape, lambda: lambda x: torch.cat([x, x], dim=1))

    for name, module in (
        ("AvgPool2d_k2_s2", nn.AvgPool2d(kernel_size=2, stride=2)),
        ("AvgPool2d_k3_s1", nn.AvgPool2d(kernel_size=3, stride=1, padding=1)),
        ("MaxPool2d_k2_s2", nn.MaxPool2d(kernel_size=2, stride=2)),
        ("MaxPool2d_k3_s1", nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
    ):
        probe_op(name, small_shape, lambda module=module: lambda x: maybe_distribute(module)(x))

    probe_op(
        "BilinearUpsample_x2",
        small_shape,
        lambda: lambda x: maybe_distribute(nn.Upsample(scale_factor=2, mode="bilinear"))(x),
    )

    def skip_align_op():
        pads = (1, 2, 0, 1)
        skip = make_input(
            (
                small_shape[0],
                small_shape[1],
                small_shape[2] + pads[2] + pads[3],
                small_shape[3] + pads[0] + pads[1],
            ),
            requires_grad=False,
        )
        return lambda x: F.pad(x, pads) + skip

    probe_op("SkipAlignPadAdd", small_shape, skip_align_op)

    def mse_op():
        target = make_input(main_shape, requires_grad=False)
        return lambda x: F.mse_loss(x, target, reduction="mean")

    probe_op("MSELoss_mean", main_shape, mse_op)

    reporter.write("\n" + "=" * 50)
    passed = sum(value == "PASS" for value in results.values())
    reporter.write(f"Summary: {passed}/{len(results)} ops pass.")
    for name, status in results.items():
        reporter.write(f"{name}: {status}")
    reporter.write(f"report_path={report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
