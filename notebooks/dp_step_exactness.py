#!/usr/bin/env python3
"""Compare one Samudra optimizer step on one GPU and a 2x2 ShardTensor mesh.

Run on one four-GPU node:
    torchrun --standalone --nproc_per_node=4 notebooks/dp_step_exactness.py

The default model keeps the LLC Samudra topology (four U-Net levels, dilation
1/2/4/8, GroupNorm, AvgPool, bilinear upsampling) while using modest widths so
the single-GPU reference is a correctness baseline rather than a memory test.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean_emulators.config import BlockConfig, SamudraConfig, UNetBackboneConfig
from ocean_emulators.shardtensor import (
    DomainParallelConfig,
    DomainParallelContext,
    validate_shardable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--widths", type=int, nargs=4, default=[32, 48, 64, 64])
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--max-relative-l2", type=float, default=2e-3)
    parser.add_argument(
        "--post-step-atol",
        type=float,
        default=5e-4,
        help="Absolute tolerance for parameters after the Adam update.",
    )
    parser.add_argument(
        "--post-step-rtol",
        type=float,
        default=2e-4,
        help="Relative tolerance for parameters after the Adam update.",
    )
    parser.add_argument(
        "--post-step-max-relative-l2",
        type=float,
        default=2e-4,
        help="Relative-L2 tolerance for parameters after the Adam update.",
    )
    return parser.parse_args()


def setup() -> tuple[object, torch.device, int]:
    try:
        from physicsnemo.distributed import DistributedManager
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PhysicsNeMo is required. Install the pinned recent build before "
            "running this exactness gate."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("This exactness gate requires CUDA and four GPUs.")

    DistributedManager.initialize()
    dm = DistributedManager()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise SystemExit(
            f"This gate requires exactly 4 ranks for a 2x2 mesh; got {world_size}."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dm, torch.device(f"cuda:{local_rank}"), dist.get_rank()


def build_model(args: argparse.Namespace, device: torch.device, *, domain_parallel: bool):
    wet = torch.ones(args.out_channels, args.height, args.width, device=device)
    model_cfg = SamudraConfig(
        pred_residuals=False,
        last_kernel_size=3,
        pad="constant",
        checkpointing=None,
        use_bfloat16=False,
        unet=UNetBackboneConfig(
            ch_width=args.widths,
            dilation=[1, 2, 4, 8],
            n_layers=[1, 1, 1, 1],
            core_block=BlockConfig(
                block_type="conv_next_block",
                kernel_size=3,
                activation="capped_gelu",
                upscale_factor=args.upscale_factor,
                norm="group",
                group_norm_groups=32,
            ),
            down_sampling_block="avg_pool",
            up_sampling_block="bilinear_upsample",
        ),
    )
    return model_cfg.build(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hist=0,
        wet=wet,
        area_weights=torch.ones_like(wet),
        static_data=None,
        lat=torch.linspace(-1.0, 1.0, args.height, device=device),
        lon=torch.linspace(-1.0, 1.0, args.width, device=device),
        domain_parallel=domain_parallel,
    ).to(device)


def global_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with ShardTensor's explicit global-mean reduction path."""
    error = pred - target
    return (error * error).mean()


def materialize(value: torch.Tensor) -> torch.Tensor:
    """Return a replicated tensor for a cross-rank exactness comparison."""
    with torch.no_grad():
        if hasattr(value, "full_tensor"):
            return value.full_tensor().detach()
        return value.detach()


def layout(value: torch.Tensor) -> str:
    placements = getattr(value, "placements", None)
    return f"{type(value).__name__}(placements={placements})"


def materialize_parameter_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """All ranks materialize gradients in the same named-parameter order."""
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"Missing gradient for parameter {name}")
        grads[name] = materialize(param.grad)
    return grads


def materialize_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """All ranks materialize post-step parameters in the same order."""
    return {name: materialize(param) for name, param in model.named_parameters()}


def compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    args: argparse.Namespace,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    max_relative_l2: float | None = None,
) -> bool:
    atol = args.atol if atol is None else atol
    rtol = args.rtol if rtol is None else rtol
    max_relative_l2 = (
        args.max_relative_l2 if max_relative_l2 is None else max_relative_l2
    )
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    max_abs = error.abs().max().item()
    expected_norm = torch.linalg.vector_norm(expected)
    relative_l2 = (torch.linalg.vector_norm(error) / expected_norm.clamp_min(1e-12)).item()
    scale = (
        (actual * expected).sum() / expected.square().sum().clamp_min(1e-12)
    ).item()
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    ok &= relative_l2 <= max_relative_l2
    print(
        f"{name:46s} max_abs={max_abs:.3e} rel_l2={relative_l2:.3e} "
        f"scale={scale:.6f} {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def print_run_header(args: argparse.Namespace, device: torch.device) -> None:
    import physicsnemo

    print("======== Python exactness gate ========", flush=True)
    print(
        "global_shape="
        f"({args.batch_size}, {args.in_channels}, {args.height}, {args.width}) "
        f"target_channels={args.out_channels}",
        flush=True,
    )
    print(
        f"unet_widths={args.widths} dilation=[1, 2, 4, 8] "
        f"upscale_factor={args.upscale_factor}",
        flush=True,
    )
    print(
        f"torch={torch.__version__} physicsnemo="
        f"{getattr(physicsnemo, '__version__', 'unknown')} "
        f"device={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    print(
        f"tolerance: atol={args.atol} rtol={args.rtol}; "
        f"max_relative_l2={args.max_relative_l2}; optimizer=Adam(lr={args.lr})",
        flush=True,
    )
    print(
        f"post-step tolerance: atol={args.post_step_atol} "
        f"rtol={args.post_step_rtol}; "
        f"max_relative_l2={args.post_step_max_relative_l2}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    validate_shardable(args.height, args.width, (2, 2), num_downsamples=4)
    dm, device, rank = setup()
    if rank == 0:
        print_run_header(args, device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Every rank builds the same initial model; distribute_model then replaces
    # the GroupNorm layers and replicates parameter storage over the domain mesh.
    if rank == 0:
        print("[1/4] Building and distributing the domain-parallel Samudra model", flush=True)
    distributed_model = build_model(args, device, domain_parallel=True)
    ctx = DomainParallelContext(DomainParallelConfig(cluster_shape=(2, 2)), dm, device)
    distributed_model = ctx.distribute_model(distributed_model)

    full_input = full_target = reference_model = None
    if rank == 0:
        print("[2/4] Building the dense single-GPU reference", flush=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        reference_model = build_model(args, device, domain_parallel=False)
        full_input = torch.randn(
            args.batch_size, args.in_channels, args.height, args.width, device=device
        )
        full_target = torch.randn(
            args.batch_size, args.out_channels, args.height, args.width, device=device
        )

    dist.barrier()
    if rank == 0:
        print("[3/4] Leader-scattering synthetic tensors and running one DP step", flush=True)
    sharded_input = ctx.scatter(full_input, requires_grad=True)
    sharded_target = ctx.scatter(full_target)
    # BaseModel keeps wet as a non-parameter tensor; Phase 1 explicitly gives it
    # the same layout as prediction. Phase 2 will make this part of batch scatter.
    distributed_model.wet = ctx.scatter(
        reference_model.wet.unsqueeze(0) if rank == 0 else None
    )

    distributed_optimizer = torch.optim.Adam(distributed_model.parameters(), lr=args.lr)
    distributed_optimizer.zero_grad(set_to_none=True)
    sharded_output = distributed_model.predict_step(sharded_input)
    sharded_loss = global_mse(sharded_output, sharded_target)
    sharded_loss.backward()

    gathered_output = ctx.gather(sharded_output)
    gathered_input_grad = ctx.gather(sharded_input.grad)
    gathered_loss = ctx.gather(sharded_loss)
    distributed_grads = materialize_parameter_grads(distributed_model)

    passed = torch.tensor([1], device=device, dtype=torch.int32)
    if rank == 0:
        print("[4/4] Running the dense step and comparing all exactness artifacts", flush=True)
        assert reference_model is not None
        assert full_input is not None and full_target is not None
        reference_optimizer = torch.optim.Adam(reference_model.parameters(), lr=args.lr)
        reference_input = full_input.detach().clone().requires_grad_(True)
        reference_output = reference_model.predict_step(reference_input)
        reference_loss = global_mse(reference_output, full_target)
        reference_loss.backward()

        ok = True
        ok &= compare("output", gathered_output, reference_output, args)
        ok &= compare("loss", gathered_loss.reshape(1), reference_loss.reshape(1), args)
        ok &= compare("input_grad", gathered_input_grad, reference_input.grad, args)

        ref_params = dict(reference_model.named_parameters())
        for name, grad in distributed_grads.items():
            ok &= compare(f"grad:{name}", grad, ref_params[name].grad, args)
        print(
            "gradient layouts: "
            + ", ".join(
                f"{name}={layout(param.grad)}"
                for name, param in list(distributed_model.named_parameters())[:3]
            ),
            flush=True,
        )

    # Do not overlap optimizer work with rank-0's dense reference comparison.
    # The barrier also keeps the next DTensor/ShardTensor collectives in lockstep.
    dist.barrier()
    distributed_optimizer.step()
    distributed_params = materialize_parameters(distributed_model)

    if rank == 0:
        reference_optimizer.step()
        ref_params = dict(reference_model.named_parameters())
        for name, param in distributed_params.items():
            ok &= compare(
                f"param:{name}",
                param,
                ref_params[name],
                args,
                atol=args.post_step_atol,
                rtol=args.post_step_rtol,
                max_relative_l2=args.post_step_max_relative_l2,
            )
        if not ok:
            passed.zero_()

    dist.broadcast(passed, src=0)
    dist.barrier()
    if rank == 0:
        print("\n2x2 exactness gate: " + ("PASS" if passed.item() else "FAIL"), flush=True)
    return 0 if passed.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
