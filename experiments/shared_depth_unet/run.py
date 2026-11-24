# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Small, paired 2x2 sharing/context study using Samudra ConvNeXt blocks."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from samudra.models.modules.blocks import ConvNeXtBlock

ARMS = {
    "independent": (False, False),
    "shared": (True, False),
    "shared_context": (True, True),
    "independent_context": (False, True),
}
H, W, R, PATCH, HALO = 128, 256, 4, 112, 40


def restrict(x):
    # Area-average cell values; same aligned restriction for input and target.
    return F.avg_pool2d(x, R)


def lift(x):
    # Piecewise-constant prolongation preserves R(U(x)) == x exactly.
    return F.interpolate(x, scale_factor=R, mode="nearest")


def crop(x, origins, size):
    yy = (origins[:, 0, None] + torch.arange(size, device=x.device)) % x.shape[-2]
    xx = (origins[:, 1, None] + torch.arange(size, device=x.device)) % x.shape[-1]
    return x[
        torch.arange(len(x), device=x.device)[:, None, None],
        :,
        yy[:, :, None],
        xx[:, None, :],
    ].permute(0, 3, 1, 2)


def interior(x):
    return x[..., HALO:-HALO, HALO:-HALO]


class Block(ConvNeXtBlock):
    """Repository ConvNeXt block with selectable torus / regional boundaries.

    Only padding differs from the repository forward: global synthetic fields
    are periodic in BOTH directions. No normalization or stochastic layers.
    """

    def __init__(self, a, b):
        super().__init__(
            a,
            b,
            kernel_size=3,
            dilation=1,
            upscale_factor=2,
            norm="nonorm",
            pointwise_linear=True,
        )

    def forward(self, x, periodic=False):
        skip = self.skip_module(x)
        for layer in self.convblock:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                x = F.pad(
                    x, (self.N_pad,) * 4, mode="circular" if periodic else "constant"
                )
            x = layer(x)
        return skip + x


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.enter = Block(8, 8)
        self.middle = Block(8, 12)
        self.up = nn.Conv2d(12, 8, 1)
        self.exit = Block(8, 8)

    def forward(self, x, periodic):
        skip = self.enter(x, periodic)
        mid = self.middle(F.avg_pool2d(skip, 2), periodic)
        return self.exit(
            self.up(F.interpolate(mid, scale_factor=2, mode="nearest")) + skip, periodic
        )


class Route(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 1)
        self.core = Core()
        self.head = nn.Conv2d(8, 1, 1)
        nn.init.zeros_(self.head.weight)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)


class Model(nn.Module):
    def __init__(self, shared, context):
        super().__init__()
        self.global_route = Route()
        self.local_route = (
            self.global_route if shared else copy.deepcopy(self.global_route)
        )
        self.detail = Block(1, 4)
        self.detail_down = nn.Conv2d(4, 8, 1)
        self.fine_decode = Block(12, 4)
        self.fine_head = nn.Conv2d(4, 1, 1)
        nn.init.zeros_(self.fine_head.weight)
        assert self.fine_head.bias is not None
        nn.init.zeros_(self.fine_head.bias)
        self.context = nn.Conv2d(8, 8, 1, bias=False) if context else None
        if self.context is not None:
            nn.init.zeros_(self.context.weight)

    def global_step(self, c):
        z = self.global_route.core(self.global_route.stem(c), True)
        return c + self.global_route.head(z), z

    def fine_step(self, f, z_global, origins):
        c = restrict(f)
        detail = self.detail(f - lift(c), False)
        z = self.local_route.stem(c) + self.detail_down(restrict(detail))
        if self.context is not None:
            z = z + self.context(crop(z_global, origins // R, f.shape[-1] // R))
        z = self.local_route.core(z, False)
        # The coarse tendency head is trained on both paths; high-pass correction
        # makes the distinction between resolved state and optional detail exact.
        coarse_delta = lift(self.local_route.head(z))
        fine_delta = self.fine_head(
            self.fine_decode(torch.cat((lift(z), detail), 1), False)
        )
        fine_delta = fine_delta - lift(restrict(fine_delta))
        return f + coarse_delta + fine_delta

    def forward(self, c, f, origins):
        cp, z = self.global_step(c)
        return cp, self.fine_step(f, z, origins)


def simulate(n, seed, device, frames=10, dt=0.02):
    """Dealiased pseudo-spectral 2D incompressible vorticity, RK4, no forcing.

    Domain [0,2pi) x [0,4pi), nu=.002, dt=.02, output interval .4.
    One channel omega; velocity is recovered nonlocally by inverse Laplacian.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    ky = torch.fft.fftfreq(H, d=1 / H, device=device)[:, None]
    kx = torch.fft.rfftfreq(W, d=2 / W, device=device)[None, :]
    k2 = ky.square() + kx.square()
    dealias = ((ky.abs() < H / 3) & (kx.abs() < W / 6)).float()
    envelope = torch.exp(-k2 / 24) + 0.25 * torch.exp(-((k2.sqrt() - 12) / 4).square())
    wh = torch.fft.rfft2(torch.randn(n, H, W, generator=gen, device=device)) * envelope
    wh[:, 0, 0] = 0
    omega = torch.fft.irfft2(wh, s=(H, W))
    omega = omega / omega.std(dim=(-2, -1), keepdim=True)
    wh = torch.fft.rfft2(omega) * dealias
    inv = torch.where(k2 > 0, 1 / k2.clamp_min(1e-12), 0)

    def rhs(w):
        # u=-psi_y, v=psi_x, Delta psi=omega.
        psi = -w * inv
        u = torch.fft.irfft2(-1j * ky * psi, s=(H, W))
        v = torch.fft.irfft2(1j * kx * psi, s=(H, W))
        wx = torch.fft.irfft2(1j * kx * w, s=(H, W))
        wy = torch.fft.irfft2(1j * ky * w, s=(H, W))
        return -torch.fft.rfft2(u * wx + v * wy) * dealias - 0.002 * k2 * w

    result = [torch.fft.irfft2(wh, s=(H, W))[:, None]]
    for _ in range(frames - 1):
        for _ in range(round(0.4 / dt)):
            a = rhs(wh)
            b = rhs(wh + dt / 2 * a)
            c = rhs(wh + dt / 2 * b)
            d = rhs(wh + dt * c)
            wh = (wh + dt / 6 * (a + 2 * b + 2 * c + d)) * dealias
        result.append(torch.fft.irfft2(wh, s=(H, W))[:, None])
    trajectory = torch.stack(result, 1)
    assert torch.isfinite(trajectory).all()
    return trajectory


def make_schedule(n, steps, batch, seed, device, max_time=9):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(n, (steps, batch), generator=g)
    times = torch.randint(max_time, (steps, batch), generator=g)
    # Align to both restriction and the shared core's pooling lattice.
    origins = torch.stack(
        (
            torch.randint(H // 8, (steps, batch), generator=g) * 8,
            torch.randint(W // 8, (steps, batch), generator=g) * 8,
        ),
        -1,
    )
    return [t.to(device) for t in (ids, times, origins)]


def batch_at(data, schedule, i):
    ids, times, origins = [s[i] for s in schedule]
    x, y = data[ids, times], data[ids, times + 1]
    return (
        restrict(x),
        crop(x, origins, PATCH),
        restrict(y),
        crop(y, origins, PATCH),
        origins,
    )


@torch.no_grad()
def evaluate(model, data, schedule, leads=(1, 3, 5), patch_size=PATCH):
    model.eval()
    sums: dict[str, dict[str, float]] = {str(k): {} for k in leads}
    for i in range(schedule[0].shape[0]):
        ids, times, origins = [s[i] for s in schedule]
        origins = origins - (patch_size - PATCH) // 2
        margin = (patch_size - 32) // 2

        def center(a):
            return a[..., margin:-margin, margin:-margin]

        x = data[ids, times]
        c, f = restrict(x), crop(x, origins, patch_size)
        c0, f0 = c.clone(), f.clone()
        for lead in range(1, max(leads) + 1):
            c, f = model(c, f, origins)
            if lead not in leads:
                continue
            y = data[ids, times + lead]
            ct, ft = restrict(y), crop(y, origins, patch_size)
            fp, yt = center(f), center(ft)

            def hp(a):
                return a - lift(restrict(a))

            vals = {
                "coarse": (c - ct).square().mean(),
                "fine": (fp - yt).square().mean(),
                "fine_detail": (hp(fp) - hp(yt)).square().mean(),
                "persistence_coarse": (c0 - ct).square().mean(),
                "persistence_fine": (center(f0) - yt).square().mean(),
                "persistence_detail": (hp(center(f0)) - hp(yt)).square().mean(),
                "fine_target_rms": yt.square().mean(),
                "fine_prediction_rms": fp.square().mean(),
                "detail_target_rms": hp(yt).square().mean(),
                "detail_prediction_rms": hp(fp).square().mean(),
                "fine_outer_8": torch.cat(
                    ((f - ft)[..., :8, :].flatten(), (f - ft)[..., -8:, :].flatten())
                )
                .square()
                .mean(),
            }
            for key, val in vals.items():
                sums[str(lead)][key] = sums[str(lead)].get(key, 0.0) + val.item()
    return {
        k: {name: math.sqrt(v / schedule[0].shape[0]) for name, v in vals.items()}
        for k, vals in sums.items()
    }


def verify(device):
    torch.manual_seed(42)
    x = torch.randn(2, 1, H, W, device=device)
    origins = torch.tensor([[0, 0], [8, 16]], device=device)
    assert torch.equal(
        restrict(crop(x, origins, PATCH)), crop(restrict(x), origins // R, PATCH // R)
    )
    assert torch.allclose(
        restrict(x - lift(restrict(x))), torch.zeros_like(restrict(x)), atol=2e-7
    )
    models = [Model(*ARMS[a]).to(device) for a in ARMS]
    for m in models:
        c, f = restrict(x), crop(x, origins, PATCH)
        cp, fp = m(c, f, origins)
        assert torch.equal(cp, c) and torch.equal(fp, f)
        (cp.square().mean() + fp.square().mean()).backward()
        assert all(
            torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
        )
    assert models[1].global_route is models[1].local_route
    assert models[0].global_route is not models[0].local_route
    # Nontrivial trained-like weights: full torus vs aligned crop, valid center.
    block = Core().to(device)
    z = torch.randn(2, 8, H // R, W // R, device=device)
    zpatch = crop(z, origins // R, PATCH // R)
    a = crop(block(z, True), origins // R, PATCH // R)[..., 10:-10, 10:-10]
    b = block(zpatch, False)[..., 10:-10, 10:-10]
    discrepancy = (a - b).abs().max().item()
    assert discrepancy < 2e-6, discrepancy
    with torch.no_grad():
        a = simulate(2, 101, device, frames=2)
        b = simulate(2, 101, device, frames=2, dt=0.01)
        solver_error = (a[:, -1] - b[:, -1]).square().mean().sqrt().item()
        assert solver_error < 2e-4, solver_error
        assert a[:, -1].square().mean() < a[:, 0].square().mean()
        assert a.mean(dim=(-2, -1)).abs().max() < 1e-6
    return {
        "solver_timestep_halving_rmse": solver_error,
        "enstrophy_decay": True,
        "mean_vorticity_conserved": True,
        "restriction_alignment": True,
        "zero_mean_detail": True,
        "identity_initialization": True,
        "finite_backward": True,
        "parameter_sharing": True,
        "crop_core_max_abs_error": discrepancy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    checks = verify(device)
    print("VERIFIED", json.dumps(checks), flush=True)
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2))
    if args.verify_only:
        return
    source = Path(__file__)
    provenance = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "blocks_sha256": hashlib.sha256(
            Path("src/samudra/models/modules/blocks.py").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "geometry": {"fine": [H, W], "ratio": R, "patch": PATCH, "halo": HALO},
        "data_seeds": [12001, 12002, 12003],
        "train_val_test_trajectories": [48, 12, 12],
    }
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        for key in ("args", "source_sha256", "blocks_sha256", "geometry", "data_seeds"):
            if previous[key] != provenance[key]:
                raise ValueError(
                    f"Existing output differs in {key}; use a fresh output directory"
                )
    (args.output / "run_source.py").write_bytes(source.read_bytes())
    provenance_path.write_text(json.dumps(provenance, indent=2))
    data_path = args.output / "data.pt"
    start = time.monotonic()
    if data_path.exists():
        train, val, test = torch.load(data_path, map_location=device, weights_only=True)
    else:
        train, val, test = [
            simulate(n, s, device) for n, s in [(48, 12001), (12, 12002), (12, 12003)]
        ]
        torch.save([d.cpu() for d in (train, val, test)], data_path)
    provenance["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"DATA ready in {time.monotonic() - start:.1f}s", flush=True)
    scales = []
    for d in (
        restrict(train[:, 1:].flatten(0, 1)) - restrict(train[:, :-1].flatten(0, 1)),
        train[:, 1:] - train[:, :-1],
    ):
        scales.append(d.square().mean().item())
    (args.output / "loss_scales.json").write_text(json.dumps(scales))
    val_sched = make_schedule(len(val), 8, args.batch, 20001, device, max_time=5)
    test_sched = make_schedule(len(test), 16, args.batch, 20002, device, max_time=5)
    results = []
    for seed in args.seeds:
        schedule = make_schedule(
            len(train), args.steps, args.batch, 30000 + seed, device
        )
        for arm, spec in ARMS.items():
            run_dir = args.output / f"{arm}_seed{seed}"
            run_dir.mkdir(exist_ok=True)
            if (run_dir / "result.json").exists():
                results.append(json.loads((run_dir / "result.json").read_text()))
                continue
            torch.manual_seed(40000 + seed)
            model = Model(*spec).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            best, best_step, history = float("inf"), 0, []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.monotonic()
            for step in range(args.steps):
                model.train()
                c, f, ct, ft, origins = batch_at(train, schedule, step)
                opt.zero_grad(set_to_none=True)
                cp, fp = model(c, f, origins)
                lc = (cp - ct).square().mean() / scales[0]
                lf = (interior(fp) - interior(ft)).square().mean() / scales[1]
                loss = 0.5 * (lc + lf)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {arm} {seed} {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                for group in opt.param_groups:
                    group["lr"] = 0.0002 + 0.0008 * 0.5 * (
                        1 + math.cos(math.pi * step / args.steps)
                    )
                opt.step()
                if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                    metrics = evaluate(model, val, val_sched, leads=(1,))
                    score = 0.5 * (
                        metrics["1"]["coarse"] ** 2 / scales[0]
                        + metrics["1"]["fine"] ** 2 / scales[1]
                    )
                    row = {
                        "step": step + 1,
                        "train_coarse": lc.item(),
                        "train_fine": lf.item(),
                        "val_score": score,
                    }
                    history.append(row)
                    if score < best:
                        best, best_step = score, step + 1
                        torch.save(model.state_dict(), run_dir / "best.pt")
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "seed": seed,
                                **row,
                                "seconds": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            model.load_state_dict(
                torch.load(run_dir / "best.pt", weights_only=True, map_location=device)
            )
            metrics = evaluate(model, test, test_sched)
            extent = {
                str(size): evaluate(
                    model, test, test_sched, leads=(1,), patch_size=size
                )["1"]
                for size in (80, 128)
            }
            result = {
                "arm": arm,
                "seed": seed,
                "best_step": best_step,
                "best_val": best,
                "seconds": elapsed,
                "parameters": sum(p.numel() for p in model.parameters()),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0,
                "test": metrics,
                "extent_test": extent,
            }
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            (run_dir / "result.json").write_text(json.dumps(result, indent=2))
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print("RESULT", json.dumps(result), flush=True)
    (args.output / "COMPLETE.json").write_text(
        json.dumps({"runs": len(results), "source_sha256": provenance["source_sha256"]})
    )


if __name__ == "__main__":
    main()
