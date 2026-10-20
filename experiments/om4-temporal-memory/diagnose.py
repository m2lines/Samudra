"""Causal texture interventions and immediate/future gradient attribution."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from common import (
    PROG,
    OceanModel,
    Prepared,
    highpass,
    interior_mask,
    load_official,
    metrics,
    mse,
    sha256,
)

GROUPS = {
    "temperature_surface": [38],
    "temperature_deep": list(range(49, 57)),
    "salinity_surface": [57],
    "salinity_deep": list(range(68, 76)),
    "ssh": [76],
    "zonal_velocity_deep": list(range(11, 19)),
    "meridional_velocity_deep": list(range(30, 38)),
}


def group_mask(name, wet):
    select = torch.zeros_like(wet)
    for k in GROUPS[name]:
        select[:, k] = wet[:, k]
        select[:, k + 77] = wet[:, k + 77]
    return select


def randomized(direction, valid, seed):
    """Phase-randomized control, remasked and matched in per-plane L2 norm."""
    gen = torch.Generator(device=direction.device).manual_seed(seed)
    noise = torch.randn(direction.shape, generator=gen, device=direction.device)
    spectrum = torch.fft.rfft2(noise)
    phase = spectrum / spectrum.abs().clamp_min(1e-20)
    result = (
        torch.fft.irfft2(
            torch.fft.rfft2(direction).abs() * phase, s=direction.shape[-2:]
        )
        * valid
    )
    result *= (
        direction.square().sum((-2, -1), keepdim=True)
        / result.square().sum((-2, -1), keepdim=True).clamp_min(1e-30)
    ).sqrt()
    return result


def cosine(a, b):
    denom = a.norm() * b.norm()
    return float((a * b).sum() / denom) if denom > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--output", required=True)
    ap.add_argument("--origins", type=int, default=6)
    ap.add_argument("--calls", type=int, default=12)
    ap.add_argument("--control-seeds", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Prepared(args.data)
    device = torch.device("cuda")
    mask = torch.as_tensor(data.mask, device=device)
    std = torch.as_tensor(data.std, device=device)
    model = OceanModel(mask).to(device).eval()
    provenance = load_official(model, args.checkpoint)
    model.requires_grad_(False)
    provenance.update(
        checkpoint_sha256=sha256(args.checkpoint),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        mode="eval; BatchNorm frozen",
        variables=PROG,
        groups=GROUPS,
        source=data.meta,
        diagnostic="4-call gradient objective; up to 12-call intervention continuation",
    )
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scales = saved["loss_fn_state"]["per_channel_scale"].to(device).reshape(1, -1, 1, 1)
    if scales.shape[1] == 77:
        scales = scales.repeat(1, 2, 1, 1)
    del saved

    def archived_loss(p, t):
        return (((p - t) * model.wet).square() * scales).mean()

    records = []
    gradients = []
    for oi, start in enumerate(data.meta["diagnostic_origins"][: args.origins]):
        if (out / f"origin-{oi}.json").exists():
            continue
        x, bs, targets = data.window(start, args.calls, device)
        began = time.monotonic()
        with torch.no_grad():
            first, _ = model(x, bs[0])
        if oi == 0 and args.reference:
            ref = np.load(args.reference)["prediction"]
            actual = (
                (
                    first.reshape(2, 77, *mask.shape[-2:]) * std[None, :, None, None]
                    + torch.as_tensor(data.mean, device=device)[None, :, None, None]
                )
                .cpu()
                .numpy()
            )
            wet = np.broadcast_to(data.mask[None], actual.shape)
            diff = (actual - ref)[wet]
            provenance["official_replay"] = {
                "max_abs": float(np.max(abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff.astype("float64") ** 2))),
            }
            # Cross-device FP32 kernels need not be bit-identical; reject material
            # mismatch before interpreting any scientific intervention.
            normalized = (actual - ref) / data.std[None, :, None, None]
            if np.sqrt(np.mean(normalized[wet].astype("float64") ** 2)) > 1e-3:
                raise RuntimeError(
                    f"Official prediction replay mismatch: {provenance['official_replay']}"
                )
        inside = interior_mask(model.wet)
        error = first - targets[0]
        direction = highpass(error, inside)
        leaf = first.detach().requires_grad_(True)
        future = leaf
        future_losses = []
        future_archived = []
        for k in range(1, 4):
            future, _ = model(future, bs[k])
            future_losses.append(mse(future, targets[k], model.wet))
            future_archived.append(archived_loss(future, targets[k]))
        for objective, own, later in (
            ("fixed_wet_mse", mse(leaf, targets[0], model.wet), sum(future_losses)),
            (
                "archived_dynamic_weights",
                archived_loss(leaf, targets[0]),
                sum(future_archived),
            ),
        ):
            g0 = torch.autograd.grad(own, leaf, retain_graph=True)[0]
            gf = torch.autograd.grad(later, leaf, retain_graph=True)[0]
            for group in GROUPS:
                valid = group_mask(group, inside)
                d = direction * valid
                gradients.append(
                    {
                        "origin": oi,
                        "group": group,
                        "objective": objective,
                        "immediate_gradient_dot_error_texture": float((g0 * d).sum()),
                        "future_gradient_dot_error_texture": float((gf * d).sum()),
                        "gradient_cosine_in_group": cosine(g0 * valid, gf * valid),
                        "future_gradient_texture_cosine": cosine(gf * valid, d),
                        "direction_norm": float(d.norm()),
                        "future_gradient_norm": float((gf * valid).norm()),
                    }
                )
        del future, future_losses, future_archived, leaf, later, own, g0, gf
        torch.cuda.empty_cache()
        variants = [("baseline", "all", 0.0, first)]
        for group in GROUPS:
            valid = group_mask(group, inside)
            d = direction * valid
            for alpha in (0.25, 0.5, 1.0):
                variants.append(
                    ("remove_error_texture", group, alpha, first - alpha * d)
                )
            for seed in range(args.control_seeds):
                r = randomized(d, valid, 1000 * oi + seed)
                variants.append(
                    (f"phase_randomized_{seed}", group, 0.5, first - 0.5 * r)
                )
            # Identical intervention size on a truth state: checks whether the
            # network simply responds strongly to small physical perturbations.
            variants.append(("truth_minus_texture", group, 0.5, targets[0] - 0.5 * d))
        variants.append(("truth_restart", "all", 0.0, targets[0]))
        for kind, group, alpha, initial in variants:
            p = initial
            with torch.no_grad():
                for k in range(args.calls):
                    if k:
                        p, _ = model(p, bs[k])
                    if k in (0, 1, 3, 5, 11):
                        records.append(
                            {
                                "origin": oi,
                                "initial_time": data.times[start + 1],
                                "kind": kind,
                                "group": group,
                                "alpha": alpha,
                                "call": k + 1,
                                "lead_days": 10 * (k + 1),
                                "mse": float(mse(p, targets[k], model.wet)),
                                **metrics(p, targets[k], mask, std),
                            }
                        )
        np.savez_compressed(
            out / f"origin-{oi}-maps.npz",
            prediction=first.detach().cpu().numpy(),
            truth=targets[0].cpu().numpy(),
            error_texture=direction.detach().cpu().numpy(),
        )
        (out / f"origin-{oi}.json").write_text(
            json.dumps(
                {
                    "interventions": [r for r in records if r["origin"] == oi],
                    "gradients": [r for r in gradients if r["origin"] == oi],
                },
                indent=2,
            )
        )
        print(f"Origin {oi} complete in {time.monotonic() - began:.0f}s", flush=True)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    (out / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
