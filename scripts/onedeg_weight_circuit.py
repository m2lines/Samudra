# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Program a deep-state carry path into existing Samudra weight tensors.

No architecture changes or new parameters. A low-energy subspace of the
280-channel high-resolution skip stream is reserved for exact copies of the
latest deep tracer inputs. All learned branches are projected away from this
subspace; the decoder blends its remaining learned prediction with these states.
The alpha=0 control measures damage from reserving the subspace itself.

Run from the repository root with python -m scripts.onedeg_weight_circuit.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import copy
from pathlib import Path

import numpy as np
import torch

from samudra.utils.ctx import BatchGrid
from scripts.onedeg_weight_edit import (
    Diagnostics,
    log,
    model_from_checkpoint,
    save_json,
    sha256,
    windows,
)


def context(static, device):
    coordinates = (torch.from_numpy(static["lat"]), torch.from_numpy(static["lon"]))
    return BatchGrid(
        torch.from_numpy(np.tile(static["mask"], (2, 1, 1))).to(device),
        coordinates,
        coordinates,
    )


def inspect(args):
    """Identify weak feature directions without using any prediction targets."""
    out = args.output
    static = np.load(out / "static.npz")
    manifest = json.loads((out / "data_manifest.json").read_text())
    model = model_from_checkpoint(args)
    grams = {
        name: torch.zeros(280, 280, dtype=torch.float64)
        for name in ("first", "up", "last_residual")
    }
    counts = {name: 0 for name in grams}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            a = output.detach().cpu().double().reshape(280, -1)
            grams[name].add_(a @ a.T)
            counts[name] += a.shape[1]

        return hook

    for name, module in (
        ("first", model.unet.layers[0]),
        ("up", model.unet.layers[14]),
        ("last_residual", model.unet.layers[16].convblock[6]),
    ):
        handles.append(module.register_forward_hook(capture(name)))
    for entry in windows(args, manifest):
        data = np.load(out / (entry["name"] + ".npz"))
        prog = torch.from_numpy(data["states"][:2].reshape(1, 154, 180, 360)).to(
            args.device
        )
        boundary = torch.from_numpy(data["boundary"][:2].reshape(1, 8, 180, 360)).to(
            args.device
        )
        with torch.inference_mode():
            model.forward_once(prog, boundary, context(static, args.device))
        log(f"Captured target-free feature moments: {entry['name']}")
    for handle in handles:
        handle.remove()
    gram = sum(grams[name] / counts[name] for name in grams)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    torch.save(
        {
            "basis": eigenvectors.float(),
            "eigenvalues": eigenvalues,
            "grams": {k: grams[k] / counts[k] for k in grams},
            "windows": [entry["name"] for entry in windows(args, manifest)],
        },
        out / "carry_basis.pt",
    )
    log(
        f"Joint feature energy eigenvalues min/median/max: {eigenvalues[0]:.5g}/{eigenvalues[140]:.5g}/{eigenvalues[-1]:.5g}"
    )


def build_state(args):
    """Apply only linear-algebraic tensor edits; no training or regression."""
    original = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )["model"]
    state = {k.removeprefix("module."): v for k, v in original.items()}
    channels = [
        base + level for base in (38, 57) for level in range(args.min_level, 19)
    ]
    basis_data = torch.load(
        args.output / "carry_basis.pt", map_location="cpu", weights_only=True
    )
    if args.basis == "energy":
        basis = basis_data["basis"][:, : len(channels)]
    else:
        # Protect the decoder's tracer readout when choosing weak directions.
        # This uses weights and input feature moments, not prediction targets.
        candidates = basis_data["basis"][:, :64].double()
        decoder = state["decoder.weight"].double().clone()
        tracer_rows = [
            t * 77 + c for t in range(2) for c in range(38, 76) if c not in channels
        ]
        decoder[tracer_rows] *= 3 if args.basis == "balanced" else 30
        readout = decoder.permute(0, 2, 3, 1).reshape(-1, 280) @ candidates
        guard = readout.T @ readout
        energy = torch.diag(basis_data["eigenvalues"][:64])
        strength = 10.0 if args.basis == "guard_strong" else 1.0
        objective = energy / energy.trace() + strength * guard / guard.trace()
        _, vectors = torch.linalg.eigh(objective)
        basis = (candidates @ vectors[:, : len(channels)]).float()
    projection = torch.eye(280) - basis @ basis.T

    def project_output(prefix):
        w = state[prefix + ".weight"]
        state[prefix + ".weight"] = (projection @ w.flatten(1)).reshape_as(w)
        state[prefix + ".bias"] = projection @ state[prefix + ".bias"]

    def project_input(prefix):
        state[prefix + ".weight"] = torch.einsum(
            "oihw,ij->ojhw", state[prefix + ".weight"], projection
        )

    # The first skip block writes the reserved directions directly from x[t].
    project_output("unet.layers.0.skip_module")
    project_output("unet.layers.0.convblock.6")
    first = state["unet.layers.0.skip_module.weight"]
    first[:, [77 + c for c in channels], 0, 0] += basis

    # Exclude carried values from the downsampled branch and final nonlinear block.
    project_input("unet.layers.2.skip_module")
    project_input("unet.layers.2.convblock.0")
    project_input("unet.layers.16.convblock.0")
    # Ensure both branches added to the high-resolution skip have zero component
    # in the reserved subspace. Bilinear upsampling commutes with this projection.
    project_output("unet.layers.14.skip_module")
    project_output("unet.layers.14.convblock.6")
    project_output("unet.layers.16.convblock.6")
    project_input("decoder")
    state["decoder.bias"] = state["decoder.bias"].clone()
    rows = [t * 77 + c for t in range(2) for c in channels]
    state["decoder.weight"][rows] *= 1 - args.alpha
    state["decoder.bias"][rows] *= 1 - args.alpha
    for t in range(2):
        state["decoder.weight"][[77 * t + c for c in channels], :, 1, 1] += (
            args.alpha * basis.T
        )
    return state, basis, channels


def run(args):
    out = args.output
    tag = f"carry_{args.basis}_l{args.min_level}_{args.alpha:g}"
    manifest = json.loads((out / "data_manifest.json").read_text())
    static = np.load(out / "static.npz")
    diag = Diagnostics(static)
    model = model_from_checkpoint(args)
    state, basis, channels = build_state(args)
    model.load_state_dict(state, strict=True)
    grid = context(static, args.device)
    captured = {}

    def hook(module, inputs):
        captured["carried"] = (
            torch.einsum(
                "ik,bihw->bkhw", basis.to(args.device), inputs[0][..., 1:-1, 1:-1]
            )
            .cpu()
            .numpy()
        )

    handle = model.decoder.register_forward_pre_hook(hook)
    records = []
    for entry in windows(args, manifest):
        name = entry["name"]
        data = np.load(out / f"{name}.npz")
        start = time.monotonic()
        with torch.inference_mode():
            prog = torch.from_numpy(data["states"][:2].reshape(1, 154, 180, 360)).to(
                args.device
            )
            boundary = torch.from_numpy(
                data["boundary"][:2].reshape(1, 8, 180, 360)
            ).to(args.device)
            pred = (
                model.forward_once(prog, boundary, grid)
                .cpu()
                .numpy()
                .reshape(2, 77, 180, 360)
            )
        assert np.isfinite(pred).all()
        carry_error = float(
            np.abs(captured["carried"][0] - data["states"][1, channels]).max()
        )
        assert carry_error < 1e-4, carry_error
        np.save(out / f"{name}_{tag}.npy", pred)
        for row in diag.evaluate(pred, data["states"][2:4], data["states"][1]):
            row.update(
                window=name,
                split=entry["split"],
                candidate=tag,
                carry_max_abs_error=carry_error,
            )
            records.append(row)
        log(
            f"{name} {tag}: {time.monotonic() - start:.1f}s; carry error {carry_error:.3g}"
        )
    handle.remove()
    save_json(out / (tag + "_metrics_" + "_".join(args.splits) + ".json"), records)


def rollout(args):
    """Compare the circuit with baseline and an ideal external-blend control."""
    out = args.output
    tag = f"carry_{args.basis}_l{args.min_level}_{args.alpha:g}"
    static = np.load(out / "static.npz")
    data = np.load(out / "discovery_2015_01.npz")
    diag = Diagnostics(static)
    grid = context(static, args.device)
    state, _, channels = build_state(args)
    records = []
    for label in ("baseline", "external_blend_control", tag):
        model = model_from_checkpoint(args)
        if label == tag:
            model.load_state_dict(state, strict=True)
        prog = torch.from_numpy(data["states"][:2].reshape(1, 154, 180, 360)).to(
            args.device
        )
        predictions = []
        for step in range(6):
            offset = step * 2
            boundary = torch.from_numpy(
                data["boundary"][offset : offset + 2].reshape(1, 8, 180, 360)
            ).to(args.device)
            with torch.inference_mode():
                pred = model.forward_once(prog, boundary, grid)
                if label == "external_blend_control":
                    for t in range(2):
                        rows = [t * 77 + c for c in channels]
                        pred[:, rows] = (1 - args.alpha) * pred[
                            :, rows
                        ] + args.alpha * prog[:, [77 + c for c in channels]]
                prog = pred
            prediction = pred.cpu().numpy().reshape(2, 77, 180, 360)
            assert np.isfinite(prediction).all()
            predictions.append(prediction)
            for row in diag.evaluate(
                prediction, data["states"][offset + 2 : offset + 4], data["states"][1]
            ):
                row.update(
                    candidate=label,
                    model_call=step + 1,
                    lead_days=offset * 5 + row["lead_days"],
                )
                records.append(row)
            log(f"Rollout {label}: {step + 1}/6")
        np.save(out / f"rollout_{label}.npy", np.concatenate(predictions))
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
    # Pure deep-state persistence is a benchmark, not a candidate repair.
    for step in range(6):
        offset = 2 * step
        pred = np.repeat(data["states"][1:2], 2, axis=0)
        for row in diag.evaluate(
            pred, data["states"][offset + 2 : offset + 4], data["states"][1]
        ):
            row.update(
                candidate="persistence",
                model_call=step + 1,
                lead_days=offset * 5 + row["lead_days"],
            )
            records.append(row)
    save_json(out / f"rollout_{tag}_metrics.json", records)


def export(args):
    state, basis, channels = build_state(args)
    original = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )["model"]
    changes = {}
    for key, value in state.items():
        old = original["module." + key]
        n = int(torch.count_nonzero(value != old))
        if n:
            changes[key] = {
                "changed_values": n,
                "relative_l2_change": float(
                    (value - old).norm() / old.norm().clamp_min(1e-30)
                ),
            }
    provenance = {
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "method": "direct weight-space carry circuit; no backpropagation or target regression",
        "status": "experimental, not a guarantee of physical consistency",
        "basis": args.basis,
        "basis_source_sha256": sha256(args.output / "carry_basis.pt"),
        "alpha": args.alpha,
        "channels": channels,
        "changes": changes,
        "data_root": json.loads((args.output / "data_manifest.json").read_text())[
            "data_root"
        ],
        "calibration_windows": torch.load(
            args.output / "carry_basis.pt", weights_only=True
        )["windows"],
    }
    tag = f"carry_{args.basis}_l{args.min_level}_{args.alpha:g}"
    target = args.output / f"samudra2_onedeg_ema_{tag}.pt"
    if target.exists():
        raise FileExistsError(target)
    torch.save(
        {"model": state, "edit_provenance": provenance, "carry_basis": basis}, target
    )
    provenance["edited_checkpoint_sha256"] = sha256(target)
    save_json(args.output / f"{tag}_edit_manifest.json", provenance)
    log(
        f"Saved {target}; changed {sum(v['changed_values'] for v in changes.values())} tensor values"
    )


def verify(args):
    """Reload the exported artifact and compare with the frozen test prediction."""
    tag = f"carry_{args.basis}_l{args.min_level}_{args.alpha:g}"
    manifest = json.loads((args.output / f"{tag}_edit_manifest.json").read_text())
    assert sha256(args.checkpoint) == manifest["parent_checkpoint_sha256"]
    exported_args = copy(args)
    exported_args.checkpoint = args.output / f"samudra2_onedeg_ema_{tag}.pt"
    assert sha256(exported_args.checkpoint) == manifest["edited_checkpoint_sha256"]
    model = model_from_checkpoint(exported_args)
    static = np.load(args.output / "static.npz")
    data = np.load(args.output / "test_2019_01.npz")
    with torch.inference_mode():
        prediction = (
            model.forward_once(
                torch.from_numpy(data["states"][:2].reshape(1, 154, 180, 360)).to(
                    args.device
                ),
                torch.from_numpy(data["boundary"][:2].reshape(1, 8, 180, 360)).to(
                    args.device
                ),
                context(static, args.device),
            )
            .cpu()
            .numpy()
            .reshape(2, 77, 180, 360)
        )
    reference = np.load(args.output / f"test_2019_01_{tag}.npy")
    difference = float(np.abs(prediction - reference).max())
    np.testing.assert_allclose(prediction, reference, atol=1e-4, rtol=1e-5)
    save_json(
        args.output / "export_validation.json",
        {
            "strict_load": True,
            "export_prediction_max_abs_difference": difference,
            "original_sha256_unchanged": True,
            "checkpoint_bytes": exported_args.checkpoint.stat().st_size,
        },
    )
    log(f"Export verified; maximum prediction difference {difference:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["inspect", "run", "rollout", "export", "verify"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".LOCAL/weight-edit-original")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("samudra2_onedeg_ema_ckpt.pt")
    )
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--splits", nargs="+", default=["discovery"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-level", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--basis",
        choices=["energy", "guard", "guard_strong", "balanced"],
        default="energy",
    )
    args = parser.parse_args()
    {
        "inspect": inspect,
        "run": run,
        "rollout": rollout,
        "export": export,
        "verify": verify,
    }[args.action](args)


if __name__ == "__main__":
    main()
