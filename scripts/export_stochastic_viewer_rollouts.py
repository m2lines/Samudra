#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce checked EMA draws for three observation-initialized viewer rollouts."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from samudra.constants import build_om4_layout
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

# Training modules are provided by the frozen experiment source bundle.
from samudra.experiments.stochastic_transition_train import (  # type: ignore[import-not-found]
    build_model,
)


def save(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def checked_difference(actual, expected):
    """Compare against the previously evaluated member, not just an aggregate."""
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    result = dict(
        maximum_absolute_difference=float(np.abs(delta).max()),
        rms_difference=float(np.sqrt(np.mean(delta**2))),
        bitwise_equal=bool(np.array_equal(actual, expected)),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    return result


def validate_checkpoint(saved, args):
    contract = saved["contract"]
    if saved["step"] != args.expected_step or contract["arm"] != args.expected_arm:
        raise ValueError("Checkpoint arm/step differs from explicit expectation")
    if args.expected_contract is not None:
        if contract != json.loads(args.expected_contract.read_text()):
            raise ValueError("Checkpoint contract differs from pinned protocol")
    for key, filename in (
        ("grid", "grid.npz"),
        ("statistics", "statistics.npz"),
        ("observation_manifest", "SHA256SUMS"),
    ):
        if contract[key] != digest(args.data / filename):
            raise ValueError(f"Checkpoint data fingerprint differs: {key}")
    return contract


@torch.no_grad()
def export_om4_initializations(args, data, model, checkpoint_sha, names, units, layout):
    """Full-field reconstructions matched to the existing velocity diagnostics."""
    import cftime
    import zarr  # type: ignore[import-untyped]

    if args.om4 is None or args.om4_initializer_reference is None:
        raise ValueError(
            "OM4 initializer export requires source and reference directory"
        )
    output = args.output / "om4-initialization"
    output.mkdir(parents=True, exist_ok=True)
    store = zarr.open_consolidated(str(args.om4), mode="r")
    time = store["time"]
    labels = [
        d.strftime("%Y-%m-%dT%H:%M:%S")
        for d in cftime.num2date(
            time[:], time.attrs["units"], time.attrs.get("calendar", "standard")
        )
    ]
    mask = np.stack(
        [store[f"mask_{i}"][:] for i in range(19)] * 4 + [store["mask_0"][:]]
    ).astype(bool)
    np.testing.assert_array_equal(mask, data.grid["mask"])
    np.testing.assert_allclose(store["x"][:], data.grid["lon"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(store["y"][:], data.grid["lat"], rtol=0, atol=1e-6)
    with np.load(
        args.om4_initializer_reference / "grid.npz", allow_pickle=False
    ) as grid:
        np.testing.assert_array_equal(grid["names"], names)
        selection = grid["selection"].tolist()
    receipts = {}
    for month in ("2014-01", "2014-04", "2014-05"):
        path = args.data / "validation" / f"{month}.npz"
        if path not in data.paths("validation"):
            raise ValueError("OM4 initializer date is outside frozen validation")
        sample = data.load(path)
        wanted = (
            np.asarray(sample["raw"]["midpoints"][:19], dtype="datetime64[s]")
            .astype(str)
            .tolist()
        )
        if any(labels.count(t) != 1 for t in wanted):
            raise ValueError(f"No exact OM4 history for {month}")
        indices = [labels.index(t) for t in wanted]
        if not np.all(np.diff(indices) == 1):
            raise ValueError("OM4 history is discontinuous")
        input_receipt = json.loads(
            (args.om4_initializer_reference / f"input-{month}.json").read_text()
        )
        if (
            input_receipt["history_midpoints"] != wanted
            or input_receipt["sample_sha256"] != digest(path)
            or input_receipt["om4_metadata_sha256"] != digest(args.om4 / ".zmetadata")
            or input_receipt["seed"] != 904003
            or input_receipt["members"] != 8
        ):
            raise ValueError("OM4 initializer inputs differ from saved evaluation")
        surface = np.stack(
            [np.stack([store[n][i] for n in ("thetao_0", "zos")]) for i in indices]
        ).astype("float32")
        gold = np.stack([store[n][indices[-1]] for n in names]).astype("float32")
        if (
            not np.isfinite(surface[:, mask[[38, 76]]]).all()
            or not np.isfinite(gold[mask]).all()
        ):
            raise ValueError("Nonfinite wet OM4 fields")
        normalized = np.where(
            mask[[38, 76]],
            (surface - data.grid["mean"][[38, 76], None, None])
            / data.grid["std"][[38, 76], None, None],
            0,
        )
        valid = data.tensor(np.broadcast_to(mask[[38, 76]], surface.shape).copy())[None]
        generator = torch.Generator(device=data.device).manual_seed(904003)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond, _, _ = model.encode_native(
                data.tensor(normalized)[None],
                sample["atmosphere"][:, :19],
                sample["contexts"][:, 18],
                data.mask,
                valid,
                return_anchors=False,
            )
            cond = cond.unsqueeze(0).expand(8, *cond.shape).flatten(0, 1)
            initial = model.initialize(cond, None, None, data.mask, generator)[
                :, -1
            ].float()
        if not torch.isfinite(initial).all():
            raise ValueError("Nonfinite OM4 initialization")
        draws = initial * data.std[0, 0] + data.mean[0, 0]
        reference = (
            args.om4_initializer_reference
            / f"{args.om4_initializer_variant}-{args.expected_step:05d}-{month}"
        )
        prior = json.loads(reference.with_suffix(".json").read_text())
        if prior["checkpoint_sha256"] != checkpoint_sha or prior["weights"] != "ema":
            raise ValueError("OM4 initializer checkpoint differs from saved evaluation")
        with np.load(str(reference) + "-maps.npz", allow_pickle=False) as saved:
            checks = checked_difference(
                draws[:, selection].cpu().numpy(), saved["members"]
            )
            np.testing.assert_array_equal(
                np.where(mask[selection], gold[selection], 0), saved["gold"]
            )
        midpoint = np.asarray([wanted[-1]], dtype="datetime64[s]")
        half = np.timedelta64(216000, "s")
        metadata = dict(
            channel_names=names,
            depth=np.asarray(layout.depth_levels),
            depth_interfaces=np.r_[0.0, np.cumsum(layout.depth_thickness)],
            units=units,
            mask=mask,
            lat=data.grid["lat"],
            lon=data.grid["lon"],
            leads=np.asarray([0]),
            midpoints=midpoint.astype(str),
            starts=(midpoint - half).astype(str),
            ends=(midpoint + half).astype(str),
            origin=month,
        )
        files = {}
        for readout, member in (("mean8", None), ("draw1", 0), ("draw2", 1)):
            value = initial.mean(0) if member is None else initial[member]
            physical = (value * data.std[0, 0] + data.mean[0, 0]).cpu().numpy()[None]
            files[readout] = save(
                output / f"{month}-{readout}.npz", **metadata, fields=physical
            )
        files["gold"] = save(
            output / f"{month}-gold.npz", **metadata, fields=gold[None]
        )
        receipts[month] = dict(
            files=files,
            seed=904003,
            members=8,
            crosscheck=checks,
            source=str(args.om4),
            source_metadata_sha256=digest(args.om4 / ".zmetadata"),
            checkpoint_sha256=checkpoint_sha,
            weights="ema",
            original_maps=str(reference) + "-maps.npz",
            original_maps_sha256=digest(Path(str(reference) + "-maps.npz")),
            history_midpoints=wanted,
            input_scope="OM4 surface history plus matched ERA5; no interior or future input",
        )
        atomic_json(receipts[month], output / f"{month}-COMPLETE.json")
        del initial, draws, cond, sample
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        dict(
            complete=True,
            origins=receipts,
            fields="physical units; initialization only",
        ),
        output / "COMPLETE.json",
    )
    return receipts


def om4_comparison(store_path, output, origin, metadata):
    import cftime
    import zarr

    store = zarr.open_consolidated(str(store_path), mode="r")
    times = store["time"]
    calendar = times.attrs.get("calendar", "standard")
    dates = cftime.num2date(times[:], times.attrs["units"], calendar)
    labels = [d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates]
    wanted = metadata["midpoints"].tolist()
    missing = [label for label in wanted if labels.count(label) != 1]
    if missing:
        return dict(
            available=False,
            source=str(store_path),
            reason="No exact native five-day midpoint match; no resampling performed",
            unmatched_midpoints=missing,
        )
    indices = [labels.index(label) for label in wanted]
    np.testing.assert_allclose(store["y"][:], metadata["lat"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(store["x"][:], metadata["lon"], rtol=0, atol=1e-6)
    names = metadata["channel_names"].tolist()
    mask = np.stack(
        [
            store[f"mask_{name.rsplit('_', 1)[1]}"][:]
            if name != "zos"
            else store["mask_0"][:]
            for name in names
        ]
    ).astype(bool)
    fields = np.stack([np.stack([store[name][i] for name in names]) for i in indices])
    if not np.isfinite(fields[:, mask]).all():
        raise ValueError("Nonfinite contemporaneous OM4 on wet support")
    receipt = save(
        output / f"{origin}-om4-comparison.npz",
        **dict(metadata, fields=fields, mask=mask),
    )
    receipt.update(
        source=str(store_path),
        source_metadata_sha256=digest(store_path / ".zmetadata"),
        source_attributes=dict(store.attrs),
        calendar=calendar,
        indices=indices,
        label="Published OM4 v3 contemporaneous comparison, not observational ground truth; differs from staged Beta training-store version",
    )
    return receipt


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--om4", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-arm", choices=("A", "B", "C"), default="C")
    parser.add_argument("--expected-step", type=int, default=5000)
    parser.add_argument("--expected-contract", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--export-om4-initialization", action="store_true")
    parser.add_argument("--om4-initializer-reference", type=Path)
    parser.add_argument("--om4-initializer-variant")
    args = parser.parse_args()
    if args.checkpoint and (
        args.expected_contract is None or args.evaluation_root is None
    ):
        parser.error(
            "Explicit checkpoints require --expected-contract and --evaluation-root"
        )
    if args.export_om4_initialization and not args.om4_initializer_variant:
        parser.error("OM4 initializer export requires --om4-initializer-variant")
    torch.cuda.set_device(0)
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    paths = data.paths("validation")
    if len(paths) != 9 or [paths[i].stem for i in (0, 2, 8)] != [
        "2013-11",
        "2014-01",
        "2014-07",
    ]:
        raise ValueError("Original validation ordering changed")
    checkpoint = args.checkpoint or args.root / "production/C/step-05000.pt"
    checkpoint_sha = digest(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = validate_checkpoint(saved, args)
    counts = saved["counts"]
    model = build_model(saved, data, use_ema=True).eval()
    del saved
    args.output.mkdir(parents=True, exist_ok=True)
    names = data.grid["names"]
    layout = build_om4_layout()
    units = np.asarray(
        [
            "m"
            if n == "zos"
            else "degree_Celsius"
            if n.startswith("thetao")
            else "psu"
            if n.startswith("so_")
            else "m s-1"
            for n in names
        ]
    )
    receipts = {}
    for index in (0, 2, 8):
        path = paths[index]
        origin = path.stem
        sample = data.load(path)
        original = (
            args.evaluation_root or args.root / "production/C/evaluation/step-05000-ema"
        ) / f"members-{origin}.npz"
        completion = json.loads((original.parent / "COMPLETE.json").read_text())
        if (
            completion["checkpoint_sha256"] != checkpoint_sha
            or completion["weights"] != "ema"
            or completion["step"] != args.expected_step
            or completion.get("qualification", False)
        ):
            raise ValueError("Original maps checkpoint differs")
        seed = 904001 + index
        generator = torch.Generator(device=data.device).manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            draws, initial = model.forecast(
                sample["surface"],
                sample["atmosphere"],
                sample["contexts"],
                data.mask,
                sample["validity"],
                generator=generator,
                members=8,
            )
        if not torch.isfinite(draws).all() or not torch.isfinite(initial).all():
            raise ValueError("Nonfinite forecast")
        checks = {}
        with np.load(original, allow_pickle=False) as old:
            channels = [names.tolist().index(n) for n in old["channel_names"].tolist()]
            for phase, actual in (
                ("day30", draws[:, 0, 5, channels]),
                ("initial", initial[:, 0, -1, channels]),
            ):
                actual = actual.float().cpu().numpy()
                expected = old[phase]
                checks[phase] = checked_difference(actual, expected)
                print(
                    json.dumps(dict(origin=origin, crosscheck=phase, **checks[phase])),
                    flush=True,
                )
        midpoints = np.asarray(sample["raw"]["midpoints"][18:25], dtype="datetime64[s]")
        half = np.timedelta64(216000, "s")
        metadata = dict(
            channel_names=names,
            depth=np.asarray(layout.depth_levels),
            depth_interfaces=np.r_[0.0, np.cumsum(layout.depth_thickness)],
            units=units,
            mask=data.grid["mask"],
            lat=data.grid["lat"],
            lon=data.grid["lon"],
            leads=np.arange(0, 31, 5),
            midpoints=midpoints.astype(str),
            starts=(midpoints - half).astype(str),
            ends=(midpoints + half).astype(str),
            origin=origin,
        )
        files = {}
        for readout, member in (("mean8", None), ("draw1", 0), ("draw2", 1)):
            # Reduce in normalized FP32 exactly as original evaluation, then
            # convert units. Members retain their stochastic recurrent paths.
            if member is None:
                init = initial[:, 0, -1].float().mean(0)
                future = draws[:, 0, :6].float().mean(0)
            else:
                init = initial[member, 0, -1].float()
                future = draws[member, 0, :6].float()
            normalized = torch.cat((init[None], future)).cpu().numpy()
            physical = (
                normalized * data.grid["std"][None, :, None, None]
                + data.grid["mean"][None, :, None, None]
            )
            files[readout] = save(
                args.output / f"{origin}-{readout}.npz",
                **metadata,
                fields=physical,
                surface_reference=sample["raw"]["surface"][18:25],
                surface_reference_names=np.asarray(["thetao_0", "zos"]),
            )
            del normalized, physical
        # Monthly references are context, not instantaneous interior labels.
        files["monthly_context"] = save(
            args.output / f"{origin}-monthly-context.npz",
            interior=sample["raw"]["interior"],
            channel_names=names[data.ts_indices].reshape(2, 14),
            depth=np.asarray(layout.depth_levels[:14]),
            ohc=sample["raw"]["ohc"],
            origin=origin,
            context="Calendar-month IAP reference, not instantaneous state truth",
        )
        om4 = (
            om4_comparison(args.om4, args.output, origin, metadata)
            if args.om4
            else None
        )
        receipts[origin] = dict(
            seed=seed,
            original_validation_index=index,
            members=8,
            crosschecks=checks,
            original_maps=str(original),
            original_maps_sha256=digest(original),
            files=files,
            om4=om4,
        )
        atomic_json(receipts[origin], args.output / f"{origin}-COMPLETE.json")
        del draws, initial, sample
        gc.collect()
        torch.cuda.empty_cache()
    om4_initializations = (
        export_om4_initializations(
            args, data, model, checkpoint_sha, names, units, layout
        )
        if args.export_om4_initialization
        else {}
    )
    atomic_json(
        dict(
            complete=True,
            arm=args.expected_arm,
            variant=args.variant or args.expected_arm,
            weights="ema",
            step=args.expected_step,
            counts=counts,
            contract=contract,
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,
            fields="physical units; apply depth-dependent mask",
            origins=receipts,
            om4_initializations=om4_initializations,
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
