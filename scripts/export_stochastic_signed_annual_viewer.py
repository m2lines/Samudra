#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export the signed direct-ERA5 8k model with coherent initial-condition ensembles."""

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.constants import build_om4_layout
from samudra.experiments.observation_annual import inputs, month_weights, read_origin
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.stochastic_transition_train import (  # type: ignore[import-not-found]
    build_model,
)

LEADS = [5, 15, 30, 90, 180, 365]
SOURCE = "b4f8089226e75d0a8261c5c07a92890aa612ac7e952be896f6bfb66bd51da01c"
CHECKPOINT_SHA = "e28fb15f5e71cbb22f45cd5c974c518b296b74563add6ddc7763a605f2e82200"
READOUTS = ("mean8", "draw1")


def stream_forecast(model, arguments, generator, members=8):
    """Exact forecast algebra/RNG order, retaining only current two-state memory."""
    surface, atmosphere, contexts, mask, validity = arguments
    batch = surface.shape[0]

    def repeat(value):
        return (
            None
            if value is None
            else value.unsqueeze(0).expand(members, *value.shape).flatten(0, 1)
        )

    conditioning, known, anchor = model.encode_native(
        surface[:, : model.history],
        atmosphere[:, : model.history],
        contexts[:, model.history - 1],
        mask,
        validity[:, : model.history],
        return_anchors=False,
    )
    states = model.initialize(
        repeat(conditioning), repeat(known), repeat(anchor), mask, generator
    )
    yield 0, states[:, -1].reshape(members, batch, model.channels, *states.shape[-2:])
    for step in range(model.history, atmosphere.shape[1]):
        states = model.advance(
            states,
            repeat(atmosphere[:, step]),
            repeat(contexts[:, step]),
            mask,
            generator,
        )
        yield (
            (step - model.history + 1) * 5,
            states[:, -1].reshape(members, batch, model.channels, *states.shape[-2:]),
        )


def save(path, **arrays):
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def om4_indices(store, raw):
    import cftime

    t = store["time"]
    dates = cftime.num2date(t[:], t.attrs["units"], t.attrs.get("calendar", "standard"))
    labels = [d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates]
    wanted = np.asarray(raw["midpoint"], dtype="datetime64[s]").astype(str).tolist()
    if any(labels.count(d) != 1 for d in wanted):
        missing = [d for d in wanted if labels.count(d) != 1]
        raise ValueError(f"OM4 exact annual interval alignment unavailable: {missing}")
    return [labels.index(d) for d in wanted]


def metadata(data, raw):
    layout = build_om4_layout()
    starts = np.asarray(raw["start"][19:], dtype="datetime64[s]")
    initial_start = np.datetime64(raw["start"][18], "s")
    return dict(
        channel_names=data.grid["names"],
        mask=data.grid["mask"],
        lat=data.grid["lat"],
        lon=data.grid["lon"],
        depth=np.asarray(layout.depth_levels),
        depth_interfaces=np.r_[0.0, np.cumsum(layout.depth_thickness)],
        leads=np.arange(5, 366, 5),
        interior_leads=np.asarray(LEADS),
        starts=starts.astype(str),
        ends=(starts + np.timedelta64(5, "D")).astype(str),
        initial_start=str(initial_start),
        initial_end=str(initial_start + np.timedelta64(5, "D")),
        units=np.asarray(
            [
                "m"
                if n == "zos"
                else "degree_Celsius"
                if n.startswith("thetao")
                else "psu"
                if n.startswith("so_")
                else "m s-1"
                for n in data.grid["names"]
            ]
        ),
    )


def physical(data, values):
    return values * data.grid["std"][:, None, None] + data.grid["mean"][:, None, None]


def monthly_ohc(data, monthly):
    # Use the established complete-column OHC integration, unchanged.
    return data.ohc(torch.as_tensor(monthly, device=data.device))


def references(directory, description):
    refs = []
    for record in description["monthly_interiors"]:
        p = directory / record["file"]
        if p.name != record["file"] or digest(p) != record["sha256"]:
            raise ValueError("IAP target receipt mismatch")
        with np.load(p, allow_pickle=False) as saved:
            refs.append((p.name[:7], saved["values"], saved["ohc"]))
    if len(refs) != 12:
        raise ValueError("Expected twelve monthly interior references")
    return refs


@torch.no_grad()
def run_origin(model, data, arguments, raw, output, seed, months, *, label):
    weights = np.stack(
        [month_weights(raw["start"][19:], month) for month in months]
    ).astype(np.float32)
    # Independent short call must exactly reproduce the streaming prefix.
    short = list(arguments)
    short[1] = short[1][:, : model.history + 6]
    short[2] = short[2][:, : model.history + 6]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, expected_init = model.forecast(
            *short,
            generator=torch.Generator(device=data.device).manual_seed(seed),
            members=8,
        )
    expected = expected.float().cpu()
    expected_init = expected_init[:, :, -1].float().cpu()
    monthly = np.zeros((2, 12, 77, *data.grid["mask"].shape[-2:]), dtype=np.float32)
    surfaces: list[list[np.ndarray]] = [[], []]
    selected: list[list[np.ndarray]] = [[], []]
    initials = []
    completed = 0
    failure = None
    prefix_max = 0.0
    generator = torch.Generator(device=data.device).manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for lead, states in stream_forecast(model, arguments, generator):
            states = states.float()
            if not torch.isfinite(states).all():
                failure = dict(
                    lead=lead,
                    reason="first nonfinite recurrent state; no clipping or damping applied",
                )
                save(output / "NONFINITE.npz", lead=lead, state=states.cpu().numpy())
                break
            if lead <= 30:
                reference = (
                    expected_init if lead == 0 else expected[:, :, lead // 5 - 1]
                )
                actual = states.cpu()
                prefix_max = max(prefix_max, float((actual - reference).abs().max()))
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            readouts = torch.stack((states.mean(0)[0], states[0, 0])).cpu().numpy()
            if lead == 0:
                initials = physical(data, readouts)
                continue
            completed = lead // 5
            for j, state in enumerate(readouts):
                field = physical(data, state)
                surfaces[j].append(field[[38, 76]])
                if lead in LEADS:
                    selected[j].append(field)
                monthly[j] += weights[:, completed - 1, None, None, None] * state
    completed_months = np.sum(weights[:, :completed], axis=1) >= 1 - 1e-6
    monthly[:, ~completed_months] = np.nan
    files = {}
    meta = metadata(data, raw)
    meta["leads"] = meta["leads"][:completed]
    meta["starts"] = meta["starts"][:completed]
    meta["ends"] = meta["ends"][:completed]
    meta["interior_leads"] = np.asarray(
        [lead for lead in LEADS if lead <= completed * 5]
    )
    for j, name in enumerate(READOUTS):
        state_months = physical(data, monthly[j])
        ohc = np.full((12, 2, *data.grid["mask"].shape[-2:]), np.nan, dtype=np.float32)
        if completed_months.any():
            ohc[completed_months] = monthly_ohc(data, monthly[j, completed_months])
        shape = data.grid["mask"].shape
        files[name] = save(
            output / f"{name}.npz",
            **meta,
            initial=initials[j]
            if len(initials)
            else np.full(shape, np.nan, dtype=np.float32),
            surface=np.stack(surfaces[j])
            if completed
            else np.empty((0, 2, *shape[-2:]), dtype=np.float32),
            forecast_interior=np.stack(selected[j])
            if selected[j]
            else np.empty((0, *shape), dtype=np.float32),
            months=np.asarray(months),
            monthly_ts=state_months[:, 38:76].reshape(12, 2, 19, *shape[-2:]),
            monthly_ohc=ohc,
            complete=completed == 73,
            source=label,
        )
    return dict(
        files=files,
        complete=completed == 73,
        completed_bins=completed,
        first_nonfinite=failure,
        seed=seed,
        members=8,
        prefix_max_absolute_difference=prefix_max,
        monthly_weights=weights.tolist(),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "data", "annual-data", "output", "om4"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    import hashlib

    import zarr  # type: ignore[import-untyped]

    manifest = json.loads((args.root / "SOURCE.json").read_text())
    if (
        manifest["sha256"] != SOURCE
        or hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True).encode()
        ).hexdigest()
        != SOURCE
    ):
        raise ValueError("Unexpected frozen training source")
    for name, expected in manifest["files"].items():
        if digest(args.root / "code" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    store = zarr.open_consolidated(str(args.om4), mode="r")
    inputs_by_origin = []
    for year in (2015, 2018, 2021):
        directory = args.annual_data / "test" / f"{year}-01-01"
        description, raw = read_origin(directory)
        try:
            indices = om4_indices(store, raw)
        except ValueError as error:
            indices = dict(available=False, reason=str(error))
        inputs_by_origin.append((directory, description, raw, indices))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "exact_om4_alignment": {
                        d["origin"]: not isinstance(i, dict)
                        for _, d, _, i in inputs_by_origin
                    },
                    "origins": [d["origin"] for _, d, _, _ in inputs_by_origin],
                }
            )
        )
        return
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    for coordinate, key in (("y", "lat"), ("x", "lon")):
        np.testing.assert_allclose(
            store[coordinate][:], data.grid[key], rtol=0, atol=1e-6
        )
    args.output.mkdir(parents=True, exist_ok=True)
    all_receipts = {}
    base_data = data
    for source, step in (("obs", 8000), ("om4", 8000)):
        data = copy.copy(base_data)
        data.grid = dict(base_data.grid)
        if source == "om4":
            data.grid["mask"] = np.stack(
                [store[f"mask_{i}"][:] for i in range(19)] * 4 + [store["mask_0"][:]]
            ).astype(bool)
            np.testing.assert_array_equal(data.grid["mask"], base_data.grid["mask"])
            data.mask = data.tensor(data.grid["mask"])
        checkpoint = (
            args.root / "production/B-shared-deep-wide-signed" / f"step-{step:05d}.pt"
        )
        if digest(checkpoint) != CHECKPOINT_SHA:
            raise ValueError("Final checkpoint fingerprint differs")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        contract = saved["contract"]
        expected = dict(
            producer=SOURCE,
            forcing_mode="direct",
            om4_window_mode="native-30day",
            arm="B",
            main_updates=8000,
            effective_batch=8,
            training_members=4,
            velocity_init_signed_crps_weight=1.0,
        )
        if (
            saved["step"] != 8000
            or saved["counts"] != {"observation": 4000, "om4": 4000}
            or any(contract.get(k) != v for k, v in expected.items())
        ):
            raise ValueError("Checkpoint training contract differs")
        for key, filename in (
            ("grid", "grid.npz"),
            ("statistics", "statistics.npz"),
            ("observation_manifest", "SHA256SUMS"),
        ):
            if contract[key] != digest(args.data / filename):
                raise ValueError(f"Observation input fingerprint differs: {key}")
        model = build_model(saved, data, use_ema=True).eval()
        if model.arm != "B":
            raise ValueError(
                "Expected stochastic initializer and deterministic processor"
            )
        del saved
        for number, (directory, description, raw, indices) in enumerate(
            inputs_by_origin
        ):
            if source == "om4" and isinstance(indices, dict):
                all_receipts[f"{source}-{step}/{description['origin']}"] = indices
                continue
            output = args.output / f"{source}-{step}" / str(description["origin"])[:10]
            output.mkdir(parents=True, exist_ok=True)
            current = dict(raw)
            names = data.grid["names"].tolist()
            if source == "om4":
                current["surface"] = np.stack(
                    [
                        np.stack([store[n][i] for n in ("thetao_0", "zos")])
                        for i in indices
                    ]
                )
                omask = store["mask_0"][:].astype(bool)
                current["surface"] = np.where(omask, current["surface"], np.nan)
            argument = inputs(data, current)
            refs = references(directory, description)
            months = [r[0] for r in refs]
            result = run_origin(
                model,
                data,
                argument,
                current,
                output,
                904001 + number,
                months,
                label=source,
            )
            result.update(
                checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint),
                weights="ema",
                step=step,
                annual_manifest_sha256=digest(directory / "COMPLETE.json"),
                forcing="Prescribed shared ERA5; future surface observations never supplied",
                initialization="surface-history only; no gold ocean interior input",
            )
            context_month = str(pd.Period(str(description["origin"])[:7], freq="M") - 1)
            context_candidates = [
                args.data / split / f"{context_month}.npz"
                for split in ("train", "validation", "test")
            ]
            initial_context = {}
            for context_path in context_candidates:
                if context_path.is_file():
                    with np.load(context_path, allow_pickle=False) as saved_context:
                        initial_context = dict(
                            initial_iap_ts=saved_context["interior"],
                            initial_context_month=context_month,
                        )
                    result["initial_iap_context"] = dict(
                        path=str(context_path),
                        sha256=digest(context_path),
                        month=context_month,
                    )
                    break
            if not initial_context:
                result["initial_iap_context"] = dict(
                    available=False,
                    month=context_month,
                    reason="Preceding-month IAP context not present in prepared observation cohorts",
                )
            reference_file = save(
                output / "references.npz",
                **initial_context,
                reference_surface=raw["surface"][19:],
                reference_monthly_ts=np.stack([r[1] for r in refs]),
                reference_monthly_ohc=np.stack([r[2] for r in refs]),
                months=np.asarray(months),
            )
            result["references"] = reference_file
            if source == "om4":
                weights = np.asarray(result["monthly_weights"], dtype=np.float32)
                monthly = np.zeros(
                    (12, 77, *data.grid["mask"].shape[-2:]), dtype=np.float32
                )
                gold_initial = None
                gold_selected = []
                gold_surface = []
                for lead, index in enumerate(indices[18:]):
                    field = np.stack([store[name][index] for name in names]).astype(
                        np.float32
                    )
                    if not np.isfinite(field[data.grid["mask"].astype(bool)]).all():
                        raise ValueError(
                            "Nonfinite exact OM4 gold on declared wet mask"
                        )
                    normalized = np.where(
                        data.grid["mask"],
                        (field - data.grid["mean"][:, None, None])
                        / data.grid["std"][:, None, None],
                        0,
                    )
                    if lead == 0:
                        gold_initial = field
                    else:
                        monthly += weights[:, lead - 1, None, None, None] * normalized
                        gold_surface.append(field[[38, 76]])
                        if lead * 5 in LEADS:
                            gold_selected.append(field)
                result["gold"] = save(
                    output / "gold.npz",
                    **metadata(data, raw),
                    initial=gold_initial,
                    surface=np.stack(gold_surface),
                    forecast_interior=np.stack(gold_selected),
                    months=np.asarray(months),
                    monthly_ts=physical(data, monthly)[:, 38:76].reshape(
                        12, 2, 19, *data.grid["mask"].shape[-2:]
                    ),
                    monthly_ohc=monthly_ohc(data, monthly),
                    source="om4",
                )
                result["om4_source"] = dict(
                    path=str(args.om4),
                    metadata_sha256=digest(args.om4 / ".zmetadata"),
                    mask_matches_training=True,
                    indices=indices,
                    note="Published Torch OM4 source version; exact-time native reference, differs from staged Beta metadata",
                )
            if digest(checkpoint) != CHECKPOINT_SHA:
                raise ValueError("Checkpoint changed during export")
            atomic_json(result, output / "COMPLETE.json")
            print(
                json.dumps(
                    dict(
                        origin=description["origin"],
                        source=source,
                        completed_bins=result["completed_bins"],
                        files=list(result["files"]),
                    )
                ),
                flush=True,
            )
            all_receipts[f"{source}-{step}/{description['origin']}"] = result
            del argument
            gc.collect()
            torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        dict(
            exports=all_receipts,
            complete=True,
            scope="Final signed direct-ERA5 8k EMA; same checkpoint for OBS and exact-time OM4 surface initializations",
            checkpoint_sha256=CHECKPOINT_SHA,
            source_sha256=SOURCE,
            export_script_sha256=digest(Path(__file__)),
            readouts=list(READOUTS),
            stochasticity="Initialization only; each member evolved by deterministic processor",
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
