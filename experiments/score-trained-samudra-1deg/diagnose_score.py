# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""One continuous held-out trajectory per arm, with streaming structure diagnostics.

Run with four torchrun ranks: rank 0/1/2/3 owns A/B/C/D. Stochastic arms use
four members. Fields are standardized unless explicitly named physical. This
does not change or replace the primary 16-member repeated-forecast evaluation.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
import xarray as xr
from evaluate_score import field_metrics
from native_pointwise import configure_native_pointwise
from score_model import reduce_field, weights
from train_multitask import MultitaskTrainConfig
from train_score import ScoreTrainConfig

from samudra.backend import init_train_backend
from samudra.datasets import InferenceDataset
from samudra.utils.output import ModelInferenceOutput
from samudra.utils.writer import ZarrWriter


def spatial_diagnostics(fields, area):
    """Per-field moments and zonal structure functions, never cross land pairs.

    fields: [sample, channel, y, x]; area: [channel, y, x].
    Spectral power is not inferred from a land-zero-filled Fourier transform.
    """
    denominator = area.sum((-2, -1)).clamp_min(1e-12)
    mean = (fields * area).sum((-2, -1)) / denominator
    centered = fields - mean[..., None, None]
    result = {
        "mean": mean,
        "spatial_std": ((centered.square() * area).sum((-2, -1)) / denominator)
        .clamp_min(0)
        .sqrt(),
    }
    for lag in (1, 4, 16):
        joint = torch.minimum(area, area.roll(lag, -1))
        delta = fields - fields.roll(lag, -1)
        result[f"zonal_increment_rms_{lag}"] = (
            (delta.square() * joint).sum((-2, -1))
            / joint.sum((-2, -1)).clamp_min(1e-12)
        ).sqrt()
    return result


def paired_diagnostics(fields, area):
    """Adjacent-depth differences and same-depth physical T/S and U/V covariance."""
    result = {}
    for base, offset in (("uo", 0), ("vo", 19), ("thetao", 38), ("so", 57)):
        x, y = fields[:, offset : offset + 18], fields[:, offset + 1 : offset + 19]
        w = torch.minimum(area[offset : offset + 18], area[offset + 1 : offset + 19])
        result[f"{base}_adjacent_depth_rms"] = (
            ((y - x).square() * w).sum((-2, -1)) / w.sum((-2, -1)).clamp_min(1e-12)
        ).sqrt()
    for name, first, second in (("thetao_so", 38, 57), ("uo_vo", 0, 19)):
        x, y = fields[:, first : first + 19], fields[:, second : second + 19]
        w = torch.minimum(area[first : first + 19], area[second : second + 19])
        w = w / w.sum((-2, -1), keepdim=True).clamp_min(1e-12)
        mx, my = (x * w).sum((-2, -1)), (y * w).sum((-2, -1))
        result[f"{name}_covariance"] = (
            (x - mx[..., None, None]) * (y - my[..., None, None]) * w
        ).sum((-2, -1))
    return result


def regional_weights(ctx):
    lat, lon = [v.to(ctx.label_mask.device) for v in ctx.output_resolution_cpu]
    lon = lon.remainder(360)
    area = weights(ctx, 77, area=True)
    return {
        "north_atlantic": area
        * ((lat >= 20) & (lat <= 65))[None, :, None]
        * ((lon >= 280) | (lon <= 10))[None, None, :],
        "tropical_pacific": area
        * (lat.abs() <= 15)[None, :, None]
        * ((lon >= 120) & (lon <= 290))[None, None, :],
        "southern_ocean": area * ((lat >= -65) & (lat <= -45))[None, :, None],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--arm", choices=["F", "M"])
    parser.add_argument("--write-rollouts", action="store_true")
    parser.add_argument("--verify-replay", type=Path)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    device, dist = init_train_backend("auto" if args.arm else "nccl")
    config_type: type[ScoreTrainConfig] | type[MultitaskTrainConfig]
    cfg: ScoreTrainConfig | MultitaskTrainConfig
    if args.arm:
        if dist is not None:
            raise ValueError(
                "F/M continuous evaluation requires direct single-process execution"
            )
        arm = args.arm
        config_type = MultitaskTrainConfig
        mode = {"F": "forecast", "M": "multitask"}[arm]
        cfg = config_type.from_yaml(
            args.campaign / "configs" / f"train-{mode}-rtx.yaml"
        )
        if cfg.model.mode != mode:
            raise ValueError("Mismatched arm/config")
        configure_native_pointwise(cfg.native_pointwise)
    else:
        if dist is None or dist.world_size != 4 or dist.rank is None:
            raise ValueError("Use four ranks, one per arm")
        arm = "ABCD"[dist.rank]
        config_type = ScoreTrainConfig
        cfg = config_type.from_yaml(
            args.campaign / "configs" / f"train-{'B' if arm == 'A' else arm}.yaml"
        )
    raw = cfg.model_dump(mode="json")
    raw["data"]["sources"][0]["inference_time"] = {
        "start": "2014-10-10",
        "end": "2022-12-24",
    }
    cfg = config_type(**raw)
    bundle = cfg.data.build(
        data_root=cfg.experiment.resolved_data_root, include_inference=True
    )
    source = bundle.inference_source
    if source is None:
        raise ValueError("No held-out source")
    names = list(bundle.data_layout.prognostic_var_names)
    expected = [f"{v}_{z}" for v in ("uo", "vo", "thetao", "so") for z in range(19)] + [
        "zos"
    ]
    assert names == expected
    dataset = InferenceDataset(
        source,
        names,
        bundle.data_layout.boundary_var_names,
        2,
        2,
        cfg.data.normalize_before_mask,
        cfg.data.masked_fill_value,
        False,
    ).to(device)
    net = (
        cfg.model.build(
            prog_channels=154,
            boundary_channels=6,
            out_channels=154,
            input_steps=2,
            grid_sizes=[source.grid_size],
        )
        .to(device)
        .eval()
    )
    checkpoint_path = (
        args.campaign / "outputs" / f"train-{arm}" / "saved_nets" / "ema_ckpt.pt"
    )
    if args.arm:
        checkpoint_path = (
            args.campaign
            / "outputs"
            / f"train-{mode}-rtx"
            / "saved_nets"
            / "ema_ckpt.pt"
        )
    if arm == "A":
        checkpoint_path = (
            args.campaign.parent
            / "current-stochastic-depth-1deg-20261006/control/output/current-1deg-control-ga2-210664/saved_nets/ema_ckpt.pt"
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 70
    state = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    (net.base if arm == "A" else net).load_state_dict(state, strict=True)
    del ckpt, state
    with checkpoint_path.open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output = args.output / arm
    output.mkdir(parents=True, exist_ok=False)
    cfg.save_yaml(output / "config.yaml")
    members = 1 if arm in "AB" else 4
    steps = (
        len(dataset) if args.max_steps is None else min(len(dataset), args.max_steps)
    )
    if args.verify_replay:
        reference_manifest = json.loads(
            (args.verify_replay / arm / "manifest.json").read_text()
        )
        assert reference_manifest["checkpoint_sha256"] == checkpoint_digest
        assert reference_manifest["members"] == members
        assert reference_manifest["seed"] == 55000000
        assert reference_manifest["calls"] == steps
    writers = {}
    if args.write_rollouts:
        slots = (
            ["mean"]
            if members == 1
            else ["mean"] + [f"member{i}" for i in range(members)]
        )
        for slot_name in slots:
            directory = output / slot_name
            directory.mkdir()
            writers[slot_name] = ZarrWriter(
                directory,
                dataset.get_coords_dict(),
                2,
                checkpoint_path,
                10,
                dataset.preprocessor,
                bundle.data_layout,
            )
    run = wandb.init(
        entity=cfg.experiment.wandb.entity,
        project=cfg.experiment.wandb.project,
        name=f"score-{arm}-continuous",
        mode="online",
        dir=str(output),
        config={
            "arm": arm,
            "members": members,
            "steps": steps,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "seed": 55000000,
            "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
            "write_rollouts": args.write_rollouts,
            "verify_replay": str(args.verify_replay) if args.verify_replay else None,
        },
    )
    print(f"{arm} online continuous evaluation: {run.url}", flush=True)
    # All online runs must initialize successfully before any forecast starts.
    if dist:
        torch.distributed.barrier()
    stats = source.statistics(names)
    mean = torch.as_tensor(stats.mean, device=device).reshape(77, 1, 1)
    std = torch.as_tensor(stats.std, device=device).reshape(77, 1, 1)
    wet = dataset.ctx.label_mask[:77].float()
    area = weights(dataset.ctx, 77, area=True)
    regions = regional_weights(dataset.ctx)
    thickness = torch.tensor(bundle.data_layout.depth_thickness, device=device).reshape(
        19, 1, 1
    )
    ts_volume = area[38:57] * thickness
    salinity_volume = area[57:76] * thickness
    uv_volume = torch.minimum(area[:19], area[19:38]) * thickness
    torch.manual_seed(55000000)
    initial = dataset[0][0].to(device)
    history = initial[None].expand(members, *initial.shape)
    previous = history[:, 0, -77:].clone()
    previous_true = initial[0, -77:].clone()
    origin_time = source.time.values[1]
    times, score_rows, global_rows = [], [], []
    arrays: dict[str, list[np.ndarray]] = {}
    selected = [names.index(v) for v in ("zos", "thetao_0", "so_9", "uo_9")]
    for step in range(steps):
        pred = net.sample_step(
            history, dataset.get_boundary(step).to(device), dataset.ctx
        )
        if not torch.isfinite(pred).all():
            raise FloatingPointError(f"{arm} nonfinite forecast at {step=}")
        history = pred
        targets = dataset.inference_target(step).to(device)
        for frame in (0, 1):
            p = pred[:, 0, frame * 77 : (frame + 1) * 77]
            target = targets[0, frame * 77 : (frame + 1) * 77]
            nominal = 10 * step + 5 * (frame + 1)
            target_time = source.time.values[2 * step + 2 + frame]
            times.append(str(target_time))
            elapsed_days = (target_time - origin_time).total_seconds() / 86400
            scores = field_metrics(p[:, None], target[None], area)
            for name, region in regions.items():
                value = reduce_field((p.mean(0)[None] - target).square(), region).sqrt()
                scores[f"rmse_{name}"] = torch.where(
                    region.sum((-2, -1)) > 0, value, torch.nan
                )
            for channel, name in enumerate(names):
                score_rows.append(
                    {
                        "arm": arm,
                        "target": str(target_time),
                        "nominal_days": nominal,
                        "elapsed_days": elapsed_days,
                        "variable": name,
                        **{k: float(v[channel]) for k, v in scores.items()},
                    }
                )
            # Slot order is fixed: truth, ensemble mean, then individual members.
            fields = torch.cat((target[None], p.mean(0, keepdim=True), p))
            physical = (fields * std + mean) * wet
            moments = spatial_diagnostics(physical, area)
            moments.update(paired_diagnostics(physical, area))
            changes = (
                torch.cat(
                    (
                        (target - previous_true)[None],
                        (p - previous).mean(0, keepdim=True),
                        p - previous,
                    )
                )
                * std
            )
            moments["temporal_increment_rms"] = (
                (changes.square() * area).sum((-2, -1))
                / area.sum((-2, -1)).clamp_min(1e-12)
            ).sqrt()
            for key, value in moments.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(
                        f"Nonfinite diagnostic: {arm} {nominal=} {key}"
                    )
                arrays.setdefault(key, []).append(value.cpu().numpy())
            # Fixed full layers and cos(latitude) areas are proxies, not exact OM4 partial-cell content.
            bulk = {
                "volume_mean_thetao": (physical[:, 38:57] * ts_volume).sum((1, 2, 3))
                / ts_volume.sum(),
                "volume_mean_so": (physical[:, 57:76] * salinity_volume).sum((1, 2, 3))
                / salinity_volume.sum(),
                "volume_mean_ke": (
                    (physical[:, :19].square() + physical[:, 19:38].square())
                    * 0.5
                    * uv_volume
                ).sum((1, 2, 3))
                / uv_volume.sum(),
            }
            for slot in range(len(fields)):
                global_rows.append(
                    {
                        "arm": arm,
                        "target": str(target_time),
                        "nominal_days": nominal,
                        "elapsed_days": elapsed_days,
                        "slot": slot,
                        **{k: float(v[slot]) for k, v in bulk.items()},
                    }
                )
            previous, previous_true = p, target
            if nominal in (5, 40, 360, 1440) or (step == steps - 1 and frame == 1):
                if args.verify_replay:
                    with np.load(
                        args.verify_replay / arm / f"fields-{nominal:04d}d.npz"
                    ) as saved:
                        np.testing.assert_array_equal(
                            saved["mask"], wet[selected].cpu().numpy()
                        )
                        assert str(saved["target_time"]) == str(target_time)
                        np.testing.assert_allclose(
                            physical[:, selected].cpu().numpy(),
                            saved["physical"],
                            rtol=0,
                            atol=0,
                        )
                np.savez_compressed(
                    output / f"fields-{nominal:04d}d.npz",
                    variables=np.array([names[i] for i in selected]),
                    physical=physical[:, selected].cpu().numpy(),
                    mask=wet[selected].cpu().numpy(),
                    target_time=str(target_time),
                    nominal_days=nominal,
                    elapsed_days=elapsed_days,
                )
        for slot_name, zarr_writer in writers.items():
            value = (
                pred.mean(0)
                if slot_name == "mean"
                else pred[int(slot_name.removeprefix("member"))]
            )
            zarr_writer.record_batch(
                ModelInferenceOutput(value, targets, dataset.get_target_time(step, 1))
            )
            if (step + 1) % 5 == 0 or step == steps - 1:
                zarr_writer.write()
        if step % 36 == 0 or step == steps - 1:
            run.log(
                {
                    "nominal_days": nominal,
                    "mean_channel_rmse": float(scores["rmse"].mean()),
                    "mean_channel_crps": float(scores["crps_empirical"].mean()),
                }
            )
            print(f"{arm} continuous call {step + 1}/{steps}", flush=True)
    # Read saved coordinates and fixed-date fields back before claiming export success.
    for slot_name, zarr_writer in writers.items():
        slot_index = (
            1 if slot_name == "mean" else 2 + int(slot_name.removeprefix("member"))
        )
        with xr.open_zarr(zarr_writer.pred_path, chunks={}) as saved_rollout:
            assert saved_rollout.sizes["time"] == 2 * steps
            assert [str(t) for t in saved_rollout.time.values] == times
            for fields_path in sorted(output.glob("fields-*d.npz")):
                with np.load(fields_path) as saved_fields:
                    time_index = int(saved_fields["nominal_days"]) // 5 - 1
                    for channel, name in enumerate(saved_fields["variables"]):
                        family, _, depth = str(name).partition("_")
                        stored = saved_rollout[family].isel(time=time_index)
                        if depth:
                            stored = stored.isel(lev=int(depth))
                        mask = saved_fields["mask"][channel].astype(bool)
                        np.testing.assert_allclose(
                            stored.values[mask],
                            saved_fields["physical"][slot_index, channel][mask],
                            rtol=2e-6,
                            atol=1e-7,
                        )
    for name, rows in (("scores", score_rows), ("global", global_rows)):
        with gzip.open(output / f"{name}.csv.gz", "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(
        output / "structure.npz",
        **{k: np.stack(v) for k, v in arrays.items()},
        times=np.array(times),
        variables=np.array(names),
        slots=np.array(["truth", "mean"] + [f"member{i}" for i in range(members)]),
    )
    manifest = {
        "arm": arm,
        "members": members,
        "calls": steps,
        "origin": str(origin_time),
        "last_target": times[-1],
        "variables": names,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 70,
        "checkpoint_sha256": checkpoint_digest,
        "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
        "wandb_url": run.url,
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "seed": 55000000,
        "container": os.environ["SAMUDRA_CONTAINER_IMAGE_REF"],
        "data_root": str(cfg.experiment.resolved_data_root),
        "normalization_mean": mean.flatten().cpu().tolist(),
        "normalization_std": std.flatten().cpu().tolist(),
        "depth_thickness": list(bundle.data_layout.depth_thickness),
        "bulk_geometry": "cos(latitude), binary wet masks, full nominal layers; content proxies only",
        "regional_empty_support": "NaN, never zero skill",
        "slots": ["truth", "mean"] + [f"member{i}" for i in range(members)],
        "partial": args.max_steps is not None,
        "rollout_slots": list(writers),
        "replay_verified_against": str(args.verify_replay)
        if args.verify_replay
        else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run.finish()
    (output / "COMPLETE.json").write_text(
        json.dumps({"frames": 2 * steps, "score_rows": len(score_rows)}) + "\n"
    )
    if dist:
        torch.distributed.barrier()
    if dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
