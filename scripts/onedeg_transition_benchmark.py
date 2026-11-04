# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tracked, paired transition diagnostics for the frozen one-degree controls.

Run from a source checkout, with inputs/checkpoints bound read-only on Torch.
Thresholds use OM4 calibration windows only. The original dataset is never edited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr

from scripts.onedeg_weight_circuit import build_state, context
from scripts.onedeg_weight_edit import (
    PROG,
    Diagnostics,
    log,
    model_from_checkpoint,
    save_json,
    sha256,
)

DEEP = [base + level for base in (38, 57) for level in range(15, 19)]
REGIONS = ("wet", "coast", "interior")


def phase(index, mode):
    if index % 2:
        return "within_pair"
    if mode == "teacher_forced":
        return "reinitialized_first_lead"
    return "initial_first_lead" if index == 0 else "between_calls"


class TransitionDiagnostics:
    def __init__(self, static):
        self.base = Diagnostics(static)
        self.mask = self.base.mask
        self.regions = {
            "wet": self.mask,
            "coast": self.base.coast,
            "interior": self.mask & ~self.base.coast,
        }

    def calibrate(self, windows, quantile):
        thresholds = {}
        for region, mask in self.regions.items():
            values = []
            for c in range(len(PROG)):
                increments = np.concatenate(
                    [(w[2:, c] - w[1:-1, c])[:, mask[c]].ravel() for w in windows]
                )
                if not increments.size:
                    raise ValueError(f"Empty calibration region: {region}/{PROG[c]}")
                values.append(float(np.quantile(np.abs(increments), quantile)))
            thresholds[region] = np.array(values)
        return thresholds

    def measure(self, pred, previous, truth, true_previous, thresholds):
        # Arithmetic in float64 prevents cancellation in small deep tendencies.
        for field in (pred, previous, truth, true_previous):
            if not np.isfinite(field[self.mask]).all():
                raise ValueError("Non-finite wet prediction/reference")
        inc = np.where(self.mask, pred.astype(np.float64) - previous, 0)
        ref = np.where(self.mask, truth.astype(np.float64) - true_previous, 0)
        err = inc - ref
        state_err = np.where(self.mask, pred.astype(np.float64) - truth, 0)
        rows = []
        for region, mask in self.regions.items():
            weights = mask * self.base.area
            avg = lambda a: self.base.average(a, weights)  # noqa: E731
            pmean, rmean = avg(inc), avg(ref)
            prms, rrms = np.sqrt(avg(inc**2)), np.sqrt(avg(ref**2))
            pvar = np.maximum(avg(inc**2) - pmean**2, 0)
            rvar = np.maximum(avg(ref**2) - rmean**2, 0)
            denominator = np.sqrt(pvar * rvar)
            correlation = np.divide(
                avg(inc * ref) - pmean * rmean,
                denominator,
                out=np.full(len(PROG), np.nan),
                where=denominator > 1e-20,
            ).clip(-1, 1)
            rms_ratio = np.divide(
                prms, rrms, out=np.full(len(PROG), np.nan), where=rrms > 1e-15
            )
            threshold = thresholds[region][:, None, None]
            metrics = {
                "state_rmse_normalized": np.sqrt(avg(state_err**2)),
                "increment_rmse_normalized": np.sqrt(avg(err**2)),
                "increment_bias_physical": avg(err) * self.base.std,
                "pred_increment_rms_physical": prms * self.base.std,
                "ref_increment_rms_physical": rrms * self.base.std,
                "increment_rms_ratio": rms_ratio,
                "increment_rms_ratio_distance": np.abs(rms_ratio - 1),
                "increment_spatial_correlation": correlation,
                "pred_tail_area_fraction": avg(np.abs(inc) > threshold),
                "ref_tail_area_fraction": avg(np.abs(ref) > threshold),
                "pred_max_abs_increment_physical": np.max(
                    np.where(mask, np.abs(inc), 0), axis=(-2, -1)
                )
                * self.base.std,
                "ref_max_abs_increment_physical": np.max(
                    np.where(mask, np.abs(ref), 0), axis=(-2, -1)
                )
                * self.base.std,
            }
            # Coast gradients use edges touching coastal cells; all derivatives
            # require two wet endpoints. Interior summaries omit this edge metric.
            if region != "interior":
                metrics["increment_gradient_rmse_normalized_per_km"] = np.sqrt(
                    self.base.gradients(err, coastal=region == "coast")
                )
                metrics["pred_increment_gradient_rms_normalized_per_km"] = np.sqrt(
                    self.base.gradients(inc, coastal=region == "coast")
                )
                metrics["ref_increment_gradient_rms_normalized_per_km"] = np.sqrt(
                    self.base.gradients(ref, coastal=region == "coast")
                )
            for c, name in enumerate(PROG):
                row: dict = {"channel": name, "region": region}
                row.update(
                    {
                        k: float(v[c]) if np.isfinite(v[c]) else None
                        for k, v in metrics.items()
                    }
                )
                rows.append(row)
        return rows


def prepare_data(args, static, manifest):
    source = xr.open_zarr(args.data_root / "OM4.zarr", chunks=None, consolidated=True)
    np.testing.assert_array_equal(source.y.values, static["lat"])
    np.testing.assert_array_equal(source.x.values, static["lon"])
    np.testing.assert_array_equal(source.lev.values, static["lev"])
    for d in range(19):
        np.testing.assert_array_equal(source[f"mask_{d}"].values, static["mask"][d])
    for kind in ("means", "stds"):
        stats = xr.open_zarr(args.data_root / f"OM4_{kind}.zarr", chunks=None)
        values = []
        for name in PROG:
            if name == "zos":
                value = stats[name].item()
            else:
                var, depth = name.rsplit("_", 1)
                value = stats[var].isel(lev=int(depth)).item()
            values.append(value)
        np.testing.assert_array_equal(
            np.array(values, dtype=np.float32),
            static["mean" if kind == "means" else "std"],
        )
    forcing_stats = np.load(args.inputs / "forcing_stats.npz")
    records = []
    data_dir = args.output / "data"
    data_dir.mkdir()
    for entry in manifest["windows"]:
        start = entry["start_index"]
        block = source.isel(time=slice(start, start + 14))
        if [str(t) for t in block.time.values] != entry["times"]:
            raise ValueError("Input manifest does not match source times")
        arrays = [
            block[v].transpose("time", "lev", "y", "x").values
            for v in ("uo", "vo", "thetao", "so")
        ]
        arrays.append(block.zos.values[:, None])
        states = np.concatenate(arrays, axis=1)
        if not np.isfinite(states[:, static["mask"]]).all():
            raise ValueError("Reference contains missing wet values")
        states = np.where(
            static["mask"][None],
            (states - static["mean"][None, :, None, None])
            / static["std"][None, :, None, None],
            0,
        ).astype(np.float32)
        anomaly = np.load(args.inputs / f"{entry['name']}_anomaly.npy")
        forcing = np.stack(
            [block[v].values for v in ("tauuo", "tauvo", "hfds")] + [anomaly], axis=1
        )
        boundary = np.where(
            static["mask"][0][None, None],
            (forcing - forcing_stats["mean"][None, :, None, None])
            / forcing_stats["std"][None, :, None, None],
            0,
        ).astype(np.float32)
        if not np.isfinite(boundary).all():
            raise ValueError("Non-finite normalized forcing")
        path = data_dir / f"{entry['name']}.npz"
        np.savez(path, states=states, boundary=boundary)
        records.append({"window": entry["name"], "sha256": sha256(path)})
        if entry["name"] == "calibration_2015_01":
            import hashlib

            if (
                hashlib.sha256(states.tobytes()).hexdigest()
                != manifest["original_cached_states_sha256"]
            ):
                raise ValueError(
                    "Data pipeline differs from previous cached original-data window"
                )
        log(f"Prepared {entry['name']}")
    source.close()
    save_json(args.output / "prepared_data_checksums.json", records)


def predict(model, states, boundary, grid, device):
    with torch.inference_mode():
        return (
            model.forward_once(
                torch.from_numpy(states.reshape(1, 154, 180, 360)).to(device),
                torch.from_numpy(boundary.reshape(1, 8, 180, 360)).to(device),
                grid,
            )
            .cpu()
            .numpy()
            .reshape(2, 77, 180, 360)
        )


def trajectory(model, states, boundary, grid, device, candidate, mode):
    history = states[:2].copy()
    for offset in range(0, 12, 2):
        if mode == "teacher_forced":
            history = states[offset : offset + 2].copy()
        previous = history[-1].copy()
        pred = predict(model, history, boundary[offset : offset + 2], grid, device)
        if candidate in ("external_blend", "deep_persistence"):
            alpha = 0.5 if candidate == "external_blend" else 1.0
            pred[:, DEEP] = (1 - alpha) * pred[:, DEEP] + alpha * previous[None, DEEP]
        for lead in range(2):
            yield offset + lead, pred[lead], previous
            previous = pred[lead]
        history = pred.copy()


def plot_transitions(path, snapshots, static, channel, title):
    c = PROG.index(channel)
    fig, axes = plt.subplots(
        len(snapshots),
        4,
        figsize=(17, 3 * len(snapshots)),
        squeeze=False,
        layout="constrained",
    )
    lon = np.where(static["lon"] > 180, static["lon"] - 360, static["lon"])
    order = np.argsort(lon)
    mask = static["mask"][c]
    scale = static["std"][c]
    for row, (index, pred, ref) in enumerate(snapshots):
        p, r = pred[c] * scale, ref[c] * scale
        # Reference-only limits make all candidate panels directly comparable.
        limit = max(float(np.quantile(np.abs(r[mask]), 0.99)), 1e-12)
        error_limit = limit
        for col, (field, label, bound) in enumerate(
            (
                (r, "OM4", limit),
                (p, "Prediction", limit),
                (p - r, "Increment error", error_limit),
                (p - r, "North Atlantic error", error_limit),
            )
        ):
            image = axes[row, col].pcolormesh(
                lon[order],
                static["lat"],
                np.where(mask, field, np.nan)[:, order],
                shading="auto",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            axes[row, col].set_title(f"Day {(index + 1) * 5}: {label}")
            if col == 3:
                axes[row, col].set_xlim(-85, -15)
                axes[row, col].set_ylim(0, 65)
            fig.colorbar(image, ax=axes[row, col], shrink=0.7)
    fig.suptitle(
        title
        + " — changes per 5 days; all limits = OM4 absolute increment p99; extremes saturate"
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def window_summary(rows):
    result: dict = {}
    for group, channels in (
        ("all", PROG),
        ("deep", [PROG[c] for c in DEEP]),
        ("untargeted", [v for c, v in enumerate(PROG) if c not in DEEP]),
    ):
        for region in ("wet", "coast"):
            selected = [
                r for r in rows if r["channel"] in channels and r["region"] == region
            ]
            for metric in (
                "state_rmse_normalized",
                "increment_rmse_normalized",
                "increment_gradient_rmse_normalized_per_km",
                "increment_spatial_correlation",
                "increment_rms_ratio",
                "increment_rms_ratio_distance",
                "pred_tail_area_fraction",
                "ref_tail_area_fraction",
            ):
                values = [r[metric] for r in selected if r[metric] is not None]
                result[f"{group}/{region}/{metric}"] = (
                    float(np.mean(values)) if values else None
                )
                if metric == "increment_spatial_correlation":
                    result[f"{group}/{region}/correlation_valid_fraction"] = len(
                        values
                    ) / len(selected)
    result["per_channel_state_rmse"] = [
        float(
            np.mean(
                [
                    r["state_rmse_normalized"]
                    for r in rows
                    if r["region"] == "wet" and r["channel"] == c
                ]
            )
        )
        for c in PROG
    ]
    return result


def acceptance_report(summaries, protocol):
    """Apply predeclared gates and bootstrap paired windows, never grid cells."""
    report = []
    gates = protocol["screening_gates"]
    metrics = {
        "deep_coastal_increment_gradient_rmse_ratio": "deep/coast/increment_gradient_rmse_normalized_per_km",
        "deep_increment_rmse_ratio": "deep/wet/increment_rmse_normalized",
        "all_channel_state_rmse_ratio": "all/wet/state_rmse_normalized",
        "untargeted_channel_mean_state_rmse_ratio": "untargeted/wet/state_rmse_normalized",
    }
    for split in ("selection", "new_local_test"):
        for mode in protocol["modes"]:
            base = {
                r["window"]: r["metrics"]
                for r in summaries
                if r["candidate"] == "original"
                and r["split"] == split
                and r["mode"] == mode
            }
            for candidate in protocol["candidates"][1:]:
                edited = {
                    r["window"]: r["metrics"]
                    for r in summaries
                    if r["candidate"] == candidate
                    and r["split"] == split
                    and r["mode"] == mode
                }
                names = sorted(base)
                if sorted(edited) != names or len(names) != 4:
                    raise ValueError("Incomplete paired seasonal evaluation")
                rng = np.random.default_rng(15)
                bootstrap = rng.integers(0, len(names), size=(10000, len(names)))
                checks = {}
                for label, key in metrics.items():
                    b = np.array([base[n][key] for n in names])
                    e = np.array([edited[n][key] for n in names])
                    ratio = float(e.mean() / b.mean())
                    ci = np.quantile(
                        e[bootstrap].mean(axis=1) / b[bootstrap].mean(axis=1),
                        [0.025, 0.975],
                    ).tolist()
                    checks[label] = {
                        "value": ratio,
                        "paired_window_bootstrap_95": ci,
                        "limit": gates[label + "_max"],
                        "pass": ratio <= gates[label + "_max"],
                    }
                worst = float(
                    max(
                        np.max(
                            np.array(edited[n]["per_channel_state_rmse"])
                            / np.maximum(base[n]["per_channel_state_rmse"], 1e-15)
                        )
                        for n in names
                    )
                )
                checks["worst_window_channel_state_rmse_ratio"] = {
                    "value": worst,
                    "limit": gates["worst_window_channel_state_rmse_ratio_max"],
                    "pass": worst <= gates["worst_window_channel_state_rmse_ratio_max"],
                }
                corr = "deep/wet/increment_spatial_correlation"
                valid = "deep/wet/correlation_valid_fraction"
                available = all(
                    edited[n][corr] is not None
                    and base[n][corr] is not None
                    and edited[n][valid] >= base[n][valid]
                    for n in names
                )
                drop = (
                    float(np.mean([base[n][corr] - edited[n][corr] for n in names]))
                    if available
                    else None
                )
                checks["increment_spatial_correlation_drop"] = {
                    "value": drop,
                    "limit": gates["increment_spatial_correlation_drop_max"],
                    "pass": available
                    and drop <= gates["increment_spatial_correlation_drop_max"],
                }
                rms = "deep/wet/increment_rms_ratio_distance"
                distance_change = float(
                    np.mean([edited[n][rms] - base[n][rms] for n in names])
                )
                checks["increment_rms_ratio_distance_change"] = {
                    "value": distance_change,
                    "limit": 0,
                    "pass": distance_change <= 0,
                }
                report.append(
                    {
                        "candidate": candidate,
                        "split": split,
                        "mode": mode,
                        "windows": names,
                        "checks": checks,
                        "passes_all": all(c["pass"] for c in checks.values()),
                    }
                )
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--edited-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.inputs / "transition_protocol.json").read_text())
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    static = dict(np.load(args.inputs / "static.npz"))
    config = {
        "protocol": protocol,
        "manifest": manifest,
        "original_checkpoint_sha256": sha256(args.checkpoint),
        "edited_checkpoint_sha256": sha256(args.edited_checkpoint),
        "input_hashes": {
            p.name: sha256(p) for p in args.inputs.iterdir() if p.is_file()
        },
        "code_commit": os.environ.get("WANDB_GIT_COMMIT"),
        "code_layer_sha256": os.environ.get("SAMUDRA_CODE_LAYER_SHA256"),
        "container": os.environ.get("SIF_PATH"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "precision": "float32; TF32 disabled",
        "data_root": str(args.data_root),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save_json(args.output / "provenance.json", config)
    with wandb.init(
        entity="ocean_emulators",
        project="default",
        group="onedeg-transition-benchmark-v1",
        name=args.output.name,
        mode="online",
        dir=str(args.output),
        config=config,
        tags=["experiment", "single-step", "weight-edit", "controls"],
    ) as run:
        if run.offline:
            raise RuntimeError("Online W&B required")
        (args.output / "wandb-url.txt").write_text(run.url + "\n")
        log(f"W&B: {run.url}")
        prepare_data(args, static, manifest)
        diag = TransitionDiagnostics(static)
        calibration = [
            np.load(args.output / "data" / f"{e['name']}.npz")["states"]
            for e in manifest["windows"]
            if e["split"] == "calibration"
        ]
        thresholds = diag.calibrate(calibration, protocol["tail_quantile"])
        del calibration
        save_json(
            args.output / "tail_thresholds_normalized.json",
            {k: v.tolist() for k, v in thresholds.items()},
        )
        grid = context(static, args.device)
        build_args = SimpleNamespace(
            checkpoint=args.checkpoint,
            device=args.device,
            output=args.inputs,
            min_level=15,
            basis="balanced",
            alpha=0.0,
        )
        reserved, _, _ = build_state(build_args)
        torch.save({"model": reserved}, args.output / "reservation_only.pt")
        run.config.update(
            {
                "reservation_checkpoint_sha256": sha256(
                    args.output / "reservation_only.pt"
                )
            }
        )
        del reserved
        model = model_from_checkpoint(args)
        summaries = []
        figure_dir = args.output / "figures"
        figure_dir.mkdir()
        metrics_path = args.output / "transitions.csv"
        with metrics_path.open("w") as stream:
            writer = None
            for candidate in protocol["candidates"]:
                checkpoint = (
                    args.edited_checkpoint
                    if candidate == "carry"
                    else args.output / "reservation_only.pt"
                    if candidate == "reservation_only"
                    else args.checkpoint
                )
                state = torch.load(
                    checkpoint, map_location="cpu", mmap=True, weights_only=True
                )["model"]
                model.load_state_dict(
                    {k.removeprefix("module."): v for k, v in state.items()},
                    strict=True,
                )
                del state
                for entry in manifest["windows"]:
                    data = np.load(args.output / "data" / f"{entry['name']}.npz")
                    states, boundary = data["states"], data["boundary"]
                    for mode in protocol["modes"]:
                        rows, snapshots = [], []
                        for index, pred, previous in trajectory(
                            model, states, boundary, grid, args.device, candidate, mode
                        ):
                            if (
                                candidate == "original"
                                and entry["name"] == "calibration_2015_01"
                                and mode == "teacher_forced"
                                and index == 0
                            ):
                                reference = np.load(
                                    args.inputs / "original_reference_prediction.npy",
                                    mmap_mode="r",
                                )[0]
                                delta = np.abs(pred - reference)
                                run.summary["parity/max_abs_original_vs_cached"] = (
                                    float(delta.max())
                                )
                                if delta.max() > 1e-3:
                                    raise ValueError(
                                        "GPU baseline fails cross-device original prediction parity"
                                    )
                            transition_rows = diag.measure(
                                pred,
                                previous,
                                states[index + 2],
                                states[index + 1],
                                thresholds,
                            )
                            for row in transition_rows:
                                row.update(
                                    candidate=candidate,
                                    window=entry["name"],
                                    split=entry["split"],
                                    mode=mode,
                                    transition=index + 1,
                                    day=(index + 1) * 5,
                                    phase=phase(index, mode),
                                    time=entry["times"][index + 2],
                                )
                            rows.extend(transition_rows)
                            if (
                                mode == "autoregressive"
                                and entry["name"]
                                in ("calibration_2015_01", "new_local_test_2021_01")
                                and index in (0, 1, 2, 11)
                            ):
                                snapshots.append(
                                    (
                                        index,
                                        pred - previous,
                                        states[index + 2] - states[index + 1],
                                    )
                                )
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                            writer.writeheader()
                        writer.writerows(rows)
                        stream.flush()
                        summary = window_summary(rows)
                        summaries.append(
                            dict(
                                candidate=candidate,
                                window=entry["name"],
                                split=entry["split"],
                                mode=mode,
                                metrics=summary,
                            )
                        )
                        log_values = {
                            f"{candidate}/{mode}/{k}": v
                            for k, v in summary.items()
                            if v is not None and k != "per_channel_state_rmse"
                        }
                        log_values["window"] = entry["name"]
                        for channel in (
                            ("thetao_0", "thetao_15", "thetao_18", "so_18")
                            if snapshots
                            else ()
                        ):
                            path = (
                                figure_dir
                                / f"{candidate}_{entry['name']}_{channel}.png"
                            )
                            plot_transitions(
                                path,
                                snapshots,
                                static,
                                channel,
                                f"{candidate} / {entry['name']} / {channel}",
                            )
                            log_values[f"maps/{candidate}/{channel}"] = wandb.Image(
                                str(path)
                            )
                        run.log(log_values)
                        save_json(args.output / "window_summaries.json", summaries)
                        log(
                            f"{candidate} {entry['name']} {mode}: 12 transitions recorded"
                        )
        run.summary["completed_windows_per_candidate"] = len(manifest["windows"])
        acceptance = acceptance_report(summaries, protocol)
        save_json(args.output / "acceptance.json", acceptance)
        for result in acceptance:
            prefix = (
                f"acceptance/{result['candidate']}/{result['split']}/{result['mode']}"
            )
            run.summary[prefix + "/passes_all"] = result["passes_all"]
            for name, check in result["checks"].items():
                if check["value"] is not None:
                    run.summary[prefix + "/" + name] = check["value"]
        run.summary["transition_rows"] = (
            len(protocol["candidates"])
            * len(manifest["windows"])
            * len(protocol["modes"])
            * 12
            * 77
            * 3
        )
        artifact = wandb.Artifact(args.output.name, type="transition-benchmark")
        for name in (
            "provenance.json",
            "prepared_data_checksums.json",
            "tail_thresholds_normalized.json",
            "window_summaries.json",
            "transitions.csv",
            "acceptance.json",
        ):
            artifact.add_file(str(args.output / name))
        artifact.add_dir(str(figure_dir), name="figures")
        run.log_artifact(artifact).wait()
        log("Transition benchmark complete")


if __name__ == "__main__":
    main()
