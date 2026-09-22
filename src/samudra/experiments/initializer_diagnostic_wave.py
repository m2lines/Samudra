# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Controlled precision, memorization and single-field initializer diagnostics."""

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.surface_adaptation import thermohaline_mse
from samudra.experiments.surface_state import balanced_loss, channel_mse
from samudra.experiments.surface_wave import atomic_save, batches
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


def selected_loss(prediction, truth, weights, names, objective, channel):
    if objective == "full":
        return balanced_loss(prediction, truth, weights, names, True)
    index = names.index(channel)
    return channel_mse(
        prediction[:, :, index : index + 1],
        truth[:, :, index : index + 1],
        weights[index : index + 1],
    ).mean()


def step_indices(population, seed, step, count):
    """Matched examples across objectives, deterministic across resume."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    return rng.choice(population, size=count, replace=len(population) < count).tolist()


class DiagnosticWave(InitializerWave):
    def __init__(self, args):
        super().__init__(args)
        if self.world != 1:
            raise ValueError("These controlled diagnostic runs require one GPU")
        self.channel = self.names.index(args.channel)
        self.climate = torch.load(
            Path(args.wave1_root) / "initializer/climatology.pt",
            map_location=self.device,
            weights_only=False,
        )["monthly"].float()
        self.train_probe_ids = np.linspace(
            0, len(self.trainset) - 1, 16, dtype=int
        ).tolist()
        self.population = (
            np.linspace(0, len(self.trainset) - 1, args.subset, dtype=int).tolist()
            if args.subset
            else list(range(len(self.trainset)))
        )
        with Path(args.initial_checkpoint).open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        self.signature = {
            "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
            "checkpoint_sha256": checksum,
            "objective": args.objective,
            "channel": args.channel,
            "precision": args.precision,
            "population": self.population,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "accumulate": args.accumulate,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "data": self.config.model_dump(mode="json"),
        }
        (self.out / "diagnostic-protocol.json").write_text(
            json.dumps(self.signature, indent=2) + "\n"
        )

    def predict(self, sample, precision):
        surface, past, context, _, _, _ = sample
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            return self.initializer(surface, past, context, self.mask)

    def month_climate(self, dataset, ids):
        months = [dataset.sources[0].time.values[i + 18].month - 1 for i in ids]
        return self.climate[months, self.channel]

    def measure(self, dataset, indices, precision, split, dump=False):
        self.prepare(dataset)
        self.model.eval()
        rows, snapshots = [], {}
        with torch.no_grad():
            for ids in batches(indices, self.args.batch_size):
                sample = self.sample(dataset, ids)
                truth = sample[3]
                pred = self.predict(sample, precision)
                mse = channel_mse(pred, truth, self.weights)
                full = balanced_loss(pred, truth, self.weights, self.names, True)
                ts = thermohaline_mse(mse, self.names)
                climate = self.month_climate(dataset, ids)
                for j, idx in enumerate(ids):
                    t, p = truth[j, -1, self.channel], pred[j, -1, self.channel]
                    c = climate[j]
                    w = self.weights[self.channel]

                    def mean(value):
                        return float((value.double() * w).sum() / w.sum())

                    a, b = t - c, p - c
                    rows.append(
                        {
                            "split": split,
                            "precision": precision,
                            "index": idx,
                            "date": str(dataset.sources[0].time.values[idx + 18]),
                            "field_mse": float(mse[j, :, self.channel].mean()),
                            "current_field_mse": mean((p - t).square()),
                            "ts_mse": float(ts[j].mean()),
                            "batch_full_mse": float(full),
                            "climatology_mse": mean(a.square()),
                            "bias": mean(p - t),
                            "pred_anomaly_m2": mean(b.square()),
                            "true_anomaly_m2": mean(a.square()),
                            "cross_anomaly": mean(a * b),
                            "pred_anomaly_mean": mean(b),
                            "true_anomaly_mean": mean(a),
                            "bf16_rounding_mse": mean(
                                (t.bfloat16().float() - t).square()
                            ),
                        }
                    )
                    if dump:
                        snapshots[f"{idx}_prediction"] = p.cpu().numpy()
                        snapshots[f"{idx}_truth"] = t.cpu().numpy()
                        snapshots[f"{idx}_climatology"] = c.cpu().numpy()
                        snapshots[f"{idx}_date"] = np.array(rows[-1]["date"])
        if dump:
            path = self.out / f"{split}-{precision}.csv"
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(
                self.out / f"{split}-{precision}-maps.npz",
                **snapshots,
                latitude=self.lat.cpu().numpy(),
                longitude=self.source.resolution[1].cpu().numpy(),
                mask=self.mask[self.channel].cpu().numpy(),
                mean=self.mean[self.channel].cpu().numpy(),
                std=self.std[self.channel].cpu().numpy(),
            )
        return {
            k: float(np.mean([r[k] for r in rows]))
            for k in [
                "field_mse",
                "current_field_mse",
                "ts_mse",
                "batch_full_mse",
                "climatology_mse",
            ]
        }

    def precision_check(self):
        """Same FP32 weights and batches; only autocast changes."""
        for split, dataset, ids in [
            ("train", self.trainset, self.train_probe_ids),
            ("validation", self.valset, self.val_ids),
        ]:
            scores = {
                p: self.measure(dataset, ids, p, split, True) for p in ["bf16", "fp32"]
            }
            self.emit({"event": "precision", "split": split, "scores": scores})

    def fit_diagnostic(self):
        args = self.args
        self.prepare(self.trainset)
        self.prepare(self.valset)
        optimizer = torch.optim.AdamW(
            self.initializer.parameters(), lr=args.learning_rate, weight_decay=0.01
        )
        last = self.out / "last.pt"
        state = dict(step=0, best=float("inf"), elapsed=0.0)
        if last.exists():
            saved = torch.load(last, map_location=self.device, weights_only=False)
            if saved["signature"] != self.signature:
                raise ValueError("Diagnostic resume protocol differs")
            self.model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
        selection_dataset = self.trainset if args.subset else self.valset
        selection_ids = self.population if args.subset else self.val_ids
        selection_key = "field_mse"
        started = time.monotonic()
        prior_elapsed = state["elapsed"]
        last_save = started
        if not np.isfinite(state["best"]):
            initial = self.measure(
                selection_dataset, selection_ids, args.precision, "selection"
            )
            state["best"] = initial[selection_key]
            atomic_save(
                {"model": self.model.state_dict(), "state": state.copy()},
                self.out / "best.pt",
            )
            self.emit(
                {
                    "event": "baseline",
                    "selection": initial,
                    "selection_split": "fitting_set" if args.subset else "validation",
                }
            )
        while state["step"] < args.max_steps:
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            ids = step_indices(
                self.population,
                args.seed,
                state["step"],
                args.batch_size * args.accumulate,
            )
            losses = []
            for batch in batches(ids, args.batch_size):
                sample = self.sample(self.trainset, batch)
                pred = self.predict(sample, args.precision)
                loss = selected_loss(
                    pred,
                    sample[3],
                    self.weights,
                    self.names,
                    args.objective,
                    args.channel,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diagnostic loss")
                (loss / args.accumulate).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(self.initializer.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            state["step"] += 1
            state["elapsed"] = prior_elapsed + time.monotonic() - started
            if state["step"] == 1 or state["step"] % 20 == 0:
                self.emit(
                    {
                        "event": "train_step",
                        **state,
                        "loss": float(np.mean(losses)),
                        "gradient_norm": float(norm),
                    }
                )
            done = state["step"] == args.max_steps
            validate = done or state["step"] % args.eval_every == 0
            if validate:
                selection = self.measure(
                    selection_dataset, selection_ids, args.precision, "selection"
                )
                probe = self.measure(
                    self.trainset, self.train_probe_ids, args.precision, "train_probe"
                )
                if selection[selection_key] < state["best"]:
                    state["best"] = selection[selection_key]
                    atomic_save(
                        {"model": self.model.state_dict(), "state": state.copy()},
                        self.out / "best.pt",
                    )
                self.emit(
                    {
                        "event": "validation",
                        **state,
                        "selection": selection,
                        "train_probe": probe,
                    }
                )
            if validate or time.monotonic() - last_save >= args.checkpoint_seconds:
                atomic_save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "signature": self.signature,
                        "state": state.copy(),
                    },
                    last,
                )
                last_save = time.monotonic()
        self.model.load_state_dict(
            torch.load(
                self.out / "best.pt", map_location=self.device, weights_only=False
            )["model"]
        )
        self.measure(
            self.trainset,
            self.population if args.subset else self.train_probe_ids,
            args.precision,
            "train-selected",
            True,
        )
        self.measure(
            self.valset, self.val_ids, args.precision, "validation-selected", True
        )
        (self.out / "TRAIN_COMPLETE.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )

    def heldout(self):
        self.frame_caches.clear()
        self.verified_sources.clear()
        torch.cuda.empty_cache()
        source = native_om4_source(
            self.context_bundle.inference_source,
            LocalLocation(path=Path(self.args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(self.args.readers),
        )
        dataset = self.dataset(source)
        self.measure(
            dataset,
            list(range(0, len(dataset), 6)),
            self.args.precision,
            "heldout",
            True,
        )

    def run_diagnostic(self):
        try:
            if self.args.task == "precision":
                self.precision_check()
            else:
                self.fit_diagnostic()
            if self.args.heldout:
                self.heldout()
            (self.out / "COMPLETE.json").write_text(
                json.dumps(
                    {"task": self.args.task, "signature": self.signature}, indent=2
                )
                + "\n"
            )
        except BaseException:
            (self.out / "exception.log").write_text(traceback.format_exc())
            raise
        finally:
            if self.run:
                self.run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["precision", "fit"], required=True)
    p.add_argument("--objective", choices=["full", "field"], default="field")
    p.add_argument("--channel", default="so_9")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--subset", type=int, choices=[0, 1, 16], default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--initial-checkpoint", required=True)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    p.add_argument("--data-root", default="/scratch/jr7309/data/om4_onedeg_v3")
    p.add_argument(
        "--wave1-root", default="/scratch/jr7309/runs/2026-09-18-surface-wave1"
    )
    p.set_defaults(
        arm="D",
        phase="reconstruction",
        readers=4,
        device_cache=True,
        device_cache_reserve_gib=30,
        val_origins=12,
        checkpoint_seconds=180,
    )
    args = p.parse_args()
    if min(args.batch_size, args.accumulate, args.max_steps, args.eval_every) < 1:
        p.error("Positive batch sizes, steps and evaluation interval required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    DiagnosticWave(args).run_diagnostic()


if __name__ == "__main__":
    main()
