# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Single-field conditional residual diffusion with frozen D and matched control."""

import argparse
import copy
import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.initializer_diagnostic_wave import DiagnosticWave, step_indices
from samudra.experiments.initializer_diffusion import (
    ResidualDenoiser,
    area_mean,
    edm_loss,
    ensemble_crps,
    sample_edm,
)
from samudra.experiments.surface_wave import atomic_save, batches
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


class DiffusionWave(DiagnosticWave):
    def __init__(self, args):
        super().__init__(args)
        self.model.eval().requires_grad_(False)
        self.base_cache = {}
        self.field_mask = self.mask[self.channel][None, None].float()
        self.field_weights = self.weights[self.channel][None, None]
        # 19*2 surfaces, 19*3 forcings, 5 geography/season, D, climate,
        # the two observed-surface masks and target wet mask.
        self.denoiser = ResidualDenoiser(105, args.width).to(self.device)
        self.ema = copy.deepcopy(self.denoiser).eval().requires_grad_(False)
        self.residual_scale = 1.0
        self.protocol = {
            "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
            "parent": self.signature,
            "mode": args.mode,
            "width": args.width,
            "parameters": sum(p.numel() for p in self.denoiser.parameters()),
            "max_steps": args.max_steps,
            "train_hours": args.train_hours,
            "selection": "validation empirical CRPS; deterministic control uses MAE",
            "target": "current-time so_9 only; frozen D residual divided by training residual RMS",
            "sigma": "lognormal(-1.2,1.2); sigma_data=1; EDM loss and preconditioning",
            "sampler": "Heun sigma 80 to .002, rho=7, 2*N-1 evaluations",
            "validation_members": args.validation_members,
            "validation_steps": args.sampling_steps,
            "ema": 0.999,
        }
        (self.out / "diffusion-protocol.json").write_text(
            json.dumps(self.protocol, indent=2) + "\n"
        )
        # Replace the inherited generic selection label with the actual protocol.
        p = self.out / "study-manifest.json"
        m = json.loads(p.read_text())
        m["training_selection"] = self.protocol["selection"]
        p.write_text(json.dumps(m, indent=2) + "\n")
        self.emit({"event": "diffusion_ready", **self.protocol})

    @torch.no_grad()
    def cache_baseline(self, dataset, ids):
        self.prepare(dataset)
        key = id(dataset)
        if key not in self.base_cache:
            self.base_cache[key] = {}
        cache = self.base_cache[key]
        self.model.eval()
        for batch in batches([i for i in ids if i not in cache], self.args.batch_size):
            sample = self.sample(dataset, batch)
            prediction = self.predict(sample, "bf16")[:, -1, self.channel].float()
            for j, idx in enumerate(batch):
                cache[idx] = prediction[j].clone()
        self.emit({"event": "baseline_cached", "origins": len(cache)})

    def diffusion_inputs(self, dataset, ids):
        surface, past, context, truth, _, _ = self.sample(dataset, ids)
        baseline = torch.stack([self.base_cache[id(dataset)][i] for i in ids])[:, None]
        climate = self.month_climate(dataset, ids)[:, None]
        mask = torch.cat(
            (self.mask[self.initializer.surface].float(), self.field_mask[0]), 0
        )[None].expand(len(ids), -1, -1, -1)
        condition = torch.cat(
            (
                surface.flatten(1, 2),
                past.flatten(1, 2),
                context,
                baseline,
                climate,
                mask,
            ),
            1,
        )
        target = truth[:, -1, self.channel : self.channel + 1]
        return (
            condition,
            (target - baseline) * self.field_mask / self.residual_scale,
            baseline,
            target,
            climate,
        )

    @torch.no_grad()
    def ensemble(self, condition, members, steps, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.args.mode == "deterministic":
                zero = torch.zeros((1, 1, *condition.shape[-2:]), device=self.device)
                return self.ema(
                    zero, torch.ones(1, device=self.device), condition, self.field_mask
                )
            # Member microbatches keep sampling within the same memory envelope.
            results = []
            for ids in batches(list(range(members)), self.args.batch_size):
                results.append(
                    sample_edm(
                        self.ema,
                        condition.expand(len(ids), -1, -1, -1),
                        self.field_mask,
                        generator,
                        steps,
                    )
                )
            return torch.cat(results)

    @torch.no_grad()
    def evaluate(self, dataset, ids, label, members, steps, dump=False):
        self.cache_baseline(dataset, ids)
        self.ema.eval()
        rows, fields = [], {}
        physical = float(self.std[self.channel])
        for idx in ids:
            condition, _, baseline, truth, climate = self.diffusion_inputs(
                dataset, [idx]
            )
            samples = baseline + self.residual_scale * self.ensemble(
                condition, members, steps, 90210 + idx
            )
            mean = samples.mean(0, keepdim=True)
            variance = samples.var(0, unbiased=False, keepdim=True)
            t = truth[0, 0]

            def avg(x):
                return float(area_mean(x, self.field_weights).mean())

            a, b = truth - climate, mean - climate
            ac, bc = a - avg(a), b - avg(b)
            row = {
                "index": idx,
                "date": str(dataset.sources[0].time.values[idx + 18]),
                "members": samples.shape[0],
                "mean_mse": avg((mean - truth).square()) * physical**2,
                "sample_mse": avg((samples - truth).square()) * physical**2,
                "d_mse": avg((baseline - truth).square()) * physical**2,
                "climatology_mse": avg((climate - truth).square()) * physical**2,
                "crps": avg(ensemble_crps(samples, truth[0])) * physical,
                "d_mae": avg((baseline - truth).abs()) * physical,
                "spread_variance": avg(variance) * physical**2,
                "coverage90": avg(
                    (
                        (truth >= samples.quantile(0.05, dim=0))
                        & (truth <= samples.quantile(0.95, dim=0))
                    ).float()
                ),
                "anomaly_correlation": avg(ac * bc)
                / (avg(ac.square()) * avg(bc.square())) ** 0.5,
                "mean_anomaly_m2": avg(b.square()) * physical**2,
                "sample_anomaly_m2": avg((samples - climate).square()) * physical**2,
                "truth_anomaly_m2": avg(a.square()) * physical**2,
            }
            if not all(np.isfinite(v) for k, v in row.items() if k != "date"):
                raise FloatingPointError("Nonfinite ensemble metrics")
            rows.append(row)
            if dump:
                for name, value in [
                    ("truth", t),
                    ("prediction", mean[0, 0]),
                    ("climatology", climate[0, 0]),
                    ("baseline", baseline[0, 0]),
                    ("spread", variance[0, 0].sqrt()),
                ]:
                    fields[f"{idx}_{name}"] = value.cpu().numpy()
                fields[f"{idx}_date"] = np.array(row["date"])
                # Retain every member for analysis, one compressed file per origin.
                np.savez_compressed(
                    self.out / f"{label}-members-{idx}.npz",
                    samples=samples[:, 0].cpu().numpy(),
                    truth=t.cpu().numpy(),
                    climatology=climate[0, 0].cpu().numpy(),
                    baseline=baseline[0, 0].cpu().numpy(),
                    date=np.array(row["date"]),
                    mask=self.field_mask[0, 0].cpu().numpy(),
                    latitude=self.lat.cpu().numpy(),
                    std=physical,
                )
        summary = {
            k: float(np.mean([r[k] for r in rows]))
            for k in rows[0]
            if k not in ["index", "date", "members"]
        }
        if dump:
            with (self.out / f"{label}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(
                self.out / f"{label}-maps.npz",
                **fields,
                latitude=self.lat.cpu().numpy(),
                longitude=self.source.resolution[1].cpu().numpy(),
                mask=self.mask[self.channel].cpu().numpy(),
                mean=self.mean[self.channel].cpu().numpy(),
                std=physical,
            )
        self.emit(
            {
                "event": "ensemble_evaluation",
                "split": label,
                "members": members,
                "steps": steps,
                **summary,
            }
        )
        return summary

    def fit(self):
        args = self.args
        self.cache_baseline(self.trainset, self.population)
        self.cache_baseline(self.valset, self.val_ids)
        # An exact weighted training-only RMS, never estimated on validation.
        total = 0.0
        with torch.no_grad():
            for ids in batches(self.population, args.batch_size):
                _, residual, _, _, _ = self.diffusion_inputs(self.trainset, ids)
                total += float(area_mean(residual.square(), self.field_weights).sum())
        self.residual_scale = (total / len(self.population)) ** 0.5
        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(), lr=args.learning_rate, weight_decay=0.01
        )
        state = {"step": 0, "best": float("inf"), "elapsed": 0.0}
        last = self.out / "last.pt"
        if last.exists():
            saved = torch.load(last, map_location=self.device, weights_only=False)
            if saved["protocol"] != self.protocol:
                raise ValueError("Resume protocol differs")
            self.denoiser.load_state_dict(saved["denoiser"])
            self.ema.load_state_dict(saved["ema"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
            self.residual_scale = saved["residual_scale"]
        self.emit(
            {
                "event": "residual_scale",
                "normalized_rms": self.residual_scale,
                "physical_rms": self.residual_scale * float(self.std[self.channel]),
            }
        )
        start = time.monotonic()
        prior = state["elapsed"]
        last_save = start

        def save(path):
            atomic_save(
                {
                    "denoiser": self.denoiser.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "state": state.copy(),
                    "protocol": self.protocol,
                    "residual_scale": self.residual_scale,
                },
                path,
            )

        while (
            state["step"] < args.max_steps
            and state["elapsed"] < args.train_hours * 3600
        ):
            self.denoiser.train()
            optimizer.zero_grad(set_to_none=True)
            ids = step_indices(
                self.population,
                args.seed,
                state["step"],
                args.batch_size * args.accumulate,
            )
            generator = torch.Generator(device=self.device).manual_seed(
                args.seed + state["step"]
            )
            losses = []
            for batch in batches(ids, args.batch_size):
                condition, residual, _, _, _ = self.diffusion_inputs(
                    self.trainset, batch
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.mode == "diffusion":
                        loss = edm_loss(
                            self.denoiser,
                            residual,
                            condition,
                            self.field_mask,
                            self.field_weights,
                            generator,
                        )
                    else:
                        prediction = self.denoiser(
                            torch.zeros_like(residual),
                            torch.ones(len(batch), device=self.device),
                            condition,
                            self.field_mask,
                        )
                        loss = area_mean(
                            (prediction - residual).square(), self.field_weights
                        ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / args.accumulate).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            with torch.no_grad():
                for ema, param in zip(
                    self.ema.parameters(), self.denoiser.parameters(), strict=True
                ):
                    ema.lerp_(param, 0.001)
            state["step"] += 1
            state["elapsed"] = prior + time.monotonic() - start
            if state["step"] == 1 or state["step"] % 50 == 0:
                self.emit(
                    {
                        "event": "train_step",
                        **state,
                        "loss": float(np.mean(losses)),
                        "gradient_norm": float(norm),
                    }
                )
            done = (
                state["step"] == args.max_steps
                or state["elapsed"] >= args.train_hours * 3600
            )
            validate = done or state["step"] % args.eval_every == 0
            if validate:
                scores = self.evaluate(
                    self.valset,
                    self.val_ids,
                    "validation",
                    args.validation_members,
                    args.sampling_steps,
                )
                if scores["crps"] < state["best"]:
                    state["best"] = scores["crps"]
                    save(self.out / "best.pt")
                self.emit({"event": "validation_selection", **state, "scores": scores})
            if validate or time.monotonic() - last_save >= args.checkpoint_seconds:
                save(last)
                last_save = time.monotonic()
        # In case the time cap fell between validations.
        if not (self.out / "best.pt").exists():
            state["best"] = self.evaluate(
                self.valset,
                self.val_ids,
                "validation",
                args.validation_members,
                args.sampling_steps,
            )["crps"]
            save(self.out / "best.pt")
        (self.out / "TRAIN_COMPLETE.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )

    def load_best(self):
        saved = torch.load(
            self.out / "best.pt", map_location=self.device, weights_only=False
        )
        self.ema.load_state_dict(saved["ema"])
        self.residual_scale = saved["residual_scale"]
        self.emit(
            {
                "event": "selected_checkpoint",
                "state": saved["state"],
                "residual_scale": self.residual_scale,
            }
        )

    def evaluate_selected(self):
        self.load_best()
        self.evaluate(
            self.valset,
            self.val_ids,
            "validation-selected",
            self.args.members,
            self.args.final_steps,
            True,
        )
        self.frame_caches.clear()
        self.verified_sources.clear()
        self.base_cache.clear()
        torch.cuda.empty_cache()
        source = native_om4_source(
            self.context_bundle.inference_source,
            LocalLocation(path=Path(self.args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(self.args.readers),
        )
        dataset = self.dataset(source)
        ids = list(range(0, len(dataset), 6))
        if self.args.smoke:
            ids = ids[:1]
        self.evaluate(
            dataset, ids, "heldout", self.args.members, self.args.final_steps, True
        )
        (self.out / "COMPLETE.json").write_text(
            json.dumps(
                {"protocol": self.protocol, "heldout_origins": len(ids)}, indent=2
            )
            + "\n"
        )

    def execute_diffusion(self):
        try:
            if self.args.task == "fit":
                self.fit()
            if self.args.task == "evaluate" or self.args.smoke:
                self.evaluate_selected()
        except BaseException:
            (self.out / "exception.log").write_text(traceback.format_exc())
            raise
        finally:
            if self.run:
                self.run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["diffusion", "deterministic"], default="diffusion"
    )
    p.add_argument("--task", choices=["fit", "evaluate"], default="fit")
    p.add_argument("--output", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--initial-checkpoint", required=True)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--train-hours", type=float, default=4)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--validation-members", type=int, default=8)
    p.add_argument("--sampling-steps", type=int, default=16)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--final-steps", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
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
        device_cache_reserve_gib=16,
        val_origins=12,
        checkpoint_seconds=180,
        objective="field",
        channel="so_9",
        precision="bf16",
        subset=0,
    )
    args = p.parse_args()
    if args.smoke:
        args.subset = 16
        args.val_origins = 2
        args.max_steps = 3
        args.eval_every = 3
        args.validation_members = 2
        args.sampling_steps = 4
        args.members = 2
        args.final_steps = 4
    if (
        min(
            args.max_steps,
            args.eval_every,
            args.batch_size,
            args.accumulate,
            args.members,
            args.validation_members,
        )
        < 1
        or min(args.sampling_steps, args.final_steps) < 2
    ):
        p.error("Positive counts and at least two sampling steps required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    DiffusionWave(args).execute_diffusion()


if __name__ == "__main__":
    main()
