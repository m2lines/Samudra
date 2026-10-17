# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-noise observation validation of direct-window diffusion readouts."""

import numpy as np
import torch

from samudra.experiments.monthly_pilot_data import observation_batch
from samudra.experiments.observation_metrics import score


@torch.no_grad()
def evaluate(
    model, data, paths, *, members=8, steps=32, persistence=False, climatology=False
):
    model.eval()
    predictions, references, ohc, reference_ohc, climates = [], [], [], [], []
    for origin, path in enumerate(paths):
        sample = data.load(path)
        control = data.climatology_fields(sample)
        climates.append(control["surface"][0, :6].cpu().numpy())
        if climatology:
            surface = control["surface"][0]
            monthly_ohc = control["ohc"][0]
        else:
            gen = torch.Generator(device=data.device).manual_seed(904001 + origin)
            batch = observation_batch(data, sample, "surface", gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = model.trajectory(
                    batch["fields"],
                    batch["available"],
                    model.adapt(sample["atmosphere"]),
                    sample["contexts"],
                )
            native_support = torch.zeros_like(data.mask, dtype=torch.bool)
            native_support[[38, 76]] = data.mask[[38, 76]].bool()
            monthly_support = torch.zeros_like(native_support)
            monthly_support[data.ts_indices] = data.ts_mask.bool()
            native_frames = []
            for i, state in enumerate(states[1:]):
                latent = states[0] if persistence else state
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    draws = model.decoder.sample(
                        latent.expand(members, -1, -1, -1, -1),
                        native_support,
                        gen,
                        steps=steps,
                        position_days=-2.5 if persistence else 2.5 + i * 5,
                    )
                native_frames.append(draws.mean(0, keepdim=True))
            native = torch.stack(native_frames, 1)
            surface = data.physical(native)[0, :, [38, 76]]
            weights = sample["month_weights"]
            # Persistence holds the initialized latent fixed. The monthly window
            # decoder remains conditioned on the same requested duration.
            latent_month = (
                states[0]
                if persistence
                else model.monthly_condition(states[1:], weights)
            )
            import pandas as pd

            duration = float(pd.Timestamp(sample["name"]).days_in_month)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                monthly = model.decoder.sample(
                    latent_month.expand(members, -1, -1, -1, -1),
                    monthly_support,
                    gen,
                    steps=steps,
                    duration_days=duration,
                    position_days=-2.5 if persistence else duration / 2,
                ).mean(0, keepdim=True)
            # Supplied SST is excluded from IAP interior supervision. The top
            # temperature layer used for OHC comes from the sampled surface mean.
            monthly[:, 38] = (native[:, :, 38] * weights[None, :, None, None]).sum(1)
            monthly_ohc = data.ohc(monthly)[0]
        predictions.append(surface[:6].cpu().numpy())
        references.append(
            np.concatenate(
                [sample["raw"]["surface"][19:25], sample["raw"]["velocity"][19:25]], 1
            )
        )
        ohc.append(monthly_ohc)
        reference_ohc.append(
            np.where(np.isfinite(monthly_ohc), sample["raw"]["ohc"], np.nan)
        )
    result = score(
        np.stack(predictions),
        np.stack(references),
        np.stack(ohc),
        np.stack(reference_ohc),
        data.grid["lat"],
        data.grid["lon"],
        data.grid["mask"][0],
        strict_velocity_support=True,
        global_observations=True,
        compute_spectra=False,
        climatology_surface=np.stack(climates),
        **data.ohc_metric_kwargs,
    )
    result.update(
        origins=[p.stem for p in paths],
        members=members,
        sampler_steps=steps,
        readout="five-day surface mean plus direct-monthly interior; top OHC layer from mean surface",
        persistence="fixed initialized latent" if persistence else None,
    )
    return result
