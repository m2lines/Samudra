# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Monthly observation evaluation using the global integrated/spectral protocol."""

import json
from pathlib import Path

import numpy as np
import torch

from samudra.observations.metrics import score


class Evaluator:
    def __init__(self, model, data):
        self.model, self.data = model, data

    def arguments(
        self, sample
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            sample["surface"],
            sample["atmosphere"],
            sample["contexts"],
            self.data.mask,
            sample["validity"],
        )

    @torch.no_grad()
    def evaluate(
        self, paths, persistence=False, export=None, climatology=False, anomaly=False
    ):
        self.model.eval()
        predictions, references, ohc, reference_ohc = [], [], [], []
        interior_error, interior_bias, interior_count = (
            np.zeros(28),
            np.zeros(28),
            np.zeros(28),
        )
        for path in paths:
            sample = self.data.load(path)
            if climatology:
                prediction = self.data.climatology_prediction(sample)
            elif persistence or anomaly:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    initial = self.model.initialize(
                        sample["surface"][:, :19],
                        sample["atmosphere"][:, :19],
                        sample["contexts"][:, 18],
                        self.data.mask,
                        sample["validity"][:, :19],
                    )
                state = (
                    self.data.persistence_anomaly(initial, sample)
                    if anomaly
                    else initial[:, -1]
                )
                prediction = state[:, None].expand(
                    -1, len(sample["month_weights"]), -1, -1, -1
                )
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, _ = self.model.forecast(*self.arguments(sample))
            monthly = (
                prediction * sample["month_weights"][None, :, None, None, None]
            ).sum(1)
            surface = self.data.physical(prediction)[:, :6, [38, 76]].cpu().numpy()[0]
            truth = np.concatenate(
                [sample["raw"]["surface"][19:25], sample["raw"]["velocity"][19:25]],
                axis=1,
            )
            predicted_ohc = self.data.ohc(monthly)[0]
            observed_ohc = np.where(
                np.isfinite(predicted_ohc), sample["raw"]["ohc"], np.nan
            )
            predictions.append(surface)
            references.append(truth)
            ohc.append(predicted_ohc)
            reference_ohc.append(observed_ohc)
            if export:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_origin",
                            "method": Path(export).stem,
                            "origin": sample["name"],
                            "completed": len(predictions),
                            "total": len(paths),
                        }
                    ),
                    flush=True,
                )
            physical_ts = self.data.physical(monthly[:, None])[
                :, 0, self.data.ts_indices
            ]
            target = sample["interior"]
            valid = torch.isfinite(target) & self.data.ts_mask.bool()
            weights = valid * self.data.area
            error = torch.where(valid, physical_ts - target, 0)
            interior_error += (error.square() * weights).sum((0, 2, 3)).cpu().numpy()
            interior_bias += (error * weights).sum((0, 2, 3)).cpu().numpy()
            interior_count += weights.sum((0, 2, 3)).cpu().numpy()
        arrays = dict(
            prediction=np.array(predictions),
            reference=np.array(references),
            predicted_ohc=np.array(ohc),
            reference_ohc=np.array(reference_ohc),
        )
        result = score(
            **arrays,
            lat=self.data.grid["lat"],
            lon=self.data.grid["lon"],
            mask=self.data.grid["mask"][0],
        )

        supported = interior_count > 0
        result["thermohaline"] = {
            "channels": np.array(self.data.grid["names"])[self.data.ts_indices][
                supported
            ].tolist(),
            "rmse": np.sqrt(
                interior_error[supported] / interior_count[supported]
            ).tolist(),
            "bias": (interior_bias[supported] / interior_count[supported]).tolist(),
        }
        result["origins"] = [p.stem for p in paths]
        if export:
            temporary = Path(export).with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays, origins=result["origins"])
            temporary.replace(export)
        return result
