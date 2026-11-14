# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved physical-state diffusion validation; never performs model inference."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import panel as pn
from step_anatomy import image_html
from xray_images import XrayImages


@lru_cache(maxsize=2)
def load_case(path):
    with np.load(path) as data:
        return dict(data)


class PhysicalXray:
    """An explicit artifact manifest keeps missing references genuinely missing."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "COMPLETE.json").read_text())
        if self.manifest.get("schema") != "physical-xray-v1":
            raise ValueError("Not a completed physical Xray bundle")
        for name, expected in self.manifest["sha256"].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("Artifact escapes bundle")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        with np.load(self.root / "grid.npz") as data:
            self.grid = dict(data)
        self.cases = self.manifest["cases"]
        self.case = pn.widgets.Select(
            label="Example", options={c["label"]: i for i, c in enumerate(self.cases)}
        )
        self.field = pn.widgets.Select(
            label="Physical field",
            options={str(n): i for i, n in enumerate(self.grid["names"])},
            value=38,
        )
        self.readout = pn.widgets.Select(
            label="Prediction", options=["Ensemble mean", "Member 1"]
        )
        self.units = pn.widgets.RadioButtonGroup(
            label="Units", options=["Physical", "Normalized"]
        )
        self.caption = pn.pane.Markdown()
        self.maps = XrayImages(sizing_mode="stretch_width")
        self._render_version = 0
        self.metrics = pn.pane.Markdown()
        for widget in [self.case, self.field, self.readout, self.units]:
            widget.jscallback(
                args={
                    "images": self.maps,
                    "example": self.case,
                    "field": self.field,
                    "readout": self.readout,
                    "units": self.units,
                },
                value="images.data.selection = JSON.stringify([example.value, field.value, readout.value, units.active]); images.data.busy = true;",
            )
            widget.param.watch(self.update, "value")
        self.update()

    def update(self, *_):
        self._render_version += 1
        try:
            self.render_maps()
        except Exception:
            self.maps.content = {
                "selection": self.selection(),
                "html": f"<!--{self._render_version}--><p>Unable to load this view. Please choose another selection.</p>",
            }
            raise

    def selection(self):
        return [
            str(self.case.value),
            str(self.field.value),
            self.readout.value,
            0 if self.units.value == "Physical" else 1,
        ]

    def render_maps(self):
        case = self.cases[self.case.value]
        data = load_case(str(self.root / case["file"]))
        i = self.field.value
        wet = self.grid["wet_mask"][i].astype(bool)
        key = (
            "prediction_mean"
            if self.readout.value == "Ensemble mean"
            else "prediction_member0"
        )
        prediction = np.where(wet, data[key][i], np.nan)
        gold = np.where(
            wet & data["gold_support"][i].astype(bool), data["gold"][i], np.nan
        )
        spread = np.where(wet, data["spread"][i], np.nan)
        if self.units.value == "Physical":
            prediction = prediction * self.grid["std"][i] + self.grid["mean"][i]
            gold = gold * self.grid["std"][i] + self.grid["mean"][i]
            spread *= self.grid["std"][i]
        # Mean/member share limits, avoiding artificial apparent differences.
        samples = [
            data[k][i][wet] for k in ["prediction_mean", "prediction_member0", "gold"]
        ]
        flat = np.concatenate(samples)
        flat = flat[np.isfinite(flat)]
        if self.units.value == "Physical":
            flat = flat * self.grid["std"][i] + self.grid["mean"][i]
        low, high = np.percentile(flat, [2, 98]) if flat.size else (0, 1)
        if high <= low:
            high = low + 1
        error = prediction - gold
        finite_error = np.abs(error[np.isfinite(error)])
        bound = (
            max(float(np.percentile(finite_error, 98)), 1e-10)
            if finite_error.size
            else 1
        )
        finite_spread = spread[np.isfinite(spread)]
        upper_spread = (
            max(float(np.percentile(finite_spread, 98)), 1e-10)
            if finite_spread.size
            else 1
        )
        panels = []
        for title, values, lo, hi, divergent in [
            (self.readout.value, prediction, low, high, False),
            (case["reference_label"], gold, low, high, False),
            ("Prediction − reference", error, -bound, bound, True),
            ("Member standard deviation", spread, 0, upper_spread, False),
        ]:
            panels.append(
                f"<div><b>{title}</b><br>{image_html(values, lo, hi, divergent, width=720)}</div>"
            )
        self.maps.content = {
            "selection": self.selection(),
            "html": (
                f"<!--render {self._render_version}-->"
                '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px">'
                + "".join(panels)
                + "</div>"
            ),
        }
        unit = (
            "OM4 standard deviations"
            if self.units.value == "Normalized"
            else "m/s"
            if i < 38
            else "°C"
            if i < 57
            else "g/kg"
            if i < 76
            else "m"
        )
        self.caption.object = f"### {case['label']} · {self.grid['names'][i]} ({unit})\n\n{case['reference_note']}"
        paired = np.isfinite(error)
        if paired.any():
            weights = np.broadcast_to(
                np.cos(np.deg2rad(self.grid["lat"]))[:, None], error.shape
            )
            rms = np.sqrt(
                np.sum(weights[paired] * error[paired] ** 2) / np.sum(weights[paired])
            )
            bias = np.sum(weights[paired] * error[paired]) / np.sum(weights[paired])
            self.metrics.object = f"Snapshot RMS **{rms:.4g}**, bias **{bias:.4g}** ({paired.sum():,} paired cells). Cosine-weighted snapshot diagnostic; not the blended annual score."
        else:
            self.metrics.object = "No aligned reference is available for this field; no error or score is shown."

    def view(self):
        return pn.template.FastListTemplate(
            title="Physical-state diffusion Xray",
            header_background="#143b4a",
            sidebar=[
                self.case,
                self.field,
                self.readout,
                self.units,
                pn.pane.Markdown(
                    "[Results viewer](?xray=0) · [Original A Xray](?xray=1)"
                ),
            ],
            main=[
                self.caption,
                self.maps,
                self.metrics,
                pn.Accordion(
                    (
                        "What is this model?",
                        pn.pane.Markdown(self.manifest["model_description"]),
                    ),
                    (
                        "Evaluation provenance",
                        pn.pane.JSON(self.manifest["provenance"], depth=2),
                    ),
                    active=[],
                ),
            ],
        )
