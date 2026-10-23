# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved original-A denoising anatomy; no inference in the viewer."""

import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import panel as pn
from bokeh.palettes import RdBu11, Turbo256
from PIL import Image


@lru_cache(maxsize=16)
def load(path):
    with np.load(path) as z:
        return dict(z)


def image_html(values, low, high, divergent=False, width=180):
    values = np.asarray(values)
    valid = np.isfinite(values)
    scale = np.clip(
        (np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-10), 0, 1
    )
    colors = RdBu11[::-1] if divergent else Turbo256
    palette = np.array([tuple(bytes.fromhex(c[1:])) for c in colors], dtype=np.uint8)
    rgb = palette[(scale * (len(palette) - 1)).astype(int)]
    rgb[~valid] = [220, 220, 220]
    output = io.BytesIO()
    Image.fromarray(rgb[::-1]).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    clipped = (
        np.mean((values[valid] < low) | (values[valid] > high)) * 100
        if valid.any()
        else 0
    )
    gradient = ",".join(colors)
    return f'<img width="{width}" style="max-width:100%;image-rendering:pixelated" src="data:image/png;base64,{encoded}"><br><span style="display:inline-block;width:{width}px;max-width:100%;height:7px;background:linear-gradient(to right,{gradient})"></span><br><small>{low:.3g} … {high:.3g}; clipped {clipped:.1f}%</small>'


class Anatomy:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "COMPLETE.json").read_text())
        self.grid = load(str(self.root / "grid.npz"))
        self.names = self.grid["names"].tolist()
        depths = self.grid.get("depth", np.arange(19))
        self.labels = [
            f"{name} · {depths[i % 19]:g} m" if i < 76 else "zos · sea surface height"
            for i, name in enumerate(self.names)
        ]
        self.source = pn.widgets.RadioButtonGroup(
            label="Example",
            options={
                "OM4 · 2k": "om4",
                "OBS · 2k": "obs_pre",
                "OBS · 3k cooled": "obs",
            },
            value="om4",
        )
        self.stage = pn.widgets.Select(
            label="Stage",
            options={
                "Initializer inputs (19 frames)": "inputs",
                "Initialized state decode": "initial",
                "One processor step → day 5 decode": "day5",
                "Monthly decode (separate task)": "monthly",
                "Initialized full77 diagnostic (OBS: OM4 context)": "initial-full",
                "Day 5 full77 diagnostic (OBS: OM4 context)": "day5-full",
            },
            value="initial",
        )
        self.units = pn.widgets.RadioButtonGroup(
            label="Units", options=["Physical", "Normalized"], value="Physical"
        )
        self.readout = pn.widgets.Select(
            label="Display",
            options=["Final sample", "Diffusion step: input + prediction"],
            value="Final sample",
        )
        self.step = pn.widgets.IntSlider(
            label="Diffusion interval", start=1, end=32, value=32
        )
        self.corrector = pn.widgets.Checkbox(
            label="Inspect Heun correction call", value=False
        )
        self.history = pn.widgets.IntSlider(
            label="Historical frame (19 = latest)", start=1, end=19, value=19
        )
        self.field = pn.widgets.Select(
            label="Inspect field",
            options={n: i for i, n in enumerate(self.labels)},
            value=38,
        )
        self.latent_channel = pn.widgets.IntSlider(
            label="Latent channel", start=0, end=127, value=0
        )
        self.caption = pn.pane.Markdown()
        self.detail = pn.pane.HTML()
        self.auxiliary = pn.pane.HTML()
        self.inspector = pn.Accordion(
            ("Selected field at native resolution", self.detail), active=[]
        )
        self.tiles = pn.GridBox(ncols=4, sizing_mode="stretch_width")
        self.latents = pn.pane.HTML()
        self.panels = []
        # Stable variable-by-depth order for all three examples. SSH retains its own tile.
        for i in [v * 19 + d for d in range(19) for v in range(4)] + [76]:
            button = pn.widgets.Button(label=self.labels[i], color="light", width=210)
            button.on_click(lambda event, i=i: setattr(self.field, "value", i))
            pane = pn.pane.HTML(sizing_mode="stretch_width")
            self.panels.append((i, pane))
            self.tiles.append(pn.Column(button, pane, sizing_mode="stretch_width"))
        for widget in [
            self.source,
            self.stage,
            self.units,
            self.readout,
            self.step,
            self.corrector,
            self.history,
        ]:
            widget.param.watch(self.update, "value")
        self.field.param.watch(self.update_detail, "value")
        self.latent_channel.param.watch(self.update_latents, "value")
        self.update()

    def transform(self, value, channel, difference=False):
        if self.units.value == "Normalized":
            return value
        return value * self.grid["std"][channel] + (
            0 if difference else self.grid["mean"][channel]
        )

    def stage_for(self, source):
        return (
            self.stage.value.removesuffix("-full")
            if source == "om4"
            else self.stage.value
        )

    def values(self, source):
        base = self.root / source
        if self.stage.value == "inputs":
            z = load(str(base / "inputs.npz"))
            i = self.history.value - 1
            x = np.where(z["available"][i].astype(bool), z["fields"][i], np.nan)
            return x, x, np.isfinite(x)
        z = load(str(base / self.stage_for(source) / "final.npz"))
        mask = z["support"].astype(bool)
        value = z["prediction"]
        if self.readout.value != "Final sample":
            call = 2 * (self.step.value - 1) + (
                self.corrector.value and self.step.value < 32
            )
            value = load(str(base / self.stage_for(source) / f"call-{call:02d}.npz"))[
                "clean"
            ]
        return np.where(mask, value, np.nan), np.where(mask, z["gold"], np.nan), mask

    def limits(self, channel):
        # Shared scales depend on final/gold values in all three cases, never the
        # selected source or diffusion step. Early high-sigma clipping is explicit.
        values = []
        for source in ["om4", "obs_pre", "obs"]:
            if self.stage.value == "inputs":
                z = load(str(self.root / source / "inputs.npz"))
                x = np.where(
                    z["available"][:, channel].astype(bool),
                    z["fields"][:, channel],
                    np.nan,
                )
                values.append(x[np.isfinite(x)])
            else:
                z = load(str(self.root / source / self.stage_for(source) / "final.npz"))
                for key in ["prediction", "gold"]:
                    x = z[key][channel][z["support"][channel].astype(bool)]
                    values.append(x[np.isfinite(x)])
        flat = np.concatenate(values)
        if not len(flat):
            return -1.0, 1.0
        lo, hi = np.percentile(self.transform(flat, channel), [2, 98])
        return float(lo), float(hi if hi > lo else lo + 1)

    def unit(self, i):
        if self.units.value == "Normalized":
            return "OM4 standard deviations (zero = OM4 mean)"
        return "m/s" if i < 38 else "°C" if i < 57 else "g/kg" if i < 76 else "m"

    def render(self, i, width=180):
        p, g, mask = self.current
        if not mask[i].any():
            return (
                '<div style="height:150px;background:#eee;padding:8px">Not supplied</div>'
                if self.stage.value == "inputs"
                else '<div style="height:150px;background:#eee;padding:8px">Not requested / not supervised in this task. No gold claimed.</div>'
            )
        p, g = self.transform(p[i], i), self.transform(g[i], i)
        lo, hi = self.scales[i]
        if self.stage.value == "inputs":
            return f"<b>Actually visible · {self.unit(i)}</b><br>" + image_html(
                p, lo, hi, width=width
            )
        error = p - g
        rmse = float(np.sqrt(np.nanmean(error**2)))
        error_scale = max(abs(hi - lo) / 2, 1e-6)
        gold_label = (
            "OM4 context (not OBS truth)"
            if self.stage.value.endswith("-full") and self.source.value != "om4"
            else "Gold"
        )
        prediction_label = "Prediction"
        noisy_panel = ""
        if self.readout.value != "Final sample":
            call = 2 * (self.step.value - 1) + (
                self.corrector.value and self.step.value < 32
            )
            path = (
                self.root
                / self.source.value
                / self.stage_for(self.source.value)
                / f"call-{call:02d}.npz"
            )
            noisy = np.where(mask[i], load(str(path))["noisy"][i], np.nan)
            noisy = self.transform(noisy, i)
            noisy_panel = f"<div><span>Noisy input</span><br>{image_html(noisy, lo, hi, width=width)}</div>"
            prediction_label = "Denoiser prediction"
        return f'<div><b>{self.unit(i)}</b> · prediction RMSE {rmse:.4g}<br><div style="display:grid;grid-template-columns:repeat(2,max-content);gap:6px">{noisy_panel}<div><span>{prediction_label}</span><br>{image_html(p, lo, hi, width=width)}</div><div>{gold_label}<br>{image_html(g, lo, hi, width=width)}</div><div>Prediction error<br>{image_html(error, -error_scale, error_scale, True, width)}</div></div></div>'

    def update(self, *events):
        self.current = self.values(self.source.value)
        self.scales = [self.limits(i) for i in range(77)]
        case = self.manifest["cases"][self.source.value]
        source = self.source.value
        self.caption.object = f"""## Original A · {source.upper()} · {self.stage.value}
Forecast origin 2015-01-01, global 1°. Initial target: Dec 27–Jan 1; day-5 target: Jan 1–6 (exclusive ends). Monthly target: January. Spatial RMSE is an unweighted map diagnostic, not the official pooled validation score. **{case["step"]} updates, EMA weights**. Checkpoint `{case["checkpoint_sha256"][:12]}`.
OM4 2k versus OBS 2k holds checkpoint fixed. OBS 2k versus OBS 3k holds data fixed and changes cooldown exposure.

Every tile stays in the same variable/depth position. All ocean fields use the **same OM4 normalization**.
Color limits pool final predictions and gold across all three examples (2nd–98th percentiles); they stay fixed while switching sources or diffusion steps. Gray = unavailable/dry.
Maps use native grid pixels, south-to-north, with longitude increasing left-to-right; Gaussian latitude spacing is not uniform in degrees.

**Training vs sampling:** this is one fixed draw, not an eight-member mean. The 32 Heun intervals use 63 denoiser calls.
Training uses individual noisy-target denoising calls. Gold never enters latent evolution or this sampler.
Monthly fields are direct raw decoder outputs from the averaged latent, **without replacing top temperature with predicted SST**.
OBS monthly support excludes top temperature; missing channels remain blank.
"""
        if self.stage.value != "inputs" and self.readout.value != "Final sample":
            call = 2 * (self.step.value - 1) + (
                self.corrector.value and self.step.value < 32
            )
            meta = json.loads(
                (self.root / source / self.stage_for(source) / "trace.json").read_text()
            )["calls"][call]
            self.caption.object += f"\nSelected call {call}: {meta['role']}; noise sigma **{meta['sigma']:.6g}**. Early noise may exceed the fixed color range; clipping percentages are shown.\n"
        if self.stage.value.endswith("-full"):
            self.caption.object += "\n**Diagnostic full77 request:** for OBS this differs from its trained surface-only task. The reference is contemporaneous OM4 simulation context, not observational truth or OBS supervision.\n"
        for i, pane in self.panels:
            pane.object = self.render(i, 120)
        self.update_detail()
        self.update_auxiliary()
        self.update_latents()

    def update_detail(self, *events):
        i = self.field.value
        if events:
            self.inspector.active = [0]
        self.detail.object = f"<h3>{self.labels[i]}</h3>" + self.render(i, 360)

    def update_auxiliary(self):
        z = load(str(self.root / self.source.value / "inputs.npz"))
        index = (
            self.history.value - 1
            if self.stage.value == "inputs"
            else 19
            if self.stage.value.startswith("day5")
            else 18
        )
        html = "<h3>Processor / initializer forcing and context</h3><p>Adapter outputs are in model coordinates, not physical flux units. Forcing at frame 20 drives the first processor step.</p>"
        for i in range(3):
            x = z["forcing"][index, i]
            if self.source.value == "om4" and self.units.value == "Physical":
                x = x * z["forcing_std"][i] + z["forcing_mean"][i]
            lo, hi = np.percentile(x, [2, 98])
            html += f'<div style="display:inline-block">{["tauuo (N/m²)", "tauvo (N/m²)", "hfds (W/m²)"][i] if self.source.value == "om4" and self.units.value == "Physical" else "Model forcing channel " + str(i) + " (normalized/adapter coordinates)"}<br>{image_html(x, lo, hi, width=250)}</div>'
        for i, name in enumerate(
            ["Geographic x", "Geographic y", "Geographic z", "Season sin", "Season cos"]
        ):
            x = z["contexts"][index, i]
            html += f'<div style="display:inline-block">{name}<br>{image_html(x, -1, 1, width=180)}</div>'
        if self.source.value != "om4":
            atmosphere = load(str(self.root / self.source.value / "atmosphere.npz"))
            html += "<h3>Eight ERA5 input channels (before learned adapter)</h3>"
            for i in range(8):
                x = atmosphere["normalized"][index, i]
                if self.units.value == "Physical":
                    x = x * atmosphere["std"][i] + atmosphere["mean"][i]
                lo, hi = np.percentile(x, [2, 98])
                html += f'<div style="display:inline-block">{["10m zonal wind (m/s)", "10m meridional wind (m/s)", "2m temperature (K)", "2m dewpoint (K)", "Surface pressure (Pa)", "Downward solar radiation (source units)", "Downward thermal radiation (source units)", "Total precipitation (source units)"][i]}<br>{image_html(x, lo, hi, width=250)}</div>'
        self.auxiliary.object = html

    def update_latents(self, *events):
        z = load(str(self.root / self.source.value / "inputs.npz"))["latents"]
        c = self.latent_channel.value
        html = "<h3>Latent state: initialization → one processor step → difference</h3><p>Latents have learned coordinates and no physical-unit conversion. Both memory slots are shown.</p>"
        for slot in range(2):
            first, next_ = z[0, slot, c], z[1, slot, c]
            limit = max(
                float(
                    np.max(
                        np.abs(
                            load(str(self.root / source / "inputs.npz"))["latents"][
                                :2, slot, c
                            ]
                        )
                    )
                )
                for source in ["om4", "obs_pre", "obs"]
            )
            limit = max(limit, 1e-6)
            html += f"<p>Slot {slot}, channel {c}; full-slot RMS {np.sqrt(np.mean(z[0, slot] ** 2)):.4g} → {np.sqrt(np.mean(z[1, slot] ** 2)):.4g}</p>"
            html += '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start">'
            for label, a in zip(
                ["Initialized", "After one step", "Difference (after − initialized)"],
                [first, next_, next_ - first],
                strict=True,
            ):
                html += f'<div style="width:300px;max-width:100%"><strong>{label}</strong><br>{image_html(a, -limit, limit, True, 300)}</div>'
            html += "</div>"
        self.latents.object = html

    def view(self):
        return pn.template.FastListTemplate(
            title="A · initializer and processor anatomy",
            sidebar=[
                self.source,
                self.stage,
                self.units,
                self.readout,
                self.step,
                self.corrector,
                self.history,
                self.field,
                self.latent_channel,
            ],
            main=[
                self.caption,
                self.inspector,
                pn.Accordion(
                    ("Forcings and geographic/seasonal inputs", self.auxiliary),
                    ("Latent state", self.latents),
                ),
                self.tiles,
            ],
            sidebar_width=320,
        )
