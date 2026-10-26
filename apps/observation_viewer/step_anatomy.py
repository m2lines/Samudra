# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved original-A denoising anatomy; no inference in the viewer."""

import base64
import io
import json
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import panel as pn
from bokeh.palettes import RdBu11, Turbo256
from PIL import Image


@lru_cache(maxsize=16)
def load(path):
    with np.load(path) as z:
        return dict(z)


@lru_cache(maxsize=32)
def denoiser_rmse(directory):
    """Normalized, unweighted wet-cell RMSE for every saved denoiser call."""
    directory = Path(directory)
    cache = directory / "denoiser-rmse.json"
    if cache.exists():
        values = json.loads(cache.read_text())
        return np.asarray(values["calls"]), np.asarray(values["rmse"], dtype=float)
    with np.load(directory / "final.npz") as final:
        gold = final["gold"].astype(np.float64)
        support = final["support"].astype(bool) & np.isfinite(gold)
    calls, rows = [], []
    for path in sorted(directory.glob("call-*.npz")):
        with np.load(path) as trace:
            error = trace["clean"].astype(np.float64) - gold
        valid = support & np.isfinite(error)
        count = valid.sum(axis=(-2, -1))
        mse = np.divide(
            np.where(valid, error**2, 0).sum(axis=(-2, -1)),
            count,
            out=np.full(len(count), np.nan),
            where=count > 0,
        )
        rows.append(np.sqrt(mse))
        calls.append(int(path.stem.split("-")[1]))
    return np.asarray(calls), np.asarray(rows)


def rmse_sparkline(calls, values, selected, unit):
    valid = np.isfinite(values)
    if not valid.any():
        return ""
    ceiling = max(float(np.max(values[valid])), 1e-12)
    x = 5 + 140 * calls / max(float(calls[-1]), 1)
    y = 39 - 34 * values / ceiling
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x[valid], y[valid]))
    marker = ""
    for j in np.flatnonzero((calls == selected) & valid):
        marker = f'<circle cx="{x[j]:.2f}" cy="{y[j]:.2f}" r="3" fill="#dc2626"/>'
    title = f"Denoiser RMSE ({unit}); calls 0–{calls[-1]} including Heun correctors; y: 0–{ceiling:.4g}; red dot: call {selected}"
    return (
        f'<svg class="rmse-sparkline" role="img" aria-label="{title}" '
        'width="150" height="48" viewBox="0 0 150 48">'
        f'<title>{title}</title><path d="M5 5 V39 H145" fill="none" stroke="#999"/>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1.5"/>'
        f"{marker}</svg>"
    )


def image_html(values, low, high, divergent=False, width=180):
    values = np.asarray(values)
    valid = np.isfinite(values)
    scale = np.clip(
        (np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-10), 0, 1
    )
    colors = RdBu11[::-1] if divergent else Turbo256[40:216]
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
        self.nn_input = pn.widgets.Checkbox(label="NN input scaling", value=True)
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
            groups={"Available": {n: i for i, n in enumerate(self.labels)}},
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
        self.sparks = {}
        # Stable variable-by-depth order for all three examples. SSH retains its own tile.
        for i in [v * 19 + d for d in range(19) for v in range(4)] + [76]:
            button = pn.widgets.Button(label=self.labels[i], color="light", width=210)
            button.on_click(partial(self.select_field, i))
            pane = pn.pane.HTML(sizing_mode="stretch_width")
            self.panels.append((i, pane))
            spark = pn.pane.HTML(width=150, height=48, margin=0, visible=False)
            self.sparks[i] = spark
            self.tiles.append(
                pn.Column(pn.Row(button, spark), pane, sizing_mode="stretch_width")
            )
        for widget in [
            self.source,
            self.stage,
            self.units,
            self.readout,
            self.nn_input,
            self.step,
            self.corrector,
            self.history,
        ]:
            widget.param.watch(self.update, "value")
        self.field.param.watch(self.update_detail, "value")
        self.latent_channel.param.watch(self.update_latents, "value")
        self._render_generation = 0
        self.update()

    def select_field(self, index: int, event):
        self.field.value = index

    @property
    def interval(self) -> int:
        value = self.step.value
        assert value is not None
        return value

    @property
    def history_index(self) -> int:
        value = self.history.value
        assert value is not None
        return value

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
            i = self.history_index - 1
            x = np.where(z["available"][i].astype(bool), z["fields"][i], np.nan)
            return x, x, np.isfinite(x)
        z = load(str(base / self.stage_for(source) / "final.npz"))
        mask = z["support"].astype(bool)
        value = z["prediction"]
        if self.readout.value != "Final sample":
            call = 2 * (self.interval - 1) + (
                self.corrector.value and self.interval < 32
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
        panels = []
        if self.readout.value != "Final sample":
            call = 2 * (self.interval - 1) + (
                self.corrector.value and self.interval < 32
            )
            path = (
                self.root
                / self.source.value
                / self.stage_for(self.source.value)
                / f"call-{call:02d}.npz"
            )
            trace = load(str(path))
            noisy = np.where(mask[i], trace["noisy"][i], np.nan)
            input_lo, input_hi = lo, hi
            input_label = "Before NN scaling"
            if self.nn_input.value:
                sigma = float(trace["sigma"])
                factor = 1 / np.sqrt(1 + sigma**2)
                noisy = noisy * factor
                input_lo, input_hi = -3.0, 3.0
                input_label = (
                    f"NN input · dimensionless · ×{factor:.5g} · fixed ±3 scale"
                )
            else:
                noisy = self.transform(noisy, i)
                input_label += f" · {self.unit(i)}"
            panels.append(
                (
                    f"<span>Noisy input</span><br><small>{input_label}</small>",
                    image_html(noisy, input_lo, input_hi, width=width),
                )
            )
            prediction_label = "Denoiser prediction"
        panels.extend(
            [
                (
                    f"<span>{prediction_label}</span>",
                    image_html(p, lo, hi, width=width),
                ),
                (gold_label, image_html(g, lo, hi, width=width)),
                (
                    "Prediction error",
                    image_html(error, -error_scale, error_scale, True, width),
                ),
            ]
        )
        rows = []
        for start in range(0, len(panels), 2):
            pair = panels[start : start + 2]
            # Share the header row so wrapped subtitles cannot offset the maps.
            headers = "".join(f"<div>{title}</div>" for title, _ in pair)
            images = "".join(f"<div>{content}</div>" for _, content in pair)
            if len(pair) == 1:
                headers += "<div></div>"
            rows.append(headers + images)
        return (
            f"<div><b>{self.unit(i)}</b> · prediction RMSE {rmse:.4g}<br>"
            f'<div class="xray-field-grid" style="display:grid;'
            f'grid-template-columns:repeat(2,{width}px);gap:6px">'
            + "".join(rows)
            + "</div></div>"
        )

    def update(self, *events):
        self.current = self.values(self.source.value)
        self._render_generation += 1
        generation = self._render_generation
        present = self.current[2].any(axis=(-2, -1))
        groups = {"Available": {self.labels[i]: i for i in range(77) if present[i]}}
        if not present.all():
            groups["Not present in this view"] = {
                self.labels[i]: i for i in range(77) if not present[i]
            }
        selected = self.field.value
        self.field.param.update(groups=groups, value=selected)
        self.scales = {selected: self.limits(selected)}
        case = self.manifest["cases"][self.source.value]
        source = self.source.value
        self.caption.object = f"""## Original A · {source.upper()} · {self.stage.value}
Forecast origin 2015-01-01, global 1°. Initial target: Dec 27–Jan 1; day-5 target: Jan 1–6 (exclusive ends). Monthly target: January. Spatial RMSE is an unweighted map diagnostic, not the official pooled validation score. **{case["step"]} updates, EMA weights**. Checkpoint `{case["checkpoint_sha256"][:12]}`.
OM4 2k versus OBS 2k holds checkpoint fixed. OBS 2k versus OBS 3k holds data fixed and changes cooldown exposure.

Every tile stays in the same variable/depth position. All ocean fields use the **same OM4 normalization**.
Color limits pool final predictions and gold across all three examples (2nd–98th percentiles); they stay fixed while switching sources or diffusion steps. Gray = unavailable/dry.
Maps use native grid pixels, south-to-north, with longitude increasing left-to-right; Gaussian latitude spacing is not uniform in degrees.

**Training vs sampling:** this is one fixed draw, not an eight-member mean. The 32 Heun intervals use 63 denoiser calls.
With NN input scaling enabled, only the noisy-input panel shows the dimensionless first-convolution input, divided by √(1+σ²), on a fixed −3 to +3 scale. The physical/normalized units selector still controls predictions and gold.
Sparklines show unweighted denoiser-prediction RMSE versus gold across all 63 calls, including Heun correctors. Each has a zero-based vertical scale; hover for its range and units. The red dot marks the selected call.
Training uses individual noisy-target denoising calls. Gold never enters latent evolution or this sampler.
Monthly fields are direct raw decoder outputs from the averaged latent, **without replacing top temperature with predicted SST**.
OBS monthly support excludes top temperature; missing channels remain blank.
"""
        if self.stage.value != "inputs" and self.readout.value != "Final sample":
            call = 2 * (self.interval - 1) + (
                self.corrector.value and self.interval < 32
            )
            meta = json.loads(
                (self.root / source / self.stage_for(source) / "trace.json").read_text()
            )["calls"][call]
            self.caption.object += f"\nSelected call {call}: {meta['role']}; noise sigma **{meta['sigma']:.6g}**. Early noise may exceed the fixed color range; clipping percentages are shown.\n"
        if self.stage.value.endswith("-full"):
            self.caption.object += "\n**Diagnostic full77 request:** for OBS this differs from its trained surface-only task. The reference is contemporaneous OM4 simulation context, not observational truth or OBS supervision.\n"
        self.update_detail()

        def render_remaining():
            if generation != self._render_generation:
                return
            for i, pane in self.panels:
                if i not in self.scales:
                    self.scales[i] = self.limits(i)
                pane.object = self.render(i, 120)
                self.sparks[i].object = self.sparkline(i)
                self.sparks[i].visible = bool(self.sparks[i].object)
            self.update_auxiliary()
            self.update_latents()

        document = pn.state.curdoc
        if document is not None and document.session_context is not None:
            # Let the selected-field patch reach the browser before bulk rendering.
            document.add_timeout_callback(render_remaining, 50)
        else:
            render_remaining()

    def sparkline(self, i):
        if self.stage.value == "inputs" or self.readout.value == "Final sample":
            return ""
        if not self.current[2][i].any():
            return ""
        calls, errors = denoiser_rmse(
            str(self.root / self.source.value / self.stage_for(self.source.value))
        )
        values = errors[:, i]
        if self.units.value == "Physical":
            values = values * abs(self.grid["std"][i])
        selected = 2 * (self.interval - 1) + (
            self.corrector.value and self.interval < 32
        )
        return rmse_sparkline(calls, values, selected, self.unit(i))

    def update_detail(self, *events):
        i = self.field.value
        if events:
            self.inspector.active = [0]
        if i not in self.scales:
            self.scales[i] = self.limits(i)
        self.detail.object = (
            f'<div style="display:flex;align-items:center;gap:16px"><h3>{self.labels[i]}</h3>'
            + self.sparkline(i)
            + "</div>"
            + self.render(i, 360)
        )

    def update_auxiliary(self):
        z = load(str(self.root / self.source.value / "inputs.npz"))
        index = (
            self.history_index - 1
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
            title="A · initializer and processor Xray",
            sidebar=[
                self.source,
                self.stage,
                self.units,
                self.readout,
                self.nn_input,
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
