# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved LLC/OBS six-hour experiments. No inference is run by the viewer."""

import os
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LinearColorMapper,
)
from bokeh.palettes import RdBu11, Viridis256
from bokeh.plotting import figure
from llc_data import VARIABLES, LLCCatalog

from data import limits, raster_rows

pn.extension()


class LLCViewer:
    def __init__(self, catalog):
        self.catalog = catalog
        self.busy = False
        self.restoring = True
        self.url_state = ColumnDataSource(data=dict(query=[]))
        self.url_state.js_on_change(
            "data",
            CustomJS(
                code="""
            const url = new URL(window.location.href);
            const state = JSON.parse(cb_obj.data.query[0]);
            for (const [key, value] of Object.entries(state)) url.searchParams.set(key, value);
            window.history.replaceState(window.history.state, '', url);
        """
            ),
        )
        self.model = pn.widgets.Select(
            name="Model / checkpoint",
            options={v["label"]: k for k, v in catalog.models.items()},
            css_classes=["llc-model"],
        )
        self.case = pn.widgets.Select(
            name="Initialization source / origin", css_classes=["llc-case"]
        )
        self.view = pn.widgets.Select(
            name="Task",
            options={
                "Initialized states": "initial",
                "Surface rollout": "surface",
                "Interior rollout": "interior",
                "Initializer input history": "history",
            },
            css_classes=["llc-task"],
        )
        self.variable = pn.widgets.Select(
            name="Field", options=VARIABLES, value="thetao", css_classes=["llc-field"]
        )
        self.depth = pn.widgets.Select(
            name="Depth",
            options={f"{d:g} m": i for i, d in enumerate(catalog.depth)},
            value=0,
            css_classes=["llc-depth"],
        )
        self.frame = pn.widgets.Select(
            name="Valid time / forecast lead", css_classes=["llc-frame"]
        )
        self.reference = pn.widgets.Select(
            name="Reference",
            options={
                "Available OBS, then LLC": "auto",
                "LLC reference / seasonal context": "llc",
                "Observations only": "observations",
            },
            css_classes=["llc-reference"],
        )
        self.lat = pn.widgets.FloatInput(name="Latitude", value=30.0, step=1.0)
        self.lon = pn.widgets.FloatInput(name="Longitude (°E)", value=330.0, step=1.0)
        self.caption = pn.pane.Markdown(css_classes=["llc-caption"])
        self.details = pn.pane.Markdown()
        self.maps, self.sources, self.mappers = [], [], []
        for i in range(3):
            plot = figure(
                x_range=(0, 360),
                y_range=(-90, 90),
                sizing_mode="stretch_width",
                height=None,
                frame_height=160,
                min_width=280,
                tools="pan,wheel_zoom,reset,tap",
                title="Loading",
            )
            plot.js_on_change(
                "inner_width",
                CustomJS(
                    code="""
                const height = Math.round(cb_obj.inner_width / 2);
                if (height > 0 && cb_obj.frame_height !== height) cb_obj.frame_height = height;
            """
                ),
            )
            if i:
                plot.x_range, plot.y_range = self.maps[0].x_range, self.maps[0].y_range
            source = ColumnDataSource(
                dict(image=[], values=[], x=[], y=[], dw=[], dh=[])
            )
            renderer = plot.image_rgba(
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                source=source,
                nonselection_alpha=1,
            )
            mapper = LinearColorMapper(
                palette=Viridis256 if i < 2 else list(reversed(RdBu11)), low=0, high=1
            )
            plot.add_layout(ColorBar(color_mapper=mapper), "below")
            plot.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("Lon", "$x{0.00}"),
                        ("Lat", "$y{0.00}"),
                        ("Value", "@values{0.0000}"),
                    ],
                )
            )
            plot.on_event(Tap, self.tap)
            self.maps.append(plot)
            self.sources.append(source)
            self.mappers.append(mapper)
        self.maps[0].js_on_event(
            "document_ready", CustomJS(args=dict(state=self.url_state), code="")
        )
        self.series = pn.pane.Bokeh(sizing_mode="stretch_width")
        self.profile = pn.pane.Bokeh(sizing_mode="stretch_width")
        for widget in (
            self.model,
            self.case,
            self.view,
            self.variable,
            self.depth,
            self.frame,
            self.reference,
            self.lat,
            self.lon,
        ):
            widget.param.watch(self.refresh, "value")
        self.refresh()
        if pn.state.location:
            params = parse_qs(pn.state.location.search.removeprefix("?"))
            for name in (
                "model",
                "case",
                "view",
                "variable",
                "depth",
                "frame",
                "reference",
                "lat",
                "lon",
            ):
                widget = getattr(self, name)
                if name not in params:
                    continue
                raw = params[name][0]
                value = (
                    int(raw)
                    if name in ("depth", "frame")
                    else float(raw)
                    if name in ("lat", "lon")
                    else raw
                )
                options = (
                    widget.options if isinstance(widget, pn.widgets.Select) else None
                )
                allowed = options.values() if isinstance(options, dict) else options
                if allowed is None or value in allowed:
                    widget.value = value

        self.restoring = False
        self.refresh()

    def tap(self, event):
        self.busy = True
        self.lat.value, self.lon.value = event.y, event.x % 360
        self.busy = False
        self.refresh()

    @staticmethod
    def options(widget, choices):
        previous = widget.value
        widget.options = choices
        values = list(choices.values()) if isinstance(choices, dict) else choices
        if previous in values:
            widget.value = previous
        elif values:
            widget.value = values[0]

    def refresh(self, *_):
        if self.busy:
            return
        self.busy = True
        try:
            c = self.catalog
            model = c.models[self.model.value]
            self.options(
                self.case,
                {
                    f"{r['source'].upper()} · {r['origin']}": key
                    for key, r in model["cases"].items()
                },
            )
            r = c.case(self.model.value, self.case.value)
            mode = self.view.value
            is_surface = mode in ("surface", "history")
            self.options(
                self.variable,
                {
                    k: v
                    for k, v in VARIABLES.items()
                    if not is_surface or v in ("thetao", "zos")
                },
            )
            if is_surface:
                self.depth.value = 0
            self.depth.disabled = is_surface or self.variable.value == "zos"
            frame_key = {
                "initial": "initial_times",
                "surface": "times",
                "interior": "saved_times",
                "history": "history_times",
            }[mode]
            hours = (
                [-6, 0]
                if mode == "initial"
                else r["saved_hours"]
                if mode == "interior"
                else list(range(6, r["horizon_hours"] + 1, 6))
                if mode == "surface"
                else list(range(-168, 1, 6))
            )
            self.options(
                self.frame,
                {
                    f"{h:+} h · {t}": i
                    for i, (h, t) in enumerate(zip(hours, r[frame_key], strict=True))
                },
            )
            args = (
                self.model.value,
                self.case.value,
                mode,
                self.frame.value,
                self.variable.value,
                self.depth.value,
            )
            first = c.field(*args)
            second, label, scored = c.reference(*args, self.reference.value)
            channel = c.channel(self.variable.value, self.depth.value)
            wet = c.wet[channel]
            diff = first - second if scored else np.full_like(first, np.nan)
            low, high = limits(first, second)
            dlow, dhigh = limits(diff, symmetric=True)
            titles = [
                "Initializer input" if mode == "history" else model["label"],
                label,
                "Model − reference"
                if scored
                else "No error map for context / missing reference",
            ]
            for i, values in enumerate((first, second, diff)):
                mapper = self.mappers[i]
                mapper.low, mapper.high = (low, high) if i < 2 else (dlow, dhigh)
                self.sources[i].data = raster_rows(
                    values, wet, c.lat, c.lon, mapper.palette, mapper.low, mapper.high
                )
                self.maps[i].title.text = titles[i]
            time = c.timestamp(r, mode, self.frame.value)
            self.caption.object = f"**Valid time: {time} · lead {hours[self.frame.value]:+} h**  \n{label}. Gray: land; white: unavailable."
            if not scored and mode != "history":
                self.caption.object += (
                    " Context is excluded from error and skill calculations."
                )
            y, x = (
                int(np.argmin(abs(c.lat - self.lat.value))),
                int(np.argmin(abs((c.lon - self.lon.value + 180) % 360 - 180))),
            )
            self.diagnostics(r, y, x, channel)
            identity = model["identity"]
            self.details.object = f"### What is this task?\nThe initializer takes one week of SST/SSH surface inputs and ERA5 atmospheric forcing (29 frames, six hours apart), then infers two complete ocean states six hours apart. The forecast runs continuously through day 30; the first three days reproduce the training validation task. Interior snapshots are saved at selected leads. LLC seasonal context matches month, day and hour in an available archive year; its actual date is displayed.\n\n### What is this model?\n{model['label']}: Exponentially averaged model weights (EMA) at update {identity['updates']:,}. The 51 LLC depth levels reach about 946 m; fields are eastward/northward velocity, temperature, salinity and sea surface height (SSH), driven by ERA5. Observations supervise daily sea surface temperature (SST) and SSH; no interior temperature, salinity or velocity observations are provided. The OBS-only model's LLC task adapter has no LLC training.\n\nCheckpoint SHA-256: `{identity['checkpoint_sha256']}`."
            if not self.restoring:
                import json

                self.url_state.data = dict(
                    query=[
                        json.dumps(
                            {
                                name: getattr(self, name).value
                                for name in (
                                    "model",
                                    "case",
                                    "view",
                                    "variable",
                                    "depth",
                                    "frame",
                                    "reference",
                                    "lat",
                                    "lon",
                                )
                            }
                        )
                    ]
                )
        finally:
            self.busy = False

    def diagnostics(self, record, y, x, channel):
        c = self.catalog
        variable = self.variable.value
        point = f"{c.lat[y]:.2f}°N, {c.lon[x]:.2f}°E"
        plot = figure(
            title="Selected cell · " + point,
            height=240,
            sizing_mode="stretch_width",
            x_axis_label="Forecast lead (days)",
        )
        mode = self.view.value
        if mode in ("surface", "history"):
            values = c.array(record["prediction"]["surface"])[
                :, 0 if variable == "thetao" else 1, y, x
            ]
            leads = np.arange(1, len(values) + 1) / 4
            plot.line(leads, values, legend_label="Model", line_width=2)
            obs = c.array(record["observed_surface"])[
                :, 0 if variable == "thetao" else 1, y, x
            ]
            plot.scatter(leads, obs, legend_label="Observed", color="black", size=5)
        else:
            values = c.array(record["prediction"]["interior"])[:, channel, y, x]
            plot.line(
                np.array(record["saved_hours"]) / 24,
                values,
                legend_label="Model",
                line_width=2,
            )
        self.series.object = plot
        profile = figure(
            title="Selected-cell depth profile · " + point,
            height=330,
            sizing_mode="stretch_width",
            y_range=(float(c.interfaces[-1]), 0),
            y_axis_label="Depth (m)",
        )
        if mode in ("initial", "interior") and variable != "zos":
            channels = [c.channel(variable, k) for k in range(51)]
            frame = self.frame.value
            values = c.array(record["prediction"][mode])[frame, channels, y, x]
            context = c.array(record["llc_" + mode])[frame, channels, y, x]
            exact = record["llc_" + mode + "_exact"][frame]
            label = (
                ("LLC matched" if exact else "LLC seasonal")
                + " · "
                + record["llc_" + mode + "_times"][frame]
            )
            profile.line(values, c.depth, legend_label="Model", line_width=2)
            profile.line(
                context, c.depth, legend_label=label, color="orange", line_width=2
            )
        self.profile.object = profile

    def layout(self):
        return pn.template.FastListTemplate(
            title="Samudra Results Viewer · LLC / OBS",
            sidebar=[
                pn.pane.Markdown("[Other Samudra results](app)"),
                self.model,
                self.case,
                self.view,
                self.variable,
                self.depth,
                self.frame,
                self.reference,
                self.lat,
                self.lon,
            ],
            main=[
                self.caption,
                pn.Row(*self.maps, sizing_mode="stretch_width"),
                self.series,
                self.profile,
                self.details,
            ],
        )


root = Path(
    os.environ.get("SAMUDRA_LLC_VIEWER_DATA", Path(__file__).parent / ".data-llc")
)
if (root / "catalog.json").is_file():
    LLCViewer(LLCCatalog(root)).layout().servable()
else:
    pn.Column(
        "# Samudra LLC / OBS results",
        "The longer experiments and their 30-day evaluation exports are pending. Verified results will appear here after both runs finish.",
        pn.pane.Markdown("[Other Samudra results](app)"),
    ).servable()
