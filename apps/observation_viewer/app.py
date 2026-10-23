# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Serve with panel serve app.py; each browser session owns its controls."""

import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LinearColorMapper,
    Range1d,
)
from bokeh.models.annotations import ColorBar, Span
from bokeh.palettes import RdBu, Viridis256
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).parent))
from model_info import description

from data import Catalog, edges, interval, limits, paired_stats, raster_rows

pn.extension(sizing_mode="stretch_width")
APP_ROOT = Path(__file__).resolve().parent
DIFFERENCE_PALETTE = list(RdBu[11])
MODES = {
    "Annual surface forecasts": "surface",
    "Initialized ocean interior": "interior",
    "Saved forecast interior": "forecast_interior",
    "Monthly ocean heat content": "heat",
}
VARIABLES = {
    "surface": {"Sea surface temperature": 0, "Sea surface height": 1},
    "interior": {"Temperature": 0, "Salinity": 1},
    "forecast_interior": {
        "Temperature": 0,
        "Salinity": 1,
        "Zonal velocity (U)": 2,
        "Meridional velocity (V)": 3,
    },
    "heat": {"0–700 m": 0, "700–2000 m": 1},
}
UNITS = {
    "surface": ["°C", "m"],
    "interior": ["°C", "psu", "m/s", "m/s"],
    "forecast_interior": ["°C", "psu", "m/s", "m/s"],
    "heat": ["GJ/m²", "GJ/m²"],
}


def style_plot(plot):
    plot.toolbar.logo = None
    plot.outline_line_color = "#d3dde2"
    plot.grid.grid_line_alpha = 0.15
    plot.axis.axis_label_text_font_style = "normal"
    plot.title.text_font_size = "13px"
    return plot


class Viewer:
    """A linked map/section viewer with independent state for every session."""

    def __init__(self, catalog):
        self._restoring_url = True
        self.url_state = ColumnDataSource(data=dict(state=[]))
        self.url_state.js_on_change(
            "data",
            CustomJS(
                code="""
            const url = new URL(window.location.href);
            url.searchParams.set("view", cb_obj.data.state[0]);
            window.history.replaceState(window.history.state, "", url);
        """
            ),
        )
        self.registry = catalog
        default = (
            "obs08000"
            if "obs08000" in catalog.all_models
            else next(iter(catalog.all_models))
        )
        self.catalog = catalog = catalog.for_model(default)
        self.lat, self.lon = catalog.geometry("surface")
        self.availability_message = ""
        self._changing = False
        self.wet = None
        self.model_options = {
            v["label"]: k for k, v in self.registry.all_models.items()
        }
        self.interior_examples = catalog.meta.get("interior_examples", {})
        self.mode = pn.widgets.Select(
            name="View", options=MODES, value="surface", css_classes=["view-select"]
        )
        self.model = pn.widgets.Select(
            name="Model / checkpoint",
            options=self.model_options,
            value=default,
            css_classes=["model-select"],
        )
        self.origin = pn.widgets.Select(
            name="Forecast origin",
            options=catalog.meta["origins"],
            css_classes=["origin-select"],
        )
        self.variable = pn.widgets.Select(
            name="Variable",
            options=VARIABLES["surface"],
            value=0,
            css_classes=["variable-select"],
        )
        self.reference = pn.widgets.Select(
            name="Compare with",
            options={"Observations": "observations", **self.model_options},
            value="observations",
            css_classes=["reference-select"],
        )
        self.frame = pn.widgets.IntSlider(
            name="Forecast lead (days)",
            start=5,
            end=365,
            step=5,
            value=30,
            css_classes=["lead-slider"],
        )
        self.player = pn.widgets.Player(
            name="Play forecast",
            start=0,
            end=72,
            value=5,
            interval=900,
            loop_policy="once",
            show_loop_controls=False,
            show_value=False,
            height=85,
            visible_buttons=["first", "previous", "pause", "play", "next", "last"],
        )
        self.depth = pn.widgets.Select(
            name="Depth",
            options={f"{v:g} m": i for i, v in enumerate(catalog.depths)},
            value=9,
            visible=False,
            css_classes=["depth-select"],
        )
        self.month = pn.widgets.DiscreteSlider(
            name="Month",
            options={"January": 0},
            value=0,
            visible=False,
            css_classes=["month-slider"],
        )
        self.saved_lead = pn.widgets.Select(
            name="Saved forecast lead (days)",
            options=[5, 15, 30, 90, 180, 365],
            value=30,
            visible=False,
            css_classes=["saved-lead-select"],
        )
        self.latitude = pn.widgets.FloatInput(
            name="Latitude (°N)",
            value=30.0,
            step=1,
            start=-90,
            end=90,
            css_classes=["latitude-input"],
        )
        self.longitude = pn.widgets.FloatInput(
            name="Longitude (°E, 0–360)",
            value=330.0,
            step=1,
            start=0,
            end=360,
            css_classes=["longitude-input"],
        )
        self.orientation = pn.widgets.RadioButtonGroup(
            name="Section direction",
            css_classes=["section-direction"],
            options=["Along longitude", "Along latitude"],
            value="Along longitude",
            visible=False,
        )
        self.lock_colors = pn.widgets.Checkbox(
            name="Keep color limits while stepping", value=True
        )
        self.rescale = pn.widgets.Button(
            name="Rescale colors to this slice", button_type="light"
        )
        self.caption = pn.pane.Markdown("")
        self.summary = pn.pane.Markdown("", css_classes=["snapshot-summary"])
        self.location = pn.pane.Markdown("", css_classes=["point-summary"])
        self.context = pn.pane.Markdown(
            "", css_classes=["task-description"], sizing_mode="stretch_width"
        )
        self.provenance = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.model_description = pn.pane.Markdown(
            "", css_classes=["model-description"], sizing_mode="stretch_width"
        )
        self.maps, self.sources, self.mappers, self.crosshairs = [], [], [], []
        self.quad_renderers, self.raster_renderers, self.raster_sources = [], [], []
        self.xrange, self.yrange = Range1d(start=0, end=360), Range1d(start=-90, end=90)
        le, la = edges(catalog.lon, (0, 360)), edges(catalog.lat, (-90, 90))
        left, bottom = np.meshgrid(le[:-1], la[:-1])
        right, top = np.meshgrid(le[1:], la[1:])
        lon, lat = np.meshgrid(catalog.lon, catalog.lat)
        self.geometry = dict(
            left=left.ravel(),
            right=right.ravel(),
            bottom=bottom.ravel(),
            top=top.ravel(),
            lon=lon.ravel(),
            lat=lat.ravel(),
        )
        for i, key in enumerate(["prediction", "reference", "difference"]):
            plot = style_plot(
                figure(  # type: ignore[call-arg]  # Bokeh's dynamic figure options lack stubs.
                    name=f"map-{key}",
                    title=key.title(),
                    x_range=self.xrange,
                    y_range=self.yrange,
                    x_axis_label="Longitude (°E)",
                    y_axis_label="Latitude (°N)",
                    tools="pan,wheel_zoom,box_zoom,reset,tap,save",
                    active_scroll="wheel_zoom",
                    sizing_mode="stretch_width",
                    height=None,
                    frame_height=160,
                    min_width=280,
                    background_fill_color="#d5dbe0",
                    output_backend="webgl",
                )
            )
            # Preserve the actual map frame's 360:180 ratio, excluding axes,
            # toolbar, title and color bar, whose pixel sizes do not scale.
            plot.js_on_change(
                "inner_width",
                CustomJS(
                    code="""
                    const height = Math.round(cb_obj.inner_width / 2);
                    if (height > 0 && cb_obj.frame_height !== height) {
                        cb_obj.frame_height = height;
                    }
                    """
                ),
            )
            source = ColumnDataSource(
                {k: [] for k in [*self.geometry, "value"]}, name=f"source-{key}"
            )
            mapper = LinearColorMapper(
                palette=Viridis256 if i < 2 else DIFFERENCE_PALETTE,
                low=0,
                high=1,
                nan_color="white",
            )
            renderer = plot.quad(
                left="left",
                right="right",
                bottom="bottom",
                top="top",
                source=source,
                line_color=None,
                fill_color={"field": "value", "transform": mapper},
                nonselection_fill_alpha=1,
            )
            self.quad_renderers.append(renderer)
            raster_source = ColumnDataSource(
                dict(image=[], values=[], x=[], y=[], dw=[], dh=[]),
                name=f"raster-{key}",
            )
            raster_renderer = plot.image_rgba(
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                source=raster_source,
                # Tapping selects a profile location, not a subset of map rows.
                nonselection_alpha=1,
                visible=False,
            )
            self.raster_sources.append(raster_source)
            self.raster_renderers.append(raster_renderer)
            plot.add_tools(
                HoverTool(
                    renderers=[raster_renderer],
                    tooltips=[
                        ("Longitude", "$x{0.00}°E"),
                        ("Latitude", "$y{0.00}°N"),
                        ("Value", "@values{0.0000}"),
                    ],
                )
            )
            plot.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("Longitude", "@lon{0.00}°E"),
                        ("Latitude", "@lat{0.00}°N"),
                        ("Value", "@value{0.0000}"),
                    ],
                )
            )
            plot.add_layout(
                ColorBar(
                    color_mapper=mapper, height=8, orientation="horizontal", title=""
                ),
                "below",
            )
            horizontal, vertical = (
                Span(
                    location=30,
                    dimension="width",
                    line_color="#263742",
                    line_dash="dashed",
                ),
                Span(
                    location=330,
                    dimension="height",
                    line_color="#263742",
                    line_dash="dashed",
                ),
            )
            plot.add_layout(horizontal)
            plot.add_layout(vertical)
            plot.on_event(Tap, self.tap)
            self.maps.append(plot)
            self.sources.append(source)
            self.mappers.append(mapper)
            self.crosshairs.append((horizontal, vertical))
        self.line_source = ColumnDataSource(
            dict(x=[], prediction=[], reference=[]), name="detail-source"
        )
        self.line = style_plot(
            figure(  # type: ignore[call-arg]
                title="Time series at selected cell",
                height=310,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,reset,save",
            )
        )
        self.line.line(
            "x",
            "prediction",
            source=self.line_source,
            color="#087f8c",
            line_width=2,
            legend_label="Selected model",
        )
        self.line.line(
            "x",
            "reference",
            source=self.line_source,
            color="#d27539",
            line_width=2,
            legend_label="Comparison",
        )
        self.line.legend.location = "top_right"
        self.line.legend.click_policy = "hide"
        self.line_hover = HoverTool(
            tooltips=[
                ("Forecast lead (days)", "@x"),
                ("Model", "@prediction{0.0000}"),
                ("Comparison", "@reference{0.0000}"),
            ],
            mode="vline",
        )
        self.line.add_tools(self.line_hover)
        self.marker = Span(
            location=30, dimension="height", line_dash="dashed", line_color="#586b78"
        )
        self.line.add_layout(self.marker)
        self.profile_source = ColumnDataSource(
            dict(depth=[], prediction=[], reference=[]), name="profile-source"
        )
        self.profile = style_plot(
            figure(  # type: ignore[call-arg]
                title="Vertical profile at selected cell",
                y_range=Range1d(start=2100, end=0),
                y_axis_label="Depth (m)",
                height=350,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,reset,save",
            )
        )
        self.profile.line(
            "prediction",
            "depth",
            source=self.profile_source,
            color="#087f8c",
            line_width=2,
            legend_label="Selected model",
        )
        self.profile.line(
            "reference",
            "depth",
            source=self.profile_source,
            color="#d27539",
            line_width=2,
            legend_label="Comparison",
        )
        self.profile.legend.click_policy = "hide"
        self.profile.add_tools(
            HoverTool(
                tooltips=[
                    ("Depth", "@depth m"),
                    ("Model", "@prediction{0.0000}"),
                    ("Comparison", "@reference{0.0000}"),
                ],
                mode="hline",
            )
        )
        self.depth_marker = Span(
            location=550, dimension="width", line_dash="dashed", line_color="#586b78"
        )
        self.profile.add_layout(self.depth_marker)
        self.sections, self.section_sources, self.section_mappers = [], [], []
        for i, key in enumerate(["prediction", "reference", "difference"]):
            plot = style_plot(
                figure(  # type: ignore[call-arg]
                    title=f"{key.title()} section",
                    y_range=Range1d(start=2100, end=0),
                    y_axis_label="Depth (m)",
                    height=280,
                    min_width=280,
                    sizing_mode="stretch_width",
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    background_fill_color="#d5dbe0",
                    output_backend="webgl",
                )
            )
            source = ColumnDataSource(
                {
                    k: []
                    for k in [
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "value",
                        "depth",
                        "coordinate",
                    ]
                },
                name=f"section-{key}",
            )
            mapper = LinearColorMapper(
                palette=Viridis256 if i < 2 else DIFFERENCE_PALETTE,
                low=0,
                high=1,
                nan_color="white",
            )
            renderer = plot.quad(
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                source=source,
                line_color=None,
                fill_color={"field": "value", "transform": mapper},
            )
            plot.add_layout(
                ColorBar(color_mapper=mapper, height=8, orientation="horizontal"),
                "below",
            )
            plot.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=[
                        ("Coordinate", "@coordinate{0.00}°"),
                        ("Depth", "@depth m"),
                        ("Value", "@value{0.0000}"),
                    ],
                )
            )
            self.sections.append(plot)
            self.section_sources.append(source)
            self.section_mappers.append(mapper)
        # Keep the URL state source attached to the rendered Bokeh document.
        self.maps[0].js_on_event(
            "document_ready", CustomJS(args=dict(state=self.url_state), code="")
        )
        self.map_row = pn.Row(
            *(pn.pane.Bokeh(p, sizing_mode="stretch_width") for p in self.maps)
        )
        self.section_row = pn.Row(
            *(pn.pane.Bokeh(p, sizing_mode="stretch_width") for p in self.sections),
            visible=False,
            css_classes=["depth-sections"],
        )
        self.detail = pn.Column(pn.pane.Bokeh(self.line, sizing_mode="stretch_width"))
        self.profile_pane = pn.Column(
            pn.pane.Bokeh(self.profile, sizing_mode="stretch_width"), visible=False
        )
        for control in [self.model, self.reference]:
            control.param.watch(self.select, "value")
        for control in [self.origin, self.variable]:
            control.param.watch(self.change_sample, "value")
        self.mode.param.watch(self.change_mode, "value")
        self.frame.param.watch(self.step, "value")
        self.player.param.watch(self.play, "value")
        self.depth.param.watch(self.change_depth, "value")
        self.month.param.watch(self.change_depth, "value")
        self.saved_lead.param.watch(self.change_depth, "value")
        for point_control in [self.latitude, self.longitude, self.orientation]:
            point_control.param.watch(self.change_point, "value")
        self.lock_colors.param.watch(self.sync_url, "value")
        self.rescale.on_click(lambda event: self.refresh(rescale=True))
        self._changing = True
        self.configure_view()
        self._changing = False
        self.refresh(rescale=True)
        self.restore_url()
        self._restoring_url = False

    def url_controls(self):
        return {
            name: getattr(self, name)
            for name in (
                "model",
                "mode",
                "origin",
                "variable",
                "reference",
                "frame",
                "depth",
                "month",
                "saved_lead",
                "latitude",
                "longitude",
                "orientation",
                "lock_colors",
            )
        }

    def sync_url(self, event=None):
        if self._restoring_url or self._changing:
            return
        state = {name: control.value for name, control in self.url_controls().items()}
        state["colors"] = [[mapper.low, mapper.high] for mapper in self.mappers]
        self.url_state.data = dict(state=[json.dumps(state, separators=(",", ":"))])

    def restore_url(self):
        location = pn.state.location
        if location is None:
            return
        raw = parse_qs(location.search.removeprefix("?")).get("view", [""])[0]
        try:
            state = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(state, dict):
            return
        # Apply in dependency order; existing callbacks rebuild each option list.
        for name, control in self.url_controls().items():
            value = state.get(name)
            if (
                isinstance(control.value, float)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                value = float(value)
            if value is None or type(value) is not type(control.value):
                continue
            if hasattr(control, "options"):
                options = control.options
                if value not in (
                    options.values() if isinstance(options, dict) else options
                ):
                    continue
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not np.isfinite(value):
                    continue
                if (
                    getattr(control, "start", None) is not None
                    and value < control.start
                ):
                    continue
                if getattr(control, "end", None) is not None and value > control.end:
                    continue
                if name == "frame" and value % 5:
                    continue
            control.value = value
        colors = state.get("colors")
        if isinstance(colors, list) and len(colors) == len(self.mappers):
            for mapper, bounds in zip(self.mappers, colors, strict=True):
                if (
                    isinstance(bounds, list)
                    and len(bounds) == 2
                    and all(type(v) in (int, float) and np.isfinite(v) for v in bounds)
                    and bounds[0] < bounds[1]
                ):
                    mapper.low, mapper.high = bounds

    @property
    def monthly_profiles(self):
        return (
            self.mode.value == "heat"
            and "temperature"
            in self.catalog.meta["records"][self.origin.value]["models"][
                self.model.value
            ]
        )

    @property
    def index(self):
        assert self.frame.value is not None
        if self.mode.value == "heat":
            return self.month.value
        return (
            self.frame.value // 5 - 1
            if self.mode.value == "surface"
            else self.depth.value
        )

    def configure_view(self, reset_variable=False):
        """Keep the model while adapting controls to its saved fields and grid."""
        previous_origin = self.origin.value
        origin_date = self.interior_examples.get(previous_origin, {}).get(
            "origin", previous_origin
        )
        self.catalog = self.registry.for_model(self.model.value)
        self.interior_examples = self.catalog.meta.get("interior_examples", {})
        available = self.catalog.available_modes(self.model.value)
        modes = {label: value for label, value in MODES.items() if value in available}
        mode = self.mode.value
        self.mode.options = modes
        self.availability_message = ""
        if mode not in available:
            old_label = next(label for label, value in MODES.items() if value == mode)
            mode = "surface" if "surface" in available else next(iter(modes.values()))
            self.availability_message = f"{old_label} is not saved for this model; displaying {next(label for label, value in MODES.items() if value == mode).lower()}. "
        self.mode.value = mode
        interior, forecast, heat = (
            mode == "interior",
            mode == "forecast_interior",
            mode == "heat",
        )
        saved_origins = self.catalog.origins_for(self.model.value, mode)
        origins = {
            (
                self.interior_examples[key]["label"]
                if interior and self.interior_examples
                else self.catalog.meta["records"][key].get(
                    "label", f"{key} (obs)" if interior else key
                )
            ): key
            for key in saved_origins
        }
        self.origin.options = origins
        self.origin.value = (
            previous_origin
            if previous_origin in origins.values()
            else origin_date
            if origin_date in origins.values()
            else next(iter(origins.values()))
        )
        variables = (
            {
                **VARIABLES["interior"],
                "Zonal velocity (U)": 2,
                "Meridional velocity (V)": 3,
            }
            if interior and self.interior_examples
            else VARIABLES[mode]
        )
        variable = self.variable.value
        self.variable.options = variables
        self.variable.value = (
            0 if reset_variable or variable not in variables.values() else variable
        )
        self.variable.name = "Layer" if heat else "Variable"
        depth = self.depth.value
        self.depth.options = {
            f"{value:g} m": i for i, value in enumerate(self.catalog.depths)
        }
        self.depth.value = (
            depth
            if depth in self.depth.options.values()
            else min(9, len(self.catalog.depths) - 1)
        )
        leads = self.catalog.meta.get(
            "forecast_interior_leads_days", [5, 15, 30, 90, 180, 365]
        )
        saved_lead = self.saved_lead.value
        self.saved_lead.options = leads
        self.saved_lead.value = saved_lead if saved_lead in leads else leads[0]
        self.frame.visible = self.player.visible = mode == "surface"
        self.saved_lead.visible = forecast
        self.detail.visible = not interior
        self.month.visible = heat
        self.depth_marker.visible = self.depth.visible = interior or forecast
        maximum_depth = float(self.catalog.depth_interfaces[-1])
        self.profile.y_range.start = maximum_depth
        for section in self.sections:
            section.y_range.start = maximum_depth
        self.configure_comparisons()

    def change_mode(self, event):
        if self._changing:
            return
        self._changing = True
        try:
            self.player.direction = 0
            self.configure_view(reset_variable=True)
        finally:
            self._changing = False
        self.refresh(rescale=True)

    def configure_comparisons(self, reset=False):
        mode = self.mode.value
        interior, forecast, heat = (
            mode == "interior",
            mode == "forecast_interior",
            mode == "heat",
        )
        record = self.interior_examples.get(self.origin.value) if interior else None
        compatible = self.catalog.models_for(mode, self.origin.value)
        models = {
            v["label"]: key
            for key, v in self.catalog.meta["models"].items()
            if key in compatible
        }
        # OM4-input examples historically have only their saved mixed endpoint.
        choices = (
            models
            if record is not None and record.get("source") == "om4"
            else self.model_options
        )
        model = self.model.value
        self.model.options = choices
        if model not in compatible:
            self.model.value = next(iter(models.values()))
        elif self.model.value != model:
            self.model.value = model
        options = {}
        data_record = self.catalog.meta["records"].get(self.origin.value, {})
        if interior:
            if record is not None:
                if "observations" in record["references"] and self.variable.value < 2:
                    options["IAP monthly context"] = "observations"
                    if "climatology" in record["references"]:
                        options["Training December climatology"] = "climatology"
                if "om4" in record["references"]:
                    options[
                        "OM4 gold"
                        if record["source"] == "om4"
                        else "OM4 contemporaneous context"
                    ] = "om4"
            else:
                options = {
                    "IAP monthly context": "observations",
                    "Training December climatology": "climatology",
                }
        else:
            if f"{mode}_reference" in data_record:
                truth = (
                    ("OM4 monthly heat content" if heat else "OM4 gold")
                    if data_record.get("source") == "om4"
                    else ("IAP monthly heat content" if heat else "Observations")
                )
                options[truth] = "observations"
            available = data_record["models"][self.model.value]
            if forecast or mode + "_persistence" in available:
                options["Own initialized persistence"] = "persistence"
        # Prefer another saved model when no observational reference exists.
        options.update(
            {label: key for label, key in models.items() if key != self.model.value}
        )
        options.update(
            {label: key for label, key in models.items() if key == self.model.value}
        )
        reference = self.reference.value
        self.reference.options = options
        self.reference.value = (
            next(iter(options.values()))
            if reset or reference not in options.values()
            else reference
        )

    def change_sample(self, event):
        if not self._changing:
            self._changing = True
            try:
                self.configure_comparisons()
            finally:
                self._changing = False
            self.refresh(rescale=True)

    def select(self, event):
        if self._changing:
            return
        self._changing = True
        try:
            if event.obj is self.model:
                self.player.direction = 0
                self.configure_view()
            else:
                self.configure_comparisons()
        finally:
            self._changing = False
        self.refresh(rescale=True)

    def step(self, event):
        if self._changing:
            return
        self._changing = True
        self.player.value = event.new // 5 - 1
        self._changing = False
        self.refresh(rescale=not self.lock_colors.value)

    def play(self, event):
        if not self._changing:
            self.frame.value = (event.new + 1) * 5

    def change_depth(self, event):
        if not self._changing:
            self.refresh(rescale=not self.lock_colors.value)

    def tap(self, event):
        if event.x is None or event.y is None:
            return
        self._changing = True
        self.longitude.value = float(np.clip(event.x, 0, 360))
        self.latitude.value = float(np.clip(event.y, -90, 90))
        self._changing = False
        self.update_point()

    def change_point(self, event):
        if not self._changing:
            self.update_point()

    def set_geometry(self):
        lat, lon = self.catalog.geometry(self.mode.value)
        signature = (lat.tobytes(), lon.tobytes())
        if signature == getattr(self, "geometry_signature", None):
            return
        self.geometry_signature = signature
        self.lat, self.lon = lat, lon
        self.wet = None
        self.use_raster = len(lat) * len(lon) > 100_000
        if self.use_raster:
            self.geometry = {}
        else:
            le, la = edges(lon, (0, 360)), edges(lat, (-90, 90))
            left, bottom = np.meshgrid(le[:-1], la[:-1])
            right, top = np.meshgrid(le[1:], la[1:])
            longitude, latitude = np.meshgrid(lon, lat)
            self.geometry = dict(
                left=left.ravel(),
                right=right.ravel(),
                bottom=bottom.ravel(),
                top=top.ravel(),
                lon=longitude.ravel(),
                lat=latitude.ravel(),
            )
        for quad, raster, source, raster_source in zip(
            self.quad_renderers,
            self.raster_renderers,
            self.sources,
            self.raster_sources,
            strict=True,
        ):
            quad.visible, raster.visible = not self.use_raster, self.use_raster
            if self.use_raster:
                source.data = {
                    key: []
                    for key in ("left", "right", "bottom", "top", "lon", "lat", "value")
                }
            else:
                raster_source.data = dict(image=[], values=[], x=[], y=[], dw=[], dh=[])

    def refresh(self, rescale=False):
        self.set_geometry()
        if self.mode.value == "heat":
            options = {
                month: index
                for index, month in enumerate(
                    self.catalog.meta["records"][self.origin.value]["heat_months"]
                )
            }
            self._changing = True
            selected = self.month.value
            self.month.options = options
            self.month.value = selected if selected in options.values() else 0
            self._changing = False
        mode, variable, index = self.mode.value, self.variable.value, self.index
        self.profile_pane.visible = (
            mode in ("interior", "forecast_interior") or self.monthly_profiles
        )
        self.orientation.visible = self.section_row.visible = self.profile_pane.visible
        self.prediction, self.comparison = self.catalog.values(
            mode, self.model.value, self.origin.value, self.reference.value
        )
        if mode == "forecast_interior":
            self.forecast_prediction, self.forecast_comparison = (
                self.prediction,
                self.comparison,
            )
            lead_index = self.catalog.meta["forecast_interior_leads_days"].index(
                self.saved_lead.value
            )
            self.prediction, self.comparison = (
                self.prediction[lead_index],
                self.comparison[lead_index],
            )
        a, b = self.prediction[index, variable], self.comparison[index, variable]
        difference = a - b
        self.unit = UNITS[mode][variable]
        if rescale:
            low, high = limits(a, b)
            for mapper in self.mappers[:2]:
                mapper.low, mapper.high = low, high
            self.mappers[2].low, self.mappers[2].high = limits(
                difference, symmetric=True
            )
        mask_mode = "interior" if mode == "forecast_interior" else mode
        mask = self.catalog.array(self.catalog.meta[f"{mask_mode}_mask"])
        wet = (
            mask[index, variable]
            if mode in ("interior", "forecast_interior")
            else mask[variable]
        )
        geometry_changed = self.wet is None or not np.array_equal(wet, self.wet)
        self.wet = wet
        if self.use_raster:
            for source, values, mapper in zip(
                self.raster_sources, [a, b, difference], self.mappers, strict=True
            ):
                source.data = raster_rows(
                    values,
                    wet,
                    self.lat,
                    self.lon,
                    mapper.palette,
                    mapper.low,
                    mapper.high,
                )
        else:
            coords = (
                {key: values[self.wet.ravel()] for key, values in self.geometry.items()}
                if geometry_changed
                else None
            )
            for source, values in zip(self.sources, [a, b, difference], strict=True):
                if coords is not None:
                    source.data = {**coords, "value": values[self.wet]}
                else:
                    source.patch(
                        {"value": [(slice(0, int(self.wet.sum())), values[self.wet])]}
                    )
        reference_label = next(
            k for k, v in self.reference.options.items() if v == self.reference.value
        )

        def short_label(model):
            explicit = self.catalog.meta["models"][model].get("short_label")
            if explicit:
                return explicit
            counts = self.catalog.meta["models"][model]["lineage"]["task_counts"]
            return f"{'Mixed' if counts['om4'] else 'Obs only'} · {counts['observation']:,} obs"

        self.maps[0].title.text = short_label(self.model.value)
        self.maps[1].title.text = (
            short_label(self.reference.value)
            if self.reference.value in self.catalog.meta["models"]
            else reference_label
        )
        self.maps[2].title.text = "Model − comparison"
        for plot in self.maps:
            plot.below[-1].title = self.unit
        variable_label = next(
            k for k, v in self.variable.options.items() if v == variable
        )
        source_record = self.catalog.meta["records"].get(self.origin.value, {})
        origin_date = source_record.get("origin", self.origin.value)
        is_om4 = source_record.get("source") == "om4"
        if mode == "surface":
            self.caption.object = (
                f"### {variable_label} · {interval(origin_date, index)}"
            )
            self.context.object = (
                (
                    "Five-day OM4 means on the model grid, with OM4 surface-history inputs and OM4's own prescribed forcing. Reference fields are OM4 simulation fields, not observations. "
                    if is_om4
                    else "Five-day means on the model grid. Observations: OISST temperature / DUACS sea level. "
                )
                + "\n\n**Explore:** Click a map to inspect a cell; pan and zoom are linked across all three maps."
            )
        elif mode == "heat":
            month = self.catalog.meta["records"][self.origin.value]["heat_months"][
                index
            ]
            self.caption.object = f"### Ocean heat content · {variable_label} · {month}"
            truth = "OM4 simulation fields" if is_om4 else "IAP analyses"
            self.context.object = f"Monthly layer-integrated heat content compared with {truth}, in GJ/m² relative to 0 °C. Gray cells lack a complete model column for the selected layer.\n\n**Explore:** Click a map for the monthly series."
            if self.monthly_profiles:
                self.context.object += " Move the month slider to explore temperature sections and the selected-cell profile. Use Along longitude / Along latitude to choose the section direction; click a map to move the slice."
            if self.monthly_profiles:
                truth = "OM4" if is_om4 else "IAP"
                self.context.object += f"\n\n**Fields and comparison:** Heat maps and profiles use the same calendar-month forecast means. Profiles show {truth} monthly temperatures at model depths."
            else:
                self.context.object += "\n\n**Available fields:** Heat maps use the native IAP half-degree comparison grid. Monthly temperature profiles were not saved; use Saved forecast interior for the available five-day states."
        elif mode == "forecast_interior":
            self.caption.object = f"### {variable_label} · {self.catalog.depths[index]:g} m · {interval(origin_date, self.saved_lead.value // 5 - 1)}"
            self.context.object = "Saved full-state snapshots at days 5, 15, 30, 90, 180 and 365. These are five-day means, not monthly temperature profiles. Compare against the other native model or this checkpoint's initialized persistence; no full-depth observational truth is implied."
        else:
            record = self.interior_examples.get(self.origin.value)
            label = record["label"] if record else f"{self.origin.value} (obs)"
            self.caption.object = f"### {variable_label} · {self.catalog.depths[index]:g} m · initialization {label}"
            if record:
                # Older catalogs store the task explanation as one long paragraph.
                paragraphs: list[str] = []
                for paragraph in record["description"].split("\n\n"):
                    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
                    paragraphs.extend(
                        " ".join(sentences[i : i + 3])
                        for i in range(0, len(sentences), 3)
                    )
                self.context.object = "\n\n".join(paragraphs)
            else:
                self.context.object = f"Initialized state from the last five-day history interval. IAP {self.catalog.meta['records'][self.origin.value]['context_month']} is monthly context, not instantaneous truth. Surface temperature at 2.5 m is copied from available inputs."
        readout_note = source_record.get(f"{mode}_readout_description")
        if readout_note:
            self.context.object += "\n\n**Decoder readout:** " + readout_note
        stats = paired_stats(a, b, self.lat)
        if stats["cells"]:
            qualifier = (
                "Context difference"
                if mode == "interior"
                and (
                    self.reference.value == "observations"
                    or (
                        self.reference.value == "om4"
                        and self.interior_examples[self.origin.value]["source"] == "obs"
                    )
                )
                else "Snapshot difference"
            )
            self.summary.object = f"**{qualifier}:** RMS {stats['rmse']:.4g} {self.unit} · bias {stats['bias']:+.4g} {self.unit} · {stats['cells']:,} paired cells. Cosine-latitude weighted; exploratory diagnostics."
        else:
            self.summary.object = "No common finite cells in this slice."
        info = self.catalog.meta["models"][self.model.value]
        lineage = info["lineage"]
        self.model_description.object = description(self.model.value, info)
        report_url = info.get("report_url", self.catalog.meta["report_url"])
        self.provenance.object = (
            f"[Source report]({report_url}) · **{self.catalog.meta['models'][self.model.value]['label']}**\n\n"
            f"Origin: `{self.origin.value}` · Checkpoint SHA-256: `{lineage['checkpoint_sha256']}`\n\n"
            "Gray = model land/depth mask. White = unavailable values in model ocean cells. "
            "Model and comparison share a color scale; difference is centered on zero. Limits use the pooled 2nd–98th percentiles and may clip extremes. "
            "The prepared catalog retains original source paths, hashes, and report receipts."
        )
        if (
            info.get("family") == "diffusion"
            or "diffusion" in info.get("description", "").lower()
        ):
            self.context.object = self.context.object.replace(
                "Surface temperature at 2.5 m is copied from available inputs.",
                "Diffusion fields are decoded predictions, not copied surface inputs.",
            )
        task_label = next(label for label, value in MODES.items() if value == mode)
        availability = (
            f"**Availability:** {self.availability_message}\n\n"
            if self.availability_message
            else ""
        )
        self.context.object = (
            f"### What is this task?\n\n**{task_label}**\n\n"
            + availability
            + self.context.object
        )
        self.update_point()

    def update_point(self):
        variable = self.variable.value
        y = int(np.abs(self.lat - self.latitude.value).argmin())
        longitude_distance = np.abs((self.lon - self.longitude.value + 180) % 360 - 180)
        x = int(longitude_distance.argmin())
        lat, lon = self.lat[y], self.lon[x]
        for horizontal, vertical in self.crosshairs:
            horizontal.location, vertical.location = float(lat), float(lon)
        a, b = self.prediction[:, variable, y, x], self.comparison[:, variable, y, x]

        def number(value):
            return f"{value:.4g} {self.unit}" if np.isfinite(value) else "unavailable"

        self.location.object = f"**Selected cell:** {lat:.2f}°N, {lon:.2f}°E · model {number(a[self.index])} · comparison {number(b[self.index])}"
        if self.mode.value not in ("interior", "forecast_interior"):
            heat = self.mode.value == "heat"
            self.line_source.data = dict(
                x=np.arange(1, 13) if heat else np.arange(5, 366, 5),
                prediction=a,
                reference=b,
            )
            self.line.xaxis.axis_label = (
                f"Month of {self.origin.value[:4]} (monthly means)"
                if heat
                else "Forecast lead (days; five-day means)"
            )
            self.line.yaxis.axis_label = self.unit
            self.line.title.text = f"{'Monthly heat content' if heat else 'Time series'} · {lat:.2f}°N, {lon:.2f}°E"
            self.line_hover.tooltips = [
                ("Month" if heat else "Forecast lead (days)", "@x"),
                ("Model", "@prediction{0.0000}"),
                ("Comparison", "@reference{0.0000}"),
            ]
            self.marker.location = self.index + 1 if heat else self.frame.value
            if self.monthly_profiles:
                prediction, comparison = self.catalog.values(
                    "temperature",
                    self.model.value,
                    self.origin.value,
                    self.reference.value,
                )
                self.profile_source.data = dict(
                    depth=self.catalog.depths,
                    prediction=prediction[self.index, :, y, x],
                    reference=comparison[self.index, :, y, x],
                )
                month = self.catalog.meta["records"][self.origin.value]["heat_months"][
                    self.index
                ]
                self.update_sections(
                    prediction[self.index],
                    comparison[self.index],
                    self.catalog.array(self.catalog.meta["interior_mask"])[:, 0],
                    y,
                    x,
                    "°C",
                    period=month,
                )
                self.profile.xaxis.axis_label = "Temperature (°C)"
                self.profile.title.text = (
                    f"Monthly temperature · {month} · {lat:.2f}°N, {lon:.2f}°E"
                )
        else:
            if self.mode.value == "forecast_interior":
                self.line_source.data = dict(
                    x=self.catalog.meta["forecast_interior_leads_days"],
                    prediction=self.forecast_prediction[
                        :, self.depth.value, variable, y, x
                    ],
                    reference=self.forecast_comparison[
                        :, self.depth.value, variable, y, x
                    ],
                )
                self.line.xaxis.axis_label = (
                    "Saved forecast lead (days; five-day means)"
                )
                self.line.yaxis.axis_label = self.unit
                self.line.title.text = f"Saved states · {self.catalog.depths[self.depth.value]:g} m · {lat:.2f}°N, {lon:.2f}°E"
                self.marker.location = self.saved_lead.value
                self.line_hover.tooltips = [
                    ("Saved forecast lead (days)", "@x"),
                    ("Model", "@prediction{0.0000}"),
                    ("Comparison", "@reference{0.0000}"),
                ]
            self.profile_source.data = dict(
                depth=self.catalog.depths, prediction=a, reference=b
            )
            self.profile.xaxis.axis_label = self.unit
            self.profile.title.text = f"Vertical profile · {lat:.2f}°N, {lon:.2f}°E"
            self.depth_marker.location = float(self.catalog.depths[self.depth.value])
            self.update_sections(
                self.prediction[:, variable],
                self.comparison[:, variable],
                self.catalog.array(self.catalog.meta["interior_mask"])[:, variable],
                y,
                x,
                self.unit,
            )

        self.sync_url()

    def update_sections(self, prediction, comparison, wet, y, x, unit, period=""):
        """Render a depth/longitude or depth/latitude slice without spatial averaging."""
        lat, lon = self.lat[y], self.lon[x]
        along_lon = self.orientation.value == "Along longitude"
        coordinate = self.lon if along_lon else self.lat
        bounds = (0, 360) if along_lon else (-90, 90)
        ce = edges(coordinate, bounds)
        de = self.catalog.depth_interfaces
        left, top = np.meshgrid(ce[:-1], de[:-1])
        right, bottom = np.meshgrid(ce[1:], de[1:])
        coord, depth = np.meshgrid(coordinate, self.catalog.depths)
        take = (lambda v: v[:, y, :]) if along_lon else (lambda v: v[:, :, x])
        support = take(wet)
        first, second = (
            take(prediction),
            take(comparison),
        )
        low, high = limits(first, second)
        for mapper in self.section_mappers[:2]:
            mapper.low, mapper.high = low, high
        self.section_mappers[2].low, self.section_mappers[2].high = limits(
            first - second, symmetric=True
        )
        for i, (plot, source, values) in enumerate(
            zip(
                self.sections,
                self.section_sources,
                [first, second, first - second],
                strict=True,
            )
        ):
            source.data = dict(
                left=left[support],
                right=right[support],
                top=top[support],
                bottom=bottom[support],
                coordinate=coord[support],
                depth=depth[support],
                value=values[support],
            )
            plot.xaxis.axis_label = "Longitude (°E)" if along_lon else "Latitude (°N)"
            plot.below[-1].title = unit
            plot.x_range.start, plot.x_range.end = bounds
            label = ["Model", "Comparison", "Difference"][i]
            label += f" · T · {period}" if period else ""
            plot.title.text = f"{label} · {'latitude' if along_lon else 'longitude'} {lat if along_lon else lon:.2f}°"

    def view(self):
        sidebar = [
            self.mode,
            self.model,
            self.origin,
            self.variable,
            self.reference,
            pn.layout.Divider(),
            self.frame,
            self.player,
            self.saved_lead,
            self.depth,
            self.month,
            self.lock_colors,
            self.rescale,
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Inspect a location**\n\nClick any map or enter coordinates."
            ),
            self.latitude,
            self.longitude,
            self.orientation,
        ]
        return pn.template.FastListTemplate(
            title="Samudra Results Viewer",
            accent_base_color="#087f8c",
            header_background="#143b4a",
            sidebar=[
                pn.Column(
                    *sidebar,
                    pn.pane.Markdown(
                        "[A: initialization & diffusion Xray](?xray=1)"
                    ),
                )
            ],
            sidebar_width=285,
            main=[
                pn.Column(
                    self.caption,
                    self.map_row,
                    self.summary,
                    self.location,
                    self.section_row,
                    self.detail,
                    self.profile_pane,
                    pn.Accordion(
                        ("Data, masks & provenance", self.provenance), active=[]
                    ),
                    self.context,
                    self.model_description,
                )
            ],
            main_layout=None,
        )


if pn.state.session_args.get("xray", [b"0"])[0] == b"1":
    from step_anatomy import Anatomy

    Anatomy(
        os.environ.get("SAMUDRA_ANATOMY_DATA", str(APP_ROOT / ".step-anatomy"))
    ).view().servable()
else:
    catalog = Catalog(
        os.environ.get("SAMUDRA_VIEWER_DATA", str(APP_ROOT / ".data")),
        extra_catalogs=[
            path
            for path in os.environ.get("SAMUDRA_VIEWER_CATALOGS", "").split(os.pathsep)
            if path
        ],
    )
    viewer = Viewer(catalog)
    viewer.view().servable()
