# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Serve with panel serve app.py; each browser session owns its controls."""

import os
import sys
from pathlib import Path

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
from data import Catalog, edges, interval, limits, paired_stats

pn.extension(sizing_mode="stretch_width")
APP_ROOT = Path(__file__).resolve().parent
DIFFERENCE_PALETTE = list(RdBu[11])
MODES = {
    "Annual surface forecasts": "surface",
    "Initialized ocean interior": "interior",
}
VARIABLES = {
    "surface": {"Sea surface temperature": 0, "Sea surface height": 1},
    "interior": {"Temperature": 0, "Salinity": 1},
}
UNITS = {"surface": ["°C", "m"], "interior": ["°C", "psu"]}


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
        self.catalog = catalog
        self._changing = False
        self.wet = None
        self.model_options = {v["label"]: k for k, v in catalog.meta["models"].items()}
        self.mode = pn.widgets.Select(
            name="View", options=MODES, value="surface", css_classes=["view-select"]
        )
        self.model = pn.widgets.Select(
            name="Model / checkpoint",
            options=self.model_options,
            value="obs08000",
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
        self.context = pn.pane.Alert("", alert_type="info")
        self.provenance = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.maps, self.sources, self.mappers, self.crosshairs = [], [], [], []
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
        self.line.add_tools(
            HoverTool(
                tooltips=[
                    ("Lead / depth", "@x"),
                    ("Model", "@prediction{0.0000}"),
                    ("Comparison", "@reference{0.0000}"),
                ],
                mode="vline",
            )
        )
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
        self.map_row = pn.Row(
            *(pn.pane.Bokeh(p, sizing_mode="stretch_width") for p in self.maps)
        )
        self.section_row = pn.Row(
            *(pn.pane.Bokeh(p, sizing_mode="stretch_width") for p in self.sections),
            visible=False,
        )
        self.detail = pn.Column(pn.pane.Bokeh(self.line, sizing_mode="stretch_width"))
        self.profile_pane = pn.Column(
            pn.pane.Bokeh(self.profile, sizing_mode="stretch_width"), visible=False
        )
        for control in [self.model, self.origin, self.variable, self.reference]:
            control.param.watch(self.select, "value")
        self.mode.param.watch(self.change_mode, "value")
        self.frame.param.watch(self.step, "value")
        self.player.param.watch(self.play, "value")
        self.depth.param.watch(self.change_depth, "value")
        for point_control in [self.latitude, self.longitude, self.orientation]:
            point_control.param.watch(self.change_point, "value")
        self.rescale.on_click(lambda event: self.refresh(rescale=True))
        self.refresh(rescale=True)

    @property
    def index(self):
        assert self.frame.value is not None
        return (
            self.frame.value // 5 - 1
            if self.mode.value == "surface"
            else self.depth.value
        )

    def change_mode(self, event):
        self._changing = True
        self.player.direction = 0
        interior = event.new == "interior"
        self.variable.options = VARIABLES[event.new]
        self.variable.value = 0
        options = {
            "IAP monthly context" if interior else "Observations": "observations"
        }
        if interior:
            options["Training December climatology"] = "climatology"
        options.update(self.model_options)
        self.reference.options = options
        self.reference.value = "observations"
        for control in [self.frame, self.player, self.detail]:
            control.visible = not interior
        for control in [
            self.depth,
            self.orientation,
            self.profile_pane,
            self.section_row,
        ]:
            control.visible = interior
        self._changing = False
        self.refresh(rescale=True)

    def select(self, event):
        if not self._changing:
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

    def refresh(self, rescale=False):
        mode, variable, index = self.mode.value, self.variable.value, self.index
        self.prediction, self.comparison = self.catalog.values(
            mode, self.model.value, self.origin.value, self.reference.value
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
        mask = self.catalog.array(self.catalog.meta[f"{mode}_mask"])
        wet = mask[variable] if mode == "surface" else mask[index, variable]
        geometry_changed = self.wet is None or not np.array_equal(wet, self.wet)
        self.wet = wet
        coords = (
            {k: v[self.wet.ravel()] for k, v in self.geometry.items()}
            if geometry_changed
            else None
        )
        for source, values in zip(self.sources, [a, b, difference], strict=True):
            if geometry_changed:
                assert coords is not None
                source.data = {**coords, "value": values[self.wet]}
            else:
                source.patch(
                    {"value": [(slice(0, int(self.wet.sum())), values[self.wet])]}
                )
        reference_label = next(
            k for k, v in self.reference.options.items() if v == self.reference.value
        )

        def short_label(model):
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
        if mode == "surface":
            self.caption.object = (
                f"### {variable_label} · {interval(self.origin.value, index)}"
            )
            self.context.object = "Five-day means on the model grid. Observations: OISST temperature / DUACS sea level. Click a map to inspect a cell; pan and zoom are linked across all three maps."
        else:
            self.caption.object = f"### {variable_label} · {self.catalog.depths[index]:g} m · initialization {self.origin.value}"
            self.context.object = f"Initialized state from the last five-day history interval. IAP {self.catalog.meta['records'][self.origin.value]['context_month']} is monthly context, not instantaneous truth. Surface temperature at 2.5 m is copied from available inputs."
        stats = paired_stats(a, b, self.catalog.lat)
        if stats["cells"]:
            qualifier = (
                "Context difference"
                if mode == "interior" and self.reference.value == "observations"
                else "Snapshot difference"
            )
            self.summary.object = f"**{qualifier}:** RMS {stats['rmse']:.4g} {self.unit} · bias {stats['bias']:+.4g} {self.unit} · {stats['cells']:,} paired cells. Cosine-latitude weighted; exploratory diagnostics."
        else:
            self.summary.object = "No common finite cells in this slice."
        lineage = self.catalog.meta["models"][self.model.value]["lineage"]
        self.provenance.object = (
            f"[Source report]({self.catalog.meta['report_url']}) · **{self.catalog.meta['models'][self.model.value]['label']}**\n\n"
            f"Origin: `{self.origin.value}` · Checkpoint SHA-256: `{lineage['checkpoint_sha256']}`\n\n"
            "Gray = model land/depth mask. White = unavailable values in model ocean cells. "
            "Model and comparison share a color scale; difference is centered on zero. Limits use the pooled 2nd–98th percentiles and may clip extremes. "
            "The prepared catalog retains original source paths, hashes, and report receipts."
        )
        self.update_point()

    def update_point(self):
        variable = self.variable.value
        y = int(np.abs(self.catalog.lat - self.latitude.value).argmin())
        longitude_distance = np.abs(
            (self.catalog.lon - self.longitude.value + 180) % 360 - 180
        )
        x = int(longitude_distance.argmin())
        lat, lon = self.catalog.lat[y], self.catalog.lon[x]
        for horizontal, vertical in self.crosshairs:
            horizontal.location, vertical.location = float(lat), float(lon)
        a, b = self.prediction[:, variable, y, x], self.comparison[:, variable, y, x]

        def number(value):
            return f"{value:.4g} {self.unit}" if np.isfinite(value) else "unavailable"

        self.location.object = f"**Selected cell:** {lat:.2f}°N, {lon:.2f}°E · model {number(a[self.index])} · comparison {number(b[self.index])}"
        if self.mode.value == "surface":
            self.line_source.data = dict(
                x=np.arange(5, 366, 5), prediction=a, reference=b
            )
            self.line.xaxis.axis_label = "Forecast lead (days; five-day means)"
            self.line.yaxis.axis_label = self.unit
            self.line.title.text = f"Time series · {lat:.2f}°N, {lon:.2f}°E"
            self.marker.location = self.frame.value
        else:
            self.profile_source.data = dict(
                depth=self.catalog.depths, prediction=a, reference=b
            )
            self.profile.xaxis.axis_label = self.unit
            self.profile.title.text = f"Vertical profile · {lat:.2f}°N, {lon:.2f}°E"
            self.depth_marker.location = float(self.catalog.depths[self.depth.value])
            along_lon = self.orientation.value == "Along longitude"
            coordinate = self.catalog.lon if along_lon else self.catalog.lat
            bounds = (0, 360) if along_lon else (-90, 90)
            ce = edges(coordinate, bounds)
            # The report uses these OM4 layer centers and corresponding interfaces.
            de = np.array(
                [0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650, 900, 1200, 1600, 2100]
            )
            np.testing.assert_array_equal((de[:-1] + de[1:]) / 2, self.catalog.depths)
            left, top = np.meshgrid(ce[:-1], de[:-1])
            right, bottom = np.meshgrid(ce[1:], de[1:])
            coord, depth = np.meshgrid(coordinate, self.catalog.depths)
            wet = self.catalog.array(self.catalog.meta["interior_mask"])[:, variable]
            take = (lambda v: v[:, y, :]) if along_lon else (lambda v: v[:, :, x])
            support = take(wet)
            first, second = (
                take(self.prediction[:, variable]),
                take(self.comparison[:, variable]),
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
                plot.xaxis.axis_label = (
                    "Longitude (°E)" if along_lon else "Latitude (°N)"
                )
                plot.below[-1].title = self.unit
                plot.x_range.start, plot.x_range.end = bounds
                label = ["Model", "Comparison", "Difference"][i]
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
            self.depth,
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
            title="Samudra · Ocean results",
            accent_base_color="#087f8c",
            header_background="#143b4a",
            sidebar=[pn.Column(*sidebar)],
            sidebar_width=285,
            main=[
                pn.Column(
                    self.caption,
                    self.context,
                    self.map_row,
                    self.summary,
                    self.location,
                    self.detail,
                    self.section_row,
                    self.profile_pane,
                    pn.Accordion(
                        ("Data, masks & provenance", self.provenance), active=[]
                    ),
                )
            ],
            main_layout=None,
        )


catalog = Catalog(os.environ.get("SAMUDRA_VIEWER_DATA", str(APP_ROOT / ".data")))
viewer = Viewer(catalog)
viewer.view().servable()
