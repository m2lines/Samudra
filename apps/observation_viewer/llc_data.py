# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Six-hour snapshot catalog with explicit reference provenance."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

VARIABLES = {
    "Temperature (°C)": "thetao",
    "Salinity (psu)": "so",
    "Eastward velocity (m/s)": "uo",
    "Northward velocity (m/s)": "vo",
    "SSH (m)": "zos",
}


def llc_label(exact, source):
    if exact:
        return (
            "LLC matched reference"
            if source == "llc"
            else "LLC contemporaneous context"
        )
    return "LLC calendar-date match (different year)"


class LLCCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root / "catalog.json").read_text())
        if self.meta["schema"] != "llc-six-hour-viewer-v1":
            raise ValueError("Unsupported LLC catalog")
        self.models = self.meta["models"]
        grid = np.load(self.path(self.meta["grid"]), allow_pickle=False)
        self.lat, self.lon, self.depth = grid["lat"], grid["lon"], grid["depth"]
        self.names, self.wet = grid["names"].tolist(), grid["wet_area"] > 0
        self.interfaces = np.r_[0, np.cumsum(grid["thickness"])]
        if not (np.all(np.diff(self.depth) > 0) and len(self.depth) == 51):
            raise ValueError("Expected 51 native LLC depth levels")

    def path(self, relative):
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Catalog path escapes bundle")
        return result

    @lru_cache(maxsize=32)
    def array(self, relative):
        return np.load(self.path(relative), mmap_mode="r", allow_pickle=False)

    def case(self, model, case):
        return self.models[model]["cases"][case]

    def channel(self, variable, depth):
        return self.names.index("zos" if variable == "zos" else f"{variable}_{depth}")

    def field(self, model, case, view, frame, variable, depth):
        record = self.case(model, case)
        channel = self.channel(variable, depth)
        if view in ("surface", "history"):
            if variable not in ("thetao", "zos") or (
                variable == "thetao" and depth != 0
            ):
                raise ValueError("Surface archive only contains SST and SSH")
            channel = 0 if variable == "thetao" else 1
        key = record["history"] if view == "history" else record["prediction"][view]
        return self.array(key)[frame, channel]

    def reference(self, model, case, view, frame, variable, depth, choice="auto"):
        record = self.case(model, case)
        channel = self.channel(variable, depth)
        empty = np.full(self.wet.shape[-2:], np.nan, dtype="f4")
        surface_channel = (
            0
            if variable == "thetao" and depth == 0
            else (1 if variable == "zos" else None)
        )
        if view == "history":
            return empty, "Input history: available SST/SSH slots only", False
        matched_llc = (
            record["source"] == "llc" and record["llc_" + view + "_exact"][frame]
        )
        use_observations = choice == "observations" or (
            choice == "auto" and not matched_llc
        )
        if use_observations and surface_channel is not None:
            if view == "initial":
                observed = self.array(record["observed_initial"])[
                    frame, surface_channel
                ]
            else:
                index = (
                    frame
                    if view == "surface"
                    else record["saved_hours"][frame] // 6 - 1
                )
                observed = self.array(record["observed_surface"])[
                    index, surface_channel
                ]
            if np.isfinite(observed).any():
                return (
                    observed,
                    "Observations · " + self.timestamp(record, view, frame),
                    True,
                )
        if choice == "observations":
            return empty, "No observational reference at this field/time", False
        index = surface_channel if view == "surface" else channel
        values = self.array(record["llc_" + view])[frame, index]
        actual = record["llc_" + view + "_times"][frame]
        exact = record["llc_" + view + "_exact"][frame]
        gold = exact and record["source"] == "llc"
        label = llc_label(exact, record["source"])
        return values, label + " · actual " + actual, gold

    @staticmethod
    def timestamp(record, view, frame):
        key = {
            "initial": "initial_times",
            "interior": "saved_times",
            "surface": "times",
            "history": "history_times",
        }[view]
        return record[key][frame]
