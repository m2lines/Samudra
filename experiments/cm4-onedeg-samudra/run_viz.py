# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the existing viz pipeline with native CM4 dates and physical fields."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cartopy  # type: ignore[import-untyped]
import dask
import numpy as np
from PIL import Image

from samudra.config import Cm4TimeConfig
from samudra.constants import build_om4_layout
from samudra.utils.location import LocalLocation
from samudra.utils.logging import handle_logging
from samudra.viz.config import VizConfig, run_steps
from samudra.viz.core import Viz, VizRun, prepare_viz_groundtruth


def main(cfg: VizConfig) -> None:
    started = time.perf_counter()
    output = cfg.output_path
    output.mkdir(parents=True, exist_ok=True)
    handle_logging(cfg.debug, output)
    cfg.save_yaml(output / "config.yaml")
    cartopy.config["data_dir"] = str(output.parent / "cartopy")
    dask.config.set(scheduler="threads", num_workers=8)
    assert cfg.data_root is not None and cfg.groundtruth_location is not None
    root = LocalLocation(path=Path.cwd()).resolve(cfg.data_root)
    source = root.resolve(cfg.groundtruth_location).open(chunks={})
    assert source.time.values[0].calendar == "noleap"
    # VizConfig currently parses OM4 Julian dates. Rebuild the date-only bounds
    # with CM4's midnight/no-leap calendar before selecting any source values.
    dates = Cm4TimeConfig.model_validate(
        {
            "start": str(cfg.groundtruth_time_range.start),
            "end": str(cfg.groundtruth_time_range.end),
        }
    )
    reference = source.sel(time=dates.time_slice)
    assert reference.sizes["time"] == 7298
    fields = ["thetao", "so", "uo", "vo", "zos"]
    layout = build_om4_layout(grid_type="gaussian")
    np.testing.assert_array_equal(source.lev.values, layout.depth_levels)
    np.testing.assert_array_equal(source.dz.values, layout.depth_thickness)
    # Bounds and averaging intervals are useful source metadata, but are not
    # physical variables to include in weighted profile/heat calculations.
    reference = reference[fields + ["wetmask", "areacello"]]
    basins = root.resolve(cfg.basins_location).open()
    runs: list[VizRun] = []
    for run_cfg in cfg.runs:
        run = run_cfg.build(root)
        np.testing.assert_array_equal(run.data.time.values, reference.time.values)
        run.data = run.data[fields]
        runs.append(run)
    report: dict[str, Any] = {
        "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
        "calendar": "noleap",
        "forecast_count": reference.sizes["time"],
        "forecast_first": str(reference.time.values[0]),
        "forecast_last": str(reference.time.values[-1]),
        "groundtruth": str(cfg.groundtruth_location),
        "runs": [run.name for run in runs],
        "steps": [],
    }
    logging.info("CM4_VIZ_INPUTS %s", json.dumps(report))
    prepared = prepare_viz_groundtruth(
        cfg.dataset_name, basins, reference, dates.time_slice, grid_type="gaussian"
    )
    viz = Viz(str(output), cfg.dataset_name, runs, prepared_groundtruth=prepared)
    for step in cfg.selected_steps:
        step_started = time.perf_counter()
        run_steps(viz, [step])
        report["steps"].append(
            {"name": step, "seconds": time.perf_counter() - step_started}
        )
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    pngs = sorted(output.rglob("*.png"))
    assert pngs, "No PNG figures generated"
    for path in pngs:
        with Image.open(path) as figure:
            figure.verify()
    movies: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*.mp4")):
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,width,height:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )
        assert int(probe["streams"][0]["nb_frames"]) == reference.sizes["time"]
        movies.append({"path": str(path.relative_to(output)), "probe": probe})
    assert len(movies) == 4, f"Expected four full-record movies, got {len(movies)}"
    report.update(
        passed=True,
        png_count=len(pngs),
        movies=movies,
        seconds=time.perf_counter() - started,
    )
    (output / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    logging.info("CM4_VIZ_COMPLETE %s", json.dumps(report))


if __name__ == "__main__":
    main(VizConfig.from_yaml_and_cli())
