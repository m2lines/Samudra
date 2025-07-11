"""Ocean visualization modules."""

from .main import run_visualization_pipeline, OceanVisualizationPipeline
from .data_processing import (
    load_groundtruth_data,
    load_basin_data,
    process_data,
    remove_climatology
)
from .analysis import (
    profile_mean,
    create_basin_masks,
    compute_profile_metrics,
    nino_index_compute_clim
)
from .timeseries_viz import generate_all_timeseries_plots
from .spatial_viz import generate_all_spatial_plots

__all__ = [
    "run_visualization_pipeline",
    "OceanVisualizationPipeline",
    "load_groundtruth_data",
    "load_basin_data",
    "process_data",
    "remove_climatology",
    "profile_mean",
    "create_basin_masks",
    "compute_profile_metrics",
    "nino_index_compute_clim",
    "generate_all_timeseries_plots",
    "generate_all_spatial_plots"
]