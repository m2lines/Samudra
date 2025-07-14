"""Ocean visualization modules."""

from .analysis import (
    compute_profile_metrics,
    create_basin_masks,
    nino_index_compute_clim,
    profile_mean,
)
from .data_processing import (
    load_basin_data,
    load_groundtruth_data,
    process_data,
    remove_climatology,
)
from .main import OceanVisualizationPipeline, run_visualization_pipeline
from .spatial_viz import generate_all_spatial_plots
from .timeseries_viz import generate_all_timeseries_plots

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
    "generate_all_spatial_plots",
]
