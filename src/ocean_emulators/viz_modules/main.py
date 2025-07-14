"""Main orchestration module for ocean visualization."""

import os
from typing import Any

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from .analysis import (
    compute_profile_metrics,
    create_basin_masks,
    nino_index_compute_clim,
    profile_mean,
)
from .config import VizConfig
from .data_processing import (
    load_basin_data,
    load_groundtruth_data,
    process_data,
    remove_climatology,
)
from .pdfs_viz import generate_all_pdf_plots
from .spatial_viz import generate_all_spatial_plots
from .timeseries_viz import generate_all_timeseries_plots


class OceanVisualizationPipeline:
    """Main class for orchestrating ocean visualization analysis."""

    def __init__(self, config: dict[str, Any] | VizConfig):
        """
        Initialize the visualization pipeline.

        Args:
            config: Configuration dictionary or VizConfig object containing paths and parameters
        """
        if isinstance(config, dict):
            # Convert dict to VizConfig for backward compatibility
            self.config = VizConfig(**config)
        else:
            self.config = config

        self.output_path = self.config.output_path
        self.pred_dict = self.config.pred_dict
        self.dataset_name = self.config.dataset_name

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Visualization settings
        self.var_labels = {
            "vo": r"$v$ $( m/s )$",
            "uo": r"$u$ $( m/s )$",
            "thetao": r"$T$ $( ^\circ C )$",
            "tos": r"$T$ $( ^\circ C )$",
            "so": r"$so$ $( psu )$",
            "zos": r"$zos$ $( m )$",
            "KE": r"$KE$ $( J/m^2 )$",
            "OHC": r"$OHC$ $Anomaly$ $( ZJ )$",
        }

        self.colors = self.config.colors
        self.titles = self.config.titles

        # Data storage
        self.ds_groundtruth = None
        self.pred_dict_processed = None
        self.basin_masks = None
        self.profile_groundtruth = None
        self.pred_profiles = {}

    def load_data(self) -> None:
        """Load all required datasets."""
        print("Loading datasets...")

        # Load ground truth data
        groundtruth_path = self.config.groundtruth_path
        groundtruth_rollout = load_groundtruth_data(groundtruth_path)

        # Load basin data
        basin_path = self.config.basin_path
        basins = load_basin_data(basin_path)

        # Process data
        print("Processing data...")
        self.ds_groundtruth, self.pred_dict_processed = process_data(
            groundtruth_rollout, self.pred_dict
        )

        # Create basin masks
        print("Creating basin masks...")
        self.basin_masks = create_basin_masks(basins, self.ds_groundtruth)

        print("Data loading completed!")

    def compute_profiles(self, use_cache: bool = True) -> None:
        """Compute spatial profiles for all datasets with optional caching."""
        print("Computing profiles...")

        # Try to load from cache first
        if use_cache and self._load_cached_profiles():
            print("Loaded profiles from cache!")
            return

        print("Computing profiles from scratch...")
        with ProgressBar():
            print(f"Ground truth {self.dataset_name}")
            self.profile_groundtruth = profile_mean(self.ds_groundtruth).load()

            for k in self.pred_dict_processed.keys():
                print(f"Processing {k}")
                self.pred_profiles[k] = profile_mean(
                    self.pred_dict_processed[k]["ds_prediction"]
                ).load()

        # Save to cache
        if use_cache:
            self._save_cached_profiles()

        print("Profile computation completed!")

    def _get_cache_path(self, dataset_type: str, key: str = None) -> str:
        """Get the cache path for a specific dataset."""
        import hashlib
        from pathlib import Path

        # Use ~/.cache/ocean_emulators_profiles for caching
        cache_base_dir = Path.home() / ".cache" / "ocean_emulators_profiles"
        cache_base_dir.mkdir(parents=True, exist_ok=True)

        if dataset_type == "groundtruth":
            # Create hash of groundtruth path for unique identification
            path_hash = hashlib.md5(self.config.groundtruth_path.encode()).hexdigest()[
                :8
            ]
            cache_file = f"groundtruth_{self.dataset_name}_{path_hash}.zarr"
        else:  # prediction
            pred_data = self.pred_dict_processed[key]
            # Create hash of prediction path for unique identification
            path_hash = hashlib.md5(pred_data["path"].encode()).hexdigest()[:8]
            cache_file = f"prediction_{key}_{path_hash}.zarr"

        return str(cache_base_dir / cache_file)

    def _load_cached_profiles(self) -> bool:
        """Load cached profiles if they exist and are valid."""
        try:
            # Load ground truth profile
            gt_cache_path = self._get_cache_path("groundtruth")
            if not os.path.exists(gt_cache_path):
                return False

            self.profile_groundtruth = xr.open_zarr(gt_cache_path)
            print(f"Loaded cached ground truth profile from {gt_cache_path}")

            # Load prediction profiles
            for k in self.pred_dict_processed.keys():
                pred_cache_path = self._get_cache_path("prediction", k)
                if not os.path.exists(pred_cache_path):
                    return False

                self.pred_profiles[k] = xr.open_zarr(pred_cache_path)
                print(
                    f"Loaded cached prediction profile for {k} from {pred_cache_path}"
                )

            return True

        except Exception as e:
            print(f"Failed to load cached profiles: {e}")
            return False

    def _save_cached_profiles(self) -> None:
        """Save computed profiles to cache."""
        try:
            # Save ground truth profile
            gt_cache_path = self._get_cache_path("groundtruth")
            self.profile_groundtruth.to_zarr(gt_cache_path, mode="w")
            print(f"Saved ground truth profile to {gt_cache_path}")

            # Save prediction profiles
            for k, profile in self.pred_profiles.items():
                pred_cache_path = self._get_cache_path("prediction", k)
                profile.to_zarr(pred_cache_path, mode="w")
                print(f"Saved prediction profile for {k} to {pred_cache_path}")

        except Exception as e:
            print(f"Failed to save profiles to cache: {e}")

    def clear_profile_cache(self) -> None:
        """Clear cached profiles for this configuration."""
        import shutil

        try:
            # Clear ground truth cache
            gt_cache_path = self._get_cache_path("groundtruth")
            if os.path.exists(gt_cache_path):
                shutil.rmtree(gt_cache_path)
                print(f"Cleared ground truth cache: {gt_cache_path}")

            # Clear prediction caches
            for k in self.pred_dict_processed.keys():
                pred_cache_path = self._get_cache_path("prediction", k)
                if os.path.exists(pred_cache_path):
                    shutil.rmtree(pred_cache_path)
                    print(f"Cleared prediction cache for {k}: {pred_cache_path}")

        except Exception as e:
            print(f"Failed to clear cache: {e}")

    def show_cache_status(self) -> None:
        """Show cache status and disk usage."""
        from pathlib import Path

        cache_base_dir = Path.home() / ".cache" / "ocean_emulators_profiles"

        if not cache_base_dir.exists():
            print("📁 No cache directory found")
            return

        print(f"📁 Cache directory: {cache_base_dir}")

        # List all cache files
        cache_files = list(cache_base_dir.glob("*.zarr"))
        if not cache_files:
            print("📄 No cached profiles found")
            return

        total_size = 0
        print("📄 Cached profiles:")
        for cache_file in cache_files:
            try:
                size = sum(
                    f.stat().st_size for f in cache_file.rglob("*") if f.is_file()
                )
                total_size += size
                size_mb = size / (1024 * 1024)
                print(f"  • {cache_file.name}: {size_mb:.1f} MB")
            except Exception as e:
                print(f"  • {cache_file.name}: Error reading size ({e})")

        total_mb = total_size / (1024 * 1024)
        print(f"💾 Total cache size: {total_mb:.1f} MB")

    def generate_timeseries_analysis(self) -> None:
        """Generate all time series visualizations."""
        print("Generating time series plots...")

        generate_all_timeseries_plots(
            self.profile_groundtruth,
            self.pred_profiles,
            self.output_path,
            self.var_labels,
            self.colors,
            self.titles,
            variables=self.config.timeseries_variables,
            ds_groundtruth=self.ds_groundtruth,
            pred_dict_processed=self.pred_dict_processed,
        )

        print("Time series plots completed!")

    def generate_spatial_analysis(self) -> None:
        """Generate all spatial visualizations."""
        print("Generating spatial plots...")

        generate_all_spatial_plots(
            self.ds_groundtruth,
            self.pred_dict_processed,
            self.basin_masks,
            self.output_path,
            self.titles,
        )

        print("Spatial plots completed!")

    def generate_ohc_analysis(self) -> None:
        """Generate Ocean Heat Content analysis plots."""
        print("Generating OHC plots...")

        from .spatial_viz import create_ohc_timeseries_plots

        create_ohc_timeseries_plots(
            self.ds_groundtruth, self.pred_dict_processed, self.output_path, self.titles
        )

        print("OHC plots completed!")

    def generate_pdf_analysis(self) -> None:
        """Generate PDF analysis plots."""
        print("Generating PDF plots...")

        generate_all_pdf_plots(
            self.ds_groundtruth,
            self.pred_dict_processed,
            self.output_path,
            self.titles,
            self.dataset_name,
        )

        print("PDF plots completed!")

    def compute_climate_indices(self) -> None:
        """Compute climate indices like ENSO."""
        print("Computing climate indices...")

        # Compute Niño 3.4 index
        if "tos" in self.ds_groundtruth.data_vars:
            sst_data = self.ds_groundtruth["tos"]
        else:
            sst_data = self.ds_groundtruth["thetao"].isel(lev=0)

        area_weights = self.ds_groundtruth["areacello"]

        # Compute for ground truth
        nino_gt, clim_gt = nino_index_compute_clim(sst_data, area_weights)

        # Compute for predictions
        nino_preds = {}
        for key, pred_data in self.pred_dict_processed.items():
            ds_pred = pred_data["ds_prediction"]
            if "tos" in ds_pred.data_vars:
                sst_pred = ds_pred["tos"]
            else:
                sst_pred = ds_pred["thetao"].isel(lev=0)

            nino_pred, _ = nino_index_compute_clim(sst_pred, area_weights)
            nino_preds[key] = nino_pred

        # Save climate indices
        self._save_climate_indices(nino_gt, nino_preds)

        # Generate ENSO visualization plots
        self._generate_enso_visualizations()

        print("Climate indices completed!")

    def _save_climate_indices(
        self, nino_gt: xr.DataArray, nino_preds: dict[str, xr.DataArray]
    ) -> None:
        """Save climate indices to files."""
        enso_path = os.path.join(self.output_path, "ENSO")
        os.makedirs(enso_path, exist_ok=True)

        # Save Niño indices
        nino_gt.to_netcdf(os.path.join(enso_path, "nino34_groundtruth.nc"))

        for key, nino_pred in nino_preds.items():
            nino_pred.to_netcdf(os.path.join(enso_path, f"nino34_{key}.nc"))

    def _generate_enso_visualizations(self) -> None:
        """Generate ENSO visualization plots."""
        from .enso_viz import generate_enso_visualizations

        generate_enso_visualizations(
            self.ds_groundtruth,
            self.pred_dict_processed,
            self.output_path,
            self.colors,
            self.titles,
            self.dataset_name,
        )

    def compute_metrics(self) -> None:
        """Compute and save evaluation metrics."""
        print("Computing metrics...")

        # Compute profile metrics
        profile_metrics = compute_profile_metrics(
            self.pred_profiles, self.profile_groundtruth
        )

        # Save metrics
        self._save_metrics(profile_metrics)

        print("Metrics computation completed!")

    def _save_metrics(self, profile_metrics: dict[str, dict[str, xr.Dataset]]) -> None:
        """Save metrics to files."""
        metrics_path = os.path.join(self.output_path, "Metrics")
        os.makedirs(metrics_path, exist_ok=True)

        for pred_key, metrics in profile_metrics.items():
            for metric_name, metric_data in metrics.items():
                filename = f"{pred_key}_{metric_name}.nc"
                metric_data.to_netcdf(os.path.join(metrics_path, filename))

        # Also save text summaries
        self._save_metrics_summary(profile_metrics, metrics_path)

    def _save_metrics_summary(
        self, profile_metrics: dict[str, dict[str, xr.Dataset]], metrics_path: str
    ) -> None:
        """Save human-readable metrics summary."""
        for pred_key, metrics in profile_metrics.items():
            summary_file = os.path.join(metrics_path, f"{pred_key}_summary.txt")

            with open(summary_file, "w") as f:
                f.write(f"Metrics Summary for {pred_key}\n")
                f.write("=" * 50 + "\n\n")

                for metric_name, metric_data in metrics.items():
                    f.write(f"{metric_name.upper()}:\n")

                    # For each variable, compute global mean
                    for var in metric_data.data_vars:
                        if var in ["thetao", "so", "uo", "vo", "zos"]:
                            global_mean = metric_data[var].mean().compute().item()
                            f.write(f"  {var}: {global_mean:.6f}\n")

                    f.write("\n")

        # Create the specific metrics files that match original viz.py naming
        self._create_original_metrics_files(profile_metrics, metrics_path)

    def _create_original_metrics_files(
        self, profile_metrics: dict[str, dict[str, xr.Dataset]], metrics_path: str
    ) -> None:
        """Create the specific metrics files that match original viz.py naming."""
        # Create thetao_mae_info.txt
        thetao_mae_file = os.path.join(metrics_path, "thetao_mae_info.txt")
        with open(thetao_mae_file, "w") as f:
            for pred_key, metrics in profile_metrics.items():
                if "mae" in metrics and "thetao" in metrics["mae"].data_vars:
                    mae_value = metrics["mae"]["thetao"].mean().compute().item()
                    pred_name = self.pred_dict_processed[pred_key].get("name", pred_key)
                    f.write(f"\n Thetao {pred_name} MAE : {mae_value}")

        # Create sst_mae_info.txt (using surface temperature data)
        sst_mae_file = os.path.join(metrics_path, "sst_mae_info.txt")
        with open(sst_mae_file, "w") as f:
            for pred_key, metrics in profile_metrics.items():
                if "mae" in metrics and "thetao" in metrics["mae"].data_vars:
                    # For SST, use surface level (lev=0) if available
                    if "lev" in metrics["mae"]["thetao"].dims:
                        sst_mae_value = (
                            metrics["mae"]["thetao"].isel(lev=0).mean().compute().item()
                        )
                    else:
                        sst_mae_value = metrics["mae"]["thetao"].mean().compute().item()
                    pred_name = self.pred_dict_processed[pred_key].get("name", pred_key)
                    f.write(f"\n SST {pred_name} MAE : {sst_mae_value}")

        # Create salinity_deseasonalized_info.txt
        # This requires deseasonalized trend computation
        salinity_deseasonalized_file = os.path.join(
            metrics_path, "salinity_deseasonalized_info.txt"
        )
        with open(salinity_deseasonalized_file, "w") as f:
            for pred_key in profile_metrics.keys():
                if pred_key in self.pred_dict_processed:
                    # Compute deseasonalized salinity trend
                    pred_data = self.pred_dict_processed[pred_key]["ds_prediction"]
                    if "so" in pred_data.data_vars:
                        # Compute global volume-weighted mean salinity
                        salinity_pred = (
                            pred_data["so"]
                            .weighted(
                                self.ds_groundtruth["areacello"]
                                * self.ds_groundtruth["dz"]
                            )
                            .mean(["x", "y", "lev"])
                        )
                        # Remove climatology (deseasonalize)
                        salinity_deseasonalized = remove_climatology(salinity_pred)

                        # Compute trend slope
                        coeffs_salinity_pred_trend = np.polyfit(
                            np.arange(salinity_deseasonalized.size),
                            salinity_deseasonalized,
                            1,
                        )
                        pred_name = self.pred_dict_processed[pred_key].get(
                            "name", pred_key
                        )
                        f.write(
                            f"\nSalinity {pred_name} Trend Slope : {coeffs_salinity_pred_trend[0]}"
                        )

    def run_complete_analysis(self) -> None:
        """Run the complete visualization analysis pipeline."""
        print("Starting Ocean Visualization Analysis...")

        # Load and process data
        self.load_data()

        # Compute profiles
        self.compute_profiles()

        # Generate visualizations
        self.generate_timeseries_analysis()
        self.generate_spatial_analysis()

        # Compute climate indices
        self.compute_climate_indices()

        # Compute metrics
        self.compute_metrics()

        print(f"Analysis completed! Results saved to: {self.output_path}")

    def run_minimal_analysis(self) -> None:
        """Run a minimal analysis for testing purposes."""
        print("Starting minimal Ocean Visualization Analysis...")

        # Load and process data
        self.load_data()

        # Compute profiles
        self.compute_profiles()

        # Generate only time series (fastest)
        self.generate_timeseries_analysis()

        print(f"Minimal analysis completed! Results saved to: {self.output_path}")

    def run_config_based_analysis(self) -> None:
        """Run analysis based on configuration."""
        print(
            f"Starting Ocean Visualization Analysis with groups: {self.config.analysis_groups}"
        )

        # Load and process data
        self.load_data()

        # Compute profiles if needed
        if any(
            group in self.config.analysis_groups for group in ["timeseries", "metrics"]
        ):
            self.compute_profiles()

        # Run analysis groups based on config
        if "timeseries" in self.config.analysis_groups:
            self.generate_timeseries_analysis()

        if "spatial" in self.config.analysis_groups:
            self.generate_spatial_analysis()

        if "ohc" in self.config.analysis_groups:
            self.generate_ohc_analysis()

        if "enso" in self.config.analysis_groups:
            self.compute_climate_indices()

        if "metrics" in self.config.analysis_groups:
            self.compute_metrics()

        if "pdfs" in self.config.analysis_groups:
            self.generate_pdf_analysis()

        print(f"Analysis completed! Results saved to: {self.output_path}")


def run_visualization_pipeline(
    config: dict[str, Any] | VizConfig, minimal: bool = False
) -> str:
    """
    Run the complete ocean visualization pipeline.

    Args:
        config: Configuration dictionary or VizConfig object
        minimal: If True, run minimal analysis for testing

    Returns:
        Path to output directory
    """
    pipeline = OceanVisualizationPipeline(config)

    if minimal:
        pipeline.run_minimal_analysis()
    elif isinstance(config, VizConfig):
        pipeline.run_config_based_analysis()
    else:
        pipeline.run_complete_analysis()

    return pipeline.output_path
