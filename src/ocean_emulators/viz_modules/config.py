"""Configuration system for ocean visualization pipeline."""

from dataclasses import dataclass


@dataclass
class VizConfig:
    """Configuration for ocean visualization pipeline."""

    # Data paths
    groundtruth_path: str = "/data/public/OM4.zarr"
    basin_path: str = "/data/basins/basin_masks_original.zarr"
    output_path: str = "/tmp/viz_outputs"

    # Dataset configuration
    dataset_name: str = "OM4"
    pred_dict: dict = None
    key1: str = "pred_1"
    levels: int = 19

    # Variables to process
    variables: list[str] = None

    # Analysis groups to run
    analysis_groups: set[str] = None

    # Timeseries configuration
    timeseries_variables: list[str] = None
    timeseries_levels: list[int] = None

    # Plotting configuration
    colors: list[str] = None
    titles: list[str] = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.pred_dict is None:
            self.pred_dict = {
                "pred_1": {
                    "name": "samudra-10-year-high-res",
                    "run_name": "samudra-10-year-high-res",
                    "path": "/data/om4_samudra_lowres_predictions/predictions.zarr",
                    "ls": ["thetao", "so", "uo", "vo", "tos", "zos"],
                }
            }

        if self.variables is None:
            self.variables = ["thetao", "so", "uo", "vo", "zos"]

        if self.analysis_groups is None:
            self.analysis_groups = {"timeseries"}

        if self.timeseries_variables is None:
            self.timeseries_variables = self.variables.copy()

        if self.timeseries_levels is None:
            self.timeseries_levels = [0, 5, 10]

        if self.colors is None:
            self.colors = ["red", "blue", "green", "orange", "purple"]

        if self.titles is None:
            self.titles = ["Prediction 1"]

    @classmethod
    def minimal_zos_only(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a minimal config for testing with only zos variable."""
        return cls(
            output_path=output_path,
            variables=["zos"],
            analysis_groups={"timeseries"},
            timeseries_variables=["zos"],
            timeseries_levels=[0],  # zos is 2D, so only level 0
        )

    @classmethod
    def minimal_timeseries(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a minimal config for timeseries testing."""
        return cls(
            output_path=output_path,
            variables=["thetao", "so", "zos"],
            analysis_groups={"timeseries"},
            timeseries_variables=["thetao", "so", "zos"],
            timeseries_levels=[0, 5, 10],
        )

    @classmethod
    def minimal_spatial(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a minimal config for spatial testing."""
        return cls(
            output_path=output_path,
            variables=["thetao", "so"],  # Temperature and salinity for spatial plots
            analysis_groups={"spatial"},
            timeseries_variables=[],  # No timeseries needed
        )

    @classmethod
    def minimal_enso(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a minimal config for ENSO/climate indices testing."""
        return cls(
            output_path=output_path,
            variables=["tos"],  # Sea surface temperature for ENSO
            analysis_groups={"enso"},
            timeseries_variables=[],  # No timeseries needed
        )

    @classmethod
    def minimal_metrics(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a minimal config for metrics testing."""
        return cls(
            output_path=output_path,
            variables=["thetao", "zos"],  # Basic variables for metrics
            analysis_groups={"metrics"},
            timeseries_variables=[],  # No timeseries needed
        )

    @classmethod
    def minimal_ohc(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a minimal config for OHC testing."""
        return cls(
            output_path=output_path,
            variables=["thetao"],  # Temperature for OHC calculation
            analysis_groups={"ohc"},
            timeseries_variables=[],  # No timeseries needed
        )

    @classmethod
    def full_analysis(cls, output_path: str = "/tmp/viz_outputs") -> "VizConfig":
        """Create a full config for complete analysis."""
        return cls(
            output_path=output_path,
            variables=["thetao", "so", "uo", "vo", "zos"],
            analysis_groups={"timeseries", "spatial", "ohc", "enso", "pdfs", "metrics"},
            timeseries_variables=["thetao", "so", "uo", "vo", "zos"],
            timeseries_levels=[0, 5, 10, 15],
        )

    def get_expected_outputs(self) -> dict[str, list[str]]:
        """Get expected output files based on configuration."""
        expected = {"plots": [], "data_files": [], "directories": []}

        if "timeseries" in self.analysis_groups:
            expected["directories"].append("Timeseries")

            # Add timeseries summary plots
            expected["plots"].extend(
                [
                    "Timeseries/timeseries_grid_shallow_all_vars.png",
                    "Timeseries/temperature_profiles_comparison.png",
                ]
            )

            # Add individual variable timeseries
            for var in self.timeseries_variables:
                expected["directories"].append(f"Timeseries/{var}_timeseries")

                if var in ["thetao", "so", "uo", "vo"] and "lev" in getattr(
                    self, f"{var}_dims", ["lev"]
                ):
                    # 3D variables with levels
                    for level in range(self.levels):
                        expected["plots"].append(
                            f"Timeseries/{var}_timeseries/{level}.png"
                        )
                else:
                    # 2D variables (like zos)
                    expected["plots"].append(f"Timeseries/{var}_timeseries/0.png")

        if "spatial" in self.analysis_groups:
            expected["directories"].extend(["Temperature", "Salinity"])
            expected["plots"].extend(
                [
                    "Temperature/SST_Global_map.png",
                    "Salinity/SeaSurfaceSalinity_Global_map.png",
                ]
            )

        if "ohc" in self.analysis_groups:
            expected["directories"].append("OHC")
            expected["plots"].extend(["OHC/OHC.png", "OHC/OHC_Global_map.png"])

        if "enso" in self.analysis_groups:
            expected["directories"].append("ENSO")
            expected["data_files"].extend(
                ["ENSO/nino34_groundtruth.nc", "ENSO/nino34_pred_1.nc"]
            )

        if "pdfs" in self.analysis_groups:
            expected["directories"].append("PDFs")
            expected["plots"].extend(["PDFs/PDF_Plots_Short.png"])

        if "metrics" in self.analysis_groups:
            expected["directories"].append("Metrics")
            expected["data_files"].extend(
                [
                    "Metrics/pred_1_rmse.nc",
                    "Metrics/pred_1_mae.nc",
                    "Metrics/pred_1_summary.txt",
                ]
            )

        return expected
