"""Visualization script for ocean emulator outputs."""

import os
from datetime import datetime
from typing import Any

from ocean_emulators.viz_modules import run_visualization_pipeline


def get_default_config() -> dict[str, Any]:
    """Get the default configuration for visualization."""
    pred_dict = {}
    pred_dict["pred_1"] = {
        "name": "samudra-10-year-high-res",
        "run_name": "samudra-10-year-high-res",
        "path": "/data/om4_samudra_lowres_predictions/predictions.zarr",
        "ls": ["thetao", "so", "uo", "vo", "tos", "zos"],
    }

    key1 = list(pred_dict.keys())[0]
    levels = 19

    output_path = (
        "/tmp/viz_outputs/"
        + str(datetime.now())[:10]
        + "_"
        + "_".join([pred_dict[k]["run_name"] for k in pred_dict.keys()])
    )

    return {
        "dataset_name": "OM4",
        "pred_dict": pred_dict,
        "key1": key1,
        "levels": levels,
        "output_path": output_path,
        "groundtruth_path": "/data/public/OM4.zarr",
        "basin_path": "/data/basins/basin_masks_original.zarr",
    }


def run_viz_analysis(
    config: dict[str, Any] | None = None, minimal: bool = False
) -> str:
    """
    Run the visualization analysis pipeline.

    Args:
        config: Configuration dictionary. If None, uses default config.
        minimal: If True, run minimal analysis for testing

    Returns:
        Path to the output directory where visualizations are saved.
    """
    if config is None:
        config = get_default_config()

    # Set up environment
    os.environ["FSSPEC_S3_ENDPOINT_URL"] = "https://nyu1.osn.mghpcc.org"
    os.environ["AWS_PROFILE"] = "m2l"

    # Run the visualization pipeline
    output_path = run_visualization_pipeline(config, minimal=minimal)

    return output_path


def main():
    """Main entry point for the visualization script."""
    config = get_default_config()
    output_path = run_viz_analysis(config)
    print(f"Visualization completed. Outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
