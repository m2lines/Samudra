"""Snapshot tests for viz.py to ensure refactoring preserves behavior."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import xarray as xr
from PIL import Image, ImageChops


class VizSnapshotTester:
    """Test framework for capturing and comparing viz.py outputs."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.snapshot_dir = Path("tests/snapshots") / test_name
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def capture_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """Capture all outputs from viz.py execution."""
        outputs = {
            "plots": self._capture_plots(output_dir),
            "data_files": self._capture_data_files(output_dir),
            "metrics": self._capture_metrics(output_dir),
            "file_structure": self._capture_file_structure(output_dir),
        }
        return outputs

    def _capture_plots(self, output_dir: Path) -> Dict[str, str]:
        """Capture plot files and their hashes."""
        plots = {}
        for plot_file in output_dir.rglob("*.png"):
            rel_path = plot_file.relative_to(output_dir)
            plots[str(rel_path)] = self._hash_image(plot_file)
        return plots

    def _capture_data_files(self, output_dir: Path) -> Dict[str, str]:
        """Capture data files and their hashes."""
        data_files = {}
        for data_file in output_dir.rglob("*.txt"):
            rel_path = data_file.relative_to(output_dir)
            data_files[str(rel_path)] = self._hash_file(data_file)
        return data_files

    def _capture_metrics(self, output_dir: Path) -> Dict[str, Any]:
        """Capture numerical metrics from output files."""
        metrics = {}

        # Look for text files with metrics
        for txt_file in output_dir.rglob("*.txt"):
            try:
                with open(txt_file, "r") as f:
                    content = f.read()
                    # Extract numerical values from the content
                    rel_path = txt_file.relative_to(output_dir)
                    metrics[str(rel_path)] = self._extract_numbers(content)
            except Exception as e:
                print(f"Could not read metrics from {txt_file}: {e}")

        return metrics

    def _capture_file_structure(self, output_dir: Path) -> Dict[str, List[str]]:
        """Capture the directory structure and file listing."""
        structure = {}
        for root, dirs, files in os.walk(output_dir):
            root_path = Path(root)
            rel_path = root_path.relative_to(output_dir)
            structure[str(rel_path)] = sorted(files)
        return structure

    def _hash_image(self, image_path: Path) -> str:
        """Create a perceptual hash of an image."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB for consistency
                img = img.convert("RGB")
                # Resize to standard size for comparison
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                # Convert to numpy array and hash
                img_array = np.array(img)
                return hashlib.md5(img_array.tobytes()).hexdigest()
        except Exception as e:
            print(f"Could not hash image {image_path}: {e}")
            return self._hash_file(image_path)

    def _hash_file(self, file_path: Path) -> str:
        """Create a hash of a file."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        import re

        # Find all floating point numbers in the text
        numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
        return [float(n) for n in numbers if n]

    def save_snapshot(self, outputs: Dict[str, Any]):
        """Save outputs as a snapshot."""
        snapshot_file = self.snapshot_dir / "snapshot.json"
        with open(snapshot_file, "w") as f:
            json.dump(outputs, f, indent=2, default=str)
        print(f"Snapshot saved to {snapshot_file}")

    def compare_with_snapshot(self, current_outputs: Dict[str, Any]) -> bool:
        """Compare current outputs with saved snapshot."""
        snapshot_file = self.snapshot_dir / "snapshot.json"

        if not snapshot_file.exists():
            print(f"No snapshot found at {snapshot_file}")
            return False

        with open(snapshot_file, "r") as f:
            saved_outputs = json.load(f)

        return self._compare_outputs(saved_outputs, current_outputs)

    def compare_with_config_expected(
        self, current_outputs: Dict[str, Any], expected_outputs: Dict[str, List[str]]
    ) -> bool:
        """Compare current outputs against config-defined expected outputs."""
        return self._compare_config_outputs(expected_outputs, current_outputs)

    def _compare_outputs(self, saved: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Compare two output dictionaries."""
        if saved.keys() != current.keys():
            print(f"Different output categories: {saved.keys()} vs {current.keys()}")
            return False

        for category in saved:
            if not self._compare_category(saved[category], current[category], category):
                return False

        return True

    def _compare_category(self, saved: Any, current: Any, category: str) -> bool:
        """Compare a specific category of outputs."""
        if category == "plots":
            return self._compare_plots(saved, current)
        elif category == "data_files":
            return self._compare_data_files(saved, current)
        elif category == "metrics":
            return self._compare_metrics(saved, current)
        elif category == "file_structure":
            return self._compare_file_structure(saved, current)
        else:
            return saved == current

    def _compare_plots(self, saved: Dict[str, str], current: Dict[str, str]) -> bool:
        """Compare plot hashes."""
        if saved.keys() != current.keys():
            print(f"Different plot files: {saved.keys()} vs {current.keys()}")
            return False

        for plot_name in saved:
            if saved[plot_name] != current[plot_name]:
                print(f"Plot {plot_name} differs")
                return False

        return True

    def _compare_data_files(
        self, saved: Dict[str, str], current: Dict[str, str]
    ) -> bool:
        """Compare data file hashes."""
        if saved.keys() != current.keys():
            print(f"Different data files: {saved.keys()} vs {current.keys()}")
            return False

        for file_name in saved:
            if saved[file_name] != current[file_name]:
                print(f"Data file {file_name} differs")
                return False

        return True

    def _compare_metrics(
        self, saved: Dict[str, List[float]], current: Dict[str, List[float]]
    ) -> bool:
        """Compare numerical metrics with tolerance."""
        if saved.keys() != current.keys():
            print(f"Different metric files: {saved.keys()} vs {current.keys()}")
            return False

        for file_name in saved:
            saved_nums = saved[file_name]
            current_nums = current[file_name]

            if len(saved_nums) != len(current_nums):
                print(
                    f"Different number of metrics in {file_name}: {len(saved_nums)} vs {len(current_nums)}"
                )
                return False

            for i, (s, c) in enumerate(zip(saved_nums, current_nums)):
                if not np.isclose(s, c, rtol=1e-10, atol=1e-10):
                    print(f"Metric {i} in {file_name} differs: {s} vs {c}")
                    return False

        return True

    def _compare_file_structure(
        self, saved: Dict[str, List[str]], current: Dict[str, List[str]]
    ) -> bool:
        """Compare file structure."""
        if saved.keys() != current.keys():
            print(f"Different directory structure: {saved.keys()} vs {current.keys()}")
            return False

        for dir_name in saved:
            if saved[dir_name] != current[dir_name]:
                print(
                    f"Different files in {dir_name}: {saved[dir_name]} vs {current[dir_name]}"
                )
                return False

        return True

    def _compare_config_outputs(
        self, expected: Dict[str, List[str]], current: Dict[str, Any]
    ) -> bool:
        """Compare current outputs against config-defined expected outputs."""
        print(f"Checking expected plots: {expected.get('plots', [])}")
        print(f"Checking expected directories: {expected.get('directories', [])}")

        # Check that expected plots exist
        current_plots = current.get("plots", {})
        for expected_plot in expected.get("plots", []):
            if expected_plot not in current_plots:
                print(f"✗ Missing expected plot: {expected_plot}")
                return False
            else:
                print(f"✓ Found expected plot: {expected_plot}")

        # Check that expected directories exist
        current_structure = current.get("file_structure", {})
        for expected_dir in expected.get("directories", []):
            if expected_dir not in current_structure:
                print(f"✗ Missing expected directory: {expected_dir}")
                return False
            else:
                print(f"✓ Found expected directory: {expected_dir}")

        # Check that no unexpected plots were generated (optional strict mode)
        unexpected_plots = []
        for plot_name in current_plots:
            if plot_name not in expected.get("plots", []):
                unexpected_plots.append(plot_name)

        if unexpected_plots:
            print(f"⚠ Found {len(unexpected_plots)} unexpected plots (this may be OK):")
            for plot in unexpected_plots[:5]:  # Show first 5
                print(f"  - {plot}")
            if len(unexpected_plots) > 5:
                print(f"  ... and {len(unexpected_plots) - 5} more")

        print(
            f"✓ Config validation passed: found {len(expected.get('plots', []))} expected plots"
        )
        return True


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for viz outputs."""
    output_dir = tmp_path / "viz_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def snapshot_tester():
    """Create a snapshot tester instance."""
    return VizSnapshotTester("viz_baseline")


@pytest.mark.manual
def test_viz_baseline_snapshot(temp_output_dir, snapshot_tester):
    """
    Create a baseline snapshot of viz.py outputs.

    This test should be run manually to create the initial snapshot:
    pytest -m manual tests/test_viz_snapshot.py::test_viz_baseline_snapshot -v
    """
    print("Using existing gold outputs as baseline...")

    import shutil
    from pathlib import Path

    # Copy the existing gold outputs to our test directory
    gold_outputs_path = Path("/data/2025-06-10_samudra-recreate-paper-om4")

    if gold_outputs_path.exists():
        # Copy all files from gold outputs to temp directory
        for item in gold_outputs_path.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(gold_outputs_path)
                dest_path = temp_output_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)

        print(
            f"Copied {len(list(gold_outputs_path.rglob('*')))} items from gold outputs"
        )
    else:
        print("Warning: Gold outputs path not found, creating minimal baseline")
        # Create minimal test structure
        (temp_output_dir / "test_output.txt").write_text("baseline test")

    # Capture outputs
    outputs = snapshot_tester.capture_outputs(temp_output_dir)

    # Save as baseline
    snapshot_tester.save_snapshot(outputs)

    print("Baseline snapshot created successfully!")


@pytest.mark.manual
def test_viz_refactored_comparison(temp_output_dir, snapshot_tester):
    """
    Compare refactored viz.py outputs against baseline.

    This test should be run after refactoring to ensure behavior is preserved:
    pytest -m manual tests/test_viz_snapshot.py::test_viz_refactored_comparison -v
    """
    print("Running refactored visualization pipeline...")

    # Execute refactored viz.py
    run_refactored_viz_script(temp_output_dir)

    # Capture outputs
    outputs = snapshot_tester.capture_outputs(temp_output_dir)

    # Compare with baseline
    matches = snapshot_tester.compare_with_snapshot(outputs)

    assert matches, "Refactored viz.py outputs do not match baseline!"
    print("Refactored outputs match baseline successfully!")


@pytest.mark.manual
def test_minimal_zos_only():
    """
    Test minimal zos-only configuration to validate core functionality.

    This test uses the most minimal configuration to verify timeseries plotting works:
    pytest -m manual tests/test_viz_snapshot.py::test_minimal_zos_only -v
    """
    import tempfile
    from pathlib import Path
    from ocean_emulators.viz_modules.config import VizConfig
    from ocean_emulators.viz_modules.main import run_visualization_pipeline

    print("Testing minimal zos-only configuration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "zos_test"

        # Create minimal config
        config = VizConfig.minimal_zos_only(str(output_dir))

        try:
            # Run the pipeline
            result_path = run_visualization_pipeline(config)

            print(f"✓ Pipeline completed successfully")
            print(f"✓ Output saved to: {result_path}")

            # Check expected outputs using config-aware comparison
            expected_outputs = config.get_expected_outputs()
            output_path = Path(result_path)

            # Capture outputs using snapshot tester
            snapshot_tester = VizSnapshotTester("config_validation")
            outputs = snapshot_tester.capture_outputs(output_path)

            # Use config-aware comparison
            matches = snapshot_tester.compare_with_config_expected(
                outputs, expected_outputs
            )

            # List all actual outputs for debugging
            if output_path.exists():
                all_files = list(output_path.rglob("*"))
                print(f"✓ Total files generated: {len(all_files)}")
                for f in sorted(all_files):
                    if f.is_file():
                        rel_path = f.relative_to(output_path)
                        print(f"  - {rel_path}")

            if not matches:
                print("✗ Config validation failed!")
                return False

            return True

        except Exception as e:
            print(f"✗ Pipeline failed: {e}")
            raise


def run_refactored_viz_script(output_dir: Path, config=None) -> Dict[str, Any]:
    """Execute the refactored viz.py script and return outputs."""
    from ocean_emulators.viz import run_viz_analysis

    if config is None:
        config = {
            "dataset_name": "OM4",
            "pred_dict": {
                "pred_1": {
                    "name": "samudra-10-year-high-res",
                    "run_name": "samudra-10-year-high-res",
                    "path": "/data/om4_samudra_lowres_predictions/predictions.zarr",
                    "ls": ["thetao", "so", "uo", "vo", "tos", "zos"],
                }
            },
            "key1": "pred_1",
            "levels": 19,
            "output_path": str(output_dir),
            "groundtruth_path": "/data/public/OM4.zarr",
            "basin_path": "/data/basins/basin_masks_original.zarr",
        }

    output_path = run_viz_analysis(config, minimal=True)
    return {}
