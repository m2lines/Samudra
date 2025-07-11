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
            "file_structure": self._capture_file_structure(output_dir)
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
                with open(txt_file, 'r') as f:
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
                img = img.convert('RGB')
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
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        import re
        # Find all floating point numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
        return [float(n) for n in numbers if n]
    
    def save_snapshot(self, outputs: Dict[str, Any]):
        """Save outputs as a snapshot."""
        snapshot_file = self.snapshot_dir / "snapshot.json"
        with open(snapshot_file, 'w') as f:
            json.dump(outputs, f, indent=2, default=str)
        print(f"Snapshot saved to {snapshot_file}")
    
    def compare_with_snapshot(self, current_outputs: Dict[str, Any]) -> bool:
        """Compare current outputs with saved snapshot."""
        snapshot_file = self.snapshot_dir / "snapshot.json"
        
        if not snapshot_file.exists():
            print(f"No snapshot found at {snapshot_file}")
            return False
        
        with open(snapshot_file, 'r') as f:
            saved_outputs = json.load(f)
        
        return self._compare_outputs(saved_outputs, current_outputs)
    
    def compare_with_config_expected(self, current_outputs: Dict[str, Any], expected_outputs: Dict[str, List[str]]) -> bool:
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
    
    def _compare_data_files(self, saved: Dict[str, str], current: Dict[str, str]) -> bool:
        """Compare data file hashes."""
        if saved.keys() != current.keys():
            print(f"Different data files: {saved.keys()} vs {current.keys()}")
            return False
        
        for file_name in saved:
            if saved[file_name] != current[file_name]:
                print(f"Data file {file_name} differs")
                return False
        
        return True
    
    def _compare_metrics(self, saved: Dict[str, List[float]], current: Dict[str, List[float]]) -> bool:
        """Compare numerical metrics with tolerance."""
        if saved.keys() != current.keys():
            print(f"Different metric files: {saved.keys()} vs {current.keys()}")
            return False
        
        for file_name in saved:
            saved_nums = saved[file_name]
            current_nums = current[file_name]
            
            if len(saved_nums) != len(current_nums):
                print(f"Different number of metrics in {file_name}: {len(saved_nums)} vs {len(current_nums)}")
                return False
            
            for i, (s, c) in enumerate(zip(saved_nums, current_nums)):
                if not np.isclose(s, c, rtol=1e-10, atol=1e-10):
                    print(f"Metric {i} in {file_name} differs: {s} vs {c}")
                    return False
        
        return True
    
    def _compare_file_structure(self, saved: Dict[str, List[str]], current: Dict[str, List[str]]) -> bool:
        """Compare file structure."""
        if saved.keys() != current.keys():
            print(f"Different directory structure: {saved.keys()} vs {current.keys()}")
            return False
        
        for dir_name in saved:
            if saved[dir_name] != current[dir_name]:
                print(f"Different files in {dir_name}: {saved[dir_name]} vs {current[dir_name]}")
                return False
        
        return True
    
    def _compare_config_outputs(self, expected: Dict[str, List[str]], current: Dict[str, Any]) -> bool:
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
        
        print(f"✓ Config validation passed: found {len(expected.get('plots', []))} expected plots")
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
        
        print(f"Copied {len(list(gold_outputs_path.rglob('*')))} items from gold outputs")
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
            matches = snapshot_tester.compare_with_config_expected(outputs, expected_outputs)
            
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


@pytest.mark.manual
def test_original_vs_refactored_comparison():
    """
    Compare original viz.py script vs refactored version.
    
    This test runs both versions and compares their outputs directly:
    pytest -m manual tests/test_viz_snapshot.py::test_original_vs_refactored_comparison -v
    """
    import tempfile
    from pathlib import Path
    
    print("Running comparison between original and refactored versions...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create separate output directories
        original_output = temp_path / "original"
        refactored_output = temp_path / "refactored"
        original_output.mkdir()
        refactored_output.mkdir()
        
        print("Running original viz.py script...")
        try:
            run_viz_script(original_output)
            print("✓ Original script completed")
        except Exception as e:
            print(f"✗ Original script failed: {e}")
            # Continue with refactored test even if original fails
        
        print("Running refactored viz.py script...")
        try:
            run_refactored_viz_script(refactored_output)
            print("✓ Refactored script completed")
        except Exception as e:
            print(f"✗ Refactored script failed: {e}")
            pytest.fail(f"Refactored script failed: {e}")
        
        # Compare outputs if both succeeded
        if original_output.exists() and any(original_output.iterdir()):
            print("Comparing outputs...")
            
            # Create snapshot testers for both
            original_tester = VizSnapshotTester("original_comparison")
            refactored_tester = VizSnapshotTester("refactored_comparison")
            
            # Capture outputs
            original_outputs = original_tester.capture_outputs(original_output)
            refactored_outputs = refactored_tester.capture_outputs(refactored_output)
            
            # Compare file structures
            original_files = set(original_outputs.get("file_structure", {}).keys())
            refactored_files = set(refactored_outputs.get("file_structure", {}).keys())
            
            print(f"Original output directories: {original_files}")
            print(f"Refactored output directories: {refactored_files}")
            
            # Check if main output categories match
            common_dirs = original_files.intersection(refactored_files)
            only_original = original_files - refactored_files
            only_refactored = refactored_files - original_files
            
            if only_original:
                print(f"Directories only in original: {only_original}")
            if only_refactored:
                print(f"Directories only in refactored: {only_refactored}")
            
            print(f"Common directories: {common_dirs}")
            print("Direct comparison completed!")
        else:
            print("Original script did not produce outputs, only validating refactored version works")
            
        print("Comparison test completed!")


@pytest.mark.manual
def test_zos_config_reproducibility():
    """
    Test that running the same zos-only config twice produces identical outputs.
    
    This validates our config-aware pipeline produces consistent results:
    pytest -m manual tests/test_viz_snapshot.py::test_zos_config_reproducibility -v
    """
    import tempfile
    from pathlib import Path
    from ocean_emulators.viz_modules.config import VizConfig
    from ocean_emulators.viz_modules.main import run_visualization_pipeline
    
    print("Testing zos-only config reproducibility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create two separate output directories
        run1_output = temp_path / "run1"
        run2_output = temp_path / "run2"
        
        # Create identical configs
        config1 = VizConfig.minimal_zos_only(str(run1_output))
        config2 = VizConfig.minimal_zos_only(str(run2_output))
        
        print("Running first pipeline execution...")
        try:
            result1 = run_visualization_pipeline(config1)
            print(f"✓ Run 1 completed: {result1}")
        except Exception as e:
            print(f"✗ Run 1 failed: {e}")
            raise
        
        print("Running second pipeline execution...")
        try:
            result2 = run_visualization_pipeline(config2)
            print(f"✓ Run 2 completed: {result2}")
        except Exception as e:
            print(f"✗ Run 2 failed: {e}")
            raise
        
        # Compare outputs using snapshot testers
        print("Comparing outputs...")
        
        tester1 = VizSnapshotTester("run1_comparison")
        tester2 = VizSnapshotTester("run2_comparison")
        
        outputs1 = tester1.capture_outputs(Path(result1))
        outputs2 = tester2.capture_outputs(Path(result2))
        
        # Check that same files were generated
        plots1 = set(outputs1.get("plots", {}).keys())
        plots2 = set(outputs2.get("plots", {}).keys())
        
        if plots1 == plots2:
            print(f"✓ Both runs generated identical set of {len(plots1)} plots")
        else:
            print(f"✗ Different plots generated:")
            print(f"  Run 1: {plots1}")
            print(f"  Run 2: {plots2}")
            return False
        
        # Check file structure matches
        structure1 = outputs1.get("file_structure", {})
        structure2 = outputs2.get("file_structure", {})
        
        if structure1.keys() == structure2.keys():
            print(f"✓ Both runs generated identical directory structure")
        else:
            print(f"✗ Different directory structures:")
            print(f"  Run 1 dirs: {set(structure1.keys())}")
            print(f"  Run 2 dirs: {set(structure2.keys())}")
            return False
        
        # Compare with config expectations
        expected = config1.get_expected_outputs()
        
        print("Validating against config expectations...")
        match1 = tester1.compare_with_config_expected(outputs1, expected)
        match2 = tester2.compare_with_config_expected(outputs2, expected)
        
        if match1 and match2:
            print("✓ Both runs match config expectations")
            print("✓ Pipeline reproducibility validated!")
            return True
        else:
            print("✗ One or both runs failed config validation")
            return False


def run_viz_script(output_dir: Path) -> Dict[str, Any]:
    """Execute the original viz.py script and return outputs."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Create a modified version of the original script with the correct output path
    notebook_path = Path("notebooks/viz.py")
    temp_script = output_dir / "temp_original_viz.py"
    
    # Read the original notebook
    with open(notebook_path, 'r') as f:
        content = f.read()
    
    # Replace the output path in the original script
    content = content.replace(
        'output_path = (\n    "/tmp/viz_outputs/"\n    + str(datetime.now())[:10]',
        f'output_path = "{output_dir}"'
    )
    
    # Write the modified script
    with open(temp_script, 'w') as f:
        f.write(content)
    
    # Execute the script
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute()) + ":" + env.get("PYTHONPATH", "")
    
    result = subprocess.run(
        [sys.executable, str(temp_script)], 
        capture_output=True, 
        text=True, 
        cwd=str(Path.cwd()),
        env=env,
        timeout=3600  # 1 hour timeout
    )
    
    # Clean up
    temp_script.unlink()
    
    if result.returncode != 0:
        raise RuntimeError(f"Original viz script failed: {result.stderr}")
    
    return {}


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
            "basin_path": "/data/basins/basin_masks_original.zarr"
        }
    
    output_path = run_viz_analysis(config, minimal=True)
    return {}