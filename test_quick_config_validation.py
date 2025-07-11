#!/usr/bin/env python3
"""Quick validation test for the config-aware comparison system."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tests.test_viz_snapshot import VizSnapshotTester
from ocean_emulators.viz_modules.config import VizConfig

def create_mock_outputs(output_dir: Path, config: VizConfig) -> None:
    """Create mock output files based on config expectations."""
    expected = config.get_expected_outputs()
    
    # Create directories
    for directory in expected["directories"]:
        dir_path = output_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create plot files
    for plot in expected["plots"]:
        plot_path = output_dir / plot
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a small dummy PNG file
        plot_path.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x02\x00\x01\x00\x00\x00\x00\x00\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x04\x00\x01\x00\x00\x00\x00\x00\x00\x05\x00\x01\x00\x00\x00\x00\x00\x00\x06\x00\x01\x00\x00\x00\x00\x00\x00\x07\x00\x01\x00\x00\x00\x00\x00\x00\x08\x00\x01\x00\x00\x00\x00\x00\x00\t\x00\x01\x00\x00\x00\x00\x00\x00\n\x00\x01\x00\x00\x00\x00\x00\x00\x0b\x00\x01\x00\x00\x00\x00\x00\x00\x0c\x00\x01\x00\x00\x00\x00\x00\x00\r\x00\x01\x00\x00\x00\x00\x00\x00\x0e\x00\x01\x00\x00\x00\x00\x00\x00\x0f\x00\x01\x00\x00\x00\x00\x00\x00\x10\x00\x01\x00\x00\x00\x00\x00\x00\x11\x00\x01\x00\x00\x00\x00\x00\x00\x12\x00\x01\x00\x00\x00\x00\x00\x00\x13\x00\x01\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82')

def test_config_aware_comparison():
    """Test that config-aware comparison works correctly."""
    print("Testing config-aware comparison logic...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test zos-only config
        config = VizConfig.minimal_zos_only(str(temp_path))
        expected_outputs = config.get_expected_outputs()
        
        print(f"Expected outputs for zos config:")
        print(f"  - Directories: {expected_outputs['directories']}")
        print(f"  - Plots: {expected_outputs['plots']}")
        
        # Create mock outputs matching the config
        create_mock_outputs(temp_path, config)
        
        # Use snapshot tester to capture and validate
        tester = VizSnapshotTester("config_test")
        outputs = tester.capture_outputs(temp_path)
        
        print(f"\\nActual captured outputs:")
        print(f"  - Directories: {list(outputs.get('file_structure', {}).keys())}")
        print(f"  - Plots: {list(outputs.get('plots', {}).keys())}")
        
        # Test config-aware comparison
        matches = tester.compare_with_config_expected(outputs, expected_outputs)
        
        if matches:
            print("✓ Config-aware comparison PASSED")
        else:
            print("✗ Config-aware comparison FAILED")
            return False
        
        # Test with extra files (should still pass)
        extra_plot = temp_path / "Timeseries" / "extra_plot.png"
        extra_plot.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x02\x00\x01\x00\x00\x00\x00\x00\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x04\x00\x01\x00\x00\x00\x00\x00\x00\x05\x00\x01\x00\x00\x00\x00\x00\x00\x06\x00\x01\x00\x00\x00\x00\x00\x00\x07\x00\x01\x00\x00\x00\x00\x00\x00\x08\x00\x01\x00\x00\x00\x00\x00\x00\t\x00\x01\x00\x00\x00\x00\x00\x00\n\x00\x01\x00\x00\x00\x00\x00\x00\x0b\x00\x01\x00\x00\x00\x00\x00\x00\x0c\x00\x01\x00\x00\x00\x00\x00\x00\r\x00\x01\x00\x00\x00\x00\x00\x00\x0e\x00\x01\x00\x00\x00\x00\x00\x00\x0f\x00\x01\x00\x00\x00\x00\x00\x00\x10\x00\x01\x00\x00\x00\x00\x00\x00\x11\x00\x01\x00\x00\x00\x00\x00\x00\x12\x00\x01\x00\x00\x00\x00\x00\x00\x13\x00\x01\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82')
        
        outputs_with_extra = tester.capture_outputs(temp_path)
        matches_with_extra = tester.compare_with_config_expected(outputs_with_extra, expected_outputs)
        
        if matches_with_extra:
            print("✓ Config-aware comparison with extra files PASSED")
        else:
            print("✗ Config-aware comparison with extra files FAILED")
            return False
        
        return True

def test_multiple_configs():
    """Test multiple config types."""
    print("\\nTesting multiple config types...")
    
    configs = {
        "zos_only": VizConfig.minimal_zos_only("/tmp/test1"),
        "timeseries": VizConfig.minimal_timeseries("/tmp/test2"),
        "full": VizConfig.full_analysis("/tmp/test3")
    }
    
    for name, config in configs.items():
        expected = config.get_expected_outputs()
        print(f"\\n{name.upper()} config:")
        print(f"  - Variables: {config.variables}")
        print(f"  - Analysis groups: {config.analysis_groups}")
        print(f"  - Expected plots: {len(expected['plots'])}")
        print(f"  - Expected directories: {len(expected['directories'])}")
    
    return True

if __name__ == "__main__":
    print("Running quick config validation tests...")
    
    try:
        # Test config-aware comparison
        success1 = test_config_aware_comparison()
        
        # Test multiple configs
        success2 = test_multiple_configs()
        
        if success1 and success2:
            print("\\n✓ All quick validation tests PASSED!")
            print("Config-aware comparison system is working correctly.")
        else:
            print("\\n✗ Some tests FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)