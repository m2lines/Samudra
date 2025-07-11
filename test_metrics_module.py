#!/usr/bin/env python3
"""Test metrics computation module."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig
from ocean_emulators.viz_modules.main import run_visualization_pipeline
from tests.test_viz_snapshot import VizSnapshotTester

def test_metrics_module():
    """Test metrics computation module with minimal config."""
    print("Testing metrics computation module...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "metrics_test"
        
        # Create minimal metrics config
        config = VizConfig.minimal_metrics(str(output_dir))
        
        print(f"Config: {config.analysis_groups}")
        print(f"Variables: {config.variables}")
        
        try:
            # Run the pipeline
            result_path = run_visualization_pipeline(config)
            
            print(f"✓ Metrics pipeline completed successfully")
            print(f"✓ Output saved to: {result_path}")
            
            # Check expected outputs using config-aware comparison
            expected_outputs = config.get_expected_outputs()
            output_path = Path(result_path)
            
            print(f"Expected outputs: {expected_outputs}")
            
            # Capture outputs using snapshot tester
            snapshot_tester = VizSnapshotTester("metrics_validation")
            outputs = snapshot_tester.capture_outputs(output_path)
            
            # List all actual outputs for debugging
            if output_path.exists():
                all_files = list(output_path.rglob("*"))
                print(f"✓ Total files generated: {len(all_files)}")
                for f in sorted(all_files):
                    if f.is_file():
                        rel_path = f.relative_to(output_path)
                        print(f"  - {rel_path}")
            
            # Use config-aware comparison
            matches = snapshot_tester.compare_with_config_expected(outputs, expected_outputs)
            
            if not matches:
                print("✗ Metrics config validation failed!")
                return False
            
            print("✓ Metrics computation module test PASSED!")
            return True
            
        except Exception as e:
            print(f"✗ Metrics pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_metrics_module()
    if not success:
        sys.exit(1)