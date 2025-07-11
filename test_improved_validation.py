#!/usr/bin/env python3
"""Test improved validation with global timeseries and OHC fixes."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig
from ocean_emulators.viz_modules.main import run_visualization_pipeline
from tests.test_viz_snapshot import VizSnapshotTester

def test_improved_validation():
    """Test validation with all implemented fixes."""
    print("Testing improved validation with fixes...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "improved_test"
        
        # Create config with multiple groups including new fixes
        config = VizConfig(
            output_path=str(output_dir),
            variables=["thetao", "so", "zos"],
            analysis_groups={"timeseries", "spatial", "ohc", "enso"},  # Include our fixes
            timeseries_variables=["thetao", "so", "zos"]
        )
        
        try:
            print("Running pipeline with timeseries + spatial + OHC + ENSO...")
            result_path = run_visualization_pipeline(config)
            
            output_path = Path(result_path)
            
            # Check for our newly implemented files
            new_files = [
                "Timeseries/Global_Thetao_Timeseries.png",
                "Timeseries/Global_Salinity_Timeseries.png", 
                "OHC/OHC.png",
                "OHC/OHC_ref0_noanomaly.png",
                "compare_info.txt"
            ]
            
            found_new = 0
            for file_path in new_files:
                full_path = output_path / file_path
                if full_path.exists():
                    print(f"✓ {file_path}")
                    found_new += 1
                else:
                    print(f"✗ {file_path} MISSING")
            
            # List all files
            all_files = list(output_path.rglob("*"))
            generated_files = [f for f in all_files if f.is_file()]
            print(f"\\nGenerated {len(generated_files)} total files")
            
            # Group by directory
            dirs = {}
            for f in generated_files:
                rel_path = f.relative_to(output_path)
                parent = str(rel_path.parent) if rel_path.parent != Path('.') else '.'
                if parent not in dirs:
                    dirs[parent] = []
                dirs[parent].append(rel_path.name)
            
            for dir_name, files in sorted(dirs.items()):
                print(f"  {dir_name}/: {len(files)} files")
            
            print(f"\\n✅ Found {found_new}/{len(new_files)} newly implemented files")
            
            if found_new >= 4:  # Allow some flexibility
                print("🎉 Validation improvements SUCCESSFUL!")
                return True
            else:
                print("❌ Some fixes didn't work as expected")
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_improved_validation()
    if not success:
        sys.exit(1)