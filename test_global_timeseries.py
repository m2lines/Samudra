#!/usr/bin/env python3
"""Quick test for global timeseries functionality."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig
from ocean_emulators.viz_modules.main import run_visualization_pipeline

def test_global_timeseries():
    """Test that global timeseries plots are generated."""
    print("Testing global timeseries functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "timeseries_test"
        
        # Create timeseries-only config
        config = VizConfig.minimal_timeseries(str(output_dir))
        
        try:
            # Run the pipeline
            result_path = run_visualization_pipeline(config)
            
            output_path = Path(result_path)
            
            # Check for global timeseries files
            global_thetao = output_path / "Timeseries" / "Global_Thetao_Timeseries.png"
            global_salinity = output_path / "Timeseries" / "Global_Salinity_Timeseries.png"
            
            if global_thetao.exists():
                print("✓ Global_Thetao_Timeseries.png generated successfully")
            else:
                print("✗ Global_Thetao_Timeseries.png MISSING")
                
            if global_salinity.exists():
                print("✓ Global_Salinity_Timeseries.png generated successfully")
            else:
                print("✗ Global_Salinity_Timeseries.png MISSING")
            
            # List all files for debugging
            all_files = list(output_path.rglob("*.png"))
            print(f"\\nGenerated {len(all_files)} plot files:")
            for f in sorted(all_files):
                rel_path = f.relative_to(output_path)
                print(f"  - {rel_path}")
            
            if global_thetao.exists() and global_salinity.exists():
                print("\\n✅ Global timeseries test PASSED!")
                return True
            else:
                print("\\n❌ Global timeseries test FAILED!")
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_global_timeseries()
    if not success:
        sys.exit(1)