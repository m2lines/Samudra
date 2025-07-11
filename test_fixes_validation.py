#!/usr/bin/env python3
"""Test our specific fixes with a faster configuration."""

import sys
import os
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig
from ocean_emulators.viz_modules.main import run_visualization_pipeline
from tests.test_viz_snapshot import VizSnapshotTester

def test_our_fixes():
    """Test our specific fixes with a focused configuration."""
    print("🧪 Testing Our Implemented Fixes")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "fixes_test"
        
        # Create config that tests our specific fixes
        config = VizConfig(
            output_path=str(output_dir),
            analysis_groups={'timeseries', 'metrics'},  # Focus on our fixes
            variables=['thetao'],  # Single variable for speed
            timeseries_variables=['thetao']
        )
        
        print(f"Testing analysis groups: {config.analysis_groups}")
        print(f"Testing variables: {config.variables}")
        
        try:
            print("\n📊 Running focused pipeline...")
            result_path = run_visualization_pipeline(config)
            
            print(f"✓ Pipeline completed successfully!")
            print(f"✓ Output saved to: {result_path}")
            
            # Check for our specific fixes
            print("\n🔍 Checking for implemented fixes...")
            
            timeseries_path = output_dir / "Timeseries"
            metrics_path = output_dir / "Metrics"
            
            # Check for timeseries grid plots
            grid_plot_1 = timeseries_path / "temperature_timeseries_grid_shallow_both.png"
            grid_plot_2 = timeseries_path / "temp_timeseries_grid_shallow_skipped.png"
            
            if grid_plot_1.exists():
                print("✅ Found temperature_timeseries_grid_shallow_both.png")
            else:
                print("❌ Missing temperature_timeseries_grid_shallow_both.png")
                
            if grid_plot_2.exists():
                print("✅ Found temp_timeseries_grid_shallow_skipped.png")
            else:
                print("❌ Missing temp_timeseries_grid_shallow_skipped.png")
            
            # Check for metrics files
            metrics_files = [
                "thetao_mae_info.txt",
                "sst_mae_info.txt", 
                "salinity_deseasonalized_info.txt"
            ]
            
            for metrics_file in metrics_files:
                metrics_file_path = metrics_path / metrics_file
                if metrics_file_path.exists():
                    print(f"✅ Found {metrics_file}")
                    # Show content
                    with open(metrics_file_path) as f:
                        content = f.read().strip()
                        if content:
                            print(f"   Content: {content[:100]}...")
                        else:
                            print("   ⚠️  File is empty")
                else:
                    print(f"❌ Missing {metrics_file}")
            
            # List all generated files for inspection
            print(f"\n📁 Generated files:")
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    print(f"  {rel_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_our_fixes()
    exit(0 if success else 1)