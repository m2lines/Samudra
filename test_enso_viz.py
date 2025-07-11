#!/usr/bin/env python3
"""Test ENSO visualization functionality."""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig
from ocean_emulators.viz_modules.main import run_visualization_pipeline

def test_enso_visualization():
    """Test ENSO visualization with a focused configuration."""
    print("🧪 Testing ENSO Visualization")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = VizConfig(
            output_path=temp_dir,
            analysis_groups={'enso'},  # Only ENSO analysis
            variables=['thetao']  # Minimal variables for speed
        )
        
        print(f"Testing analysis groups: {config.analysis_groups}")
        
        try:
            print("\n📊 Running ENSO-focused pipeline...")
            result_path = run_visualization_pipeline(config)
            
            print(f"✓ Pipeline completed successfully!")
            print(f"✓ Output saved to: {result_path}")
            
            # Check for ENSO plots
            print("\n🔍 Checking for ENSO visualization plots...")
            
            enso_path = os.path.join(temp_dir, "ENSO")
            
            # Check for the two missing plots we implemented
            climatology_plot = os.path.join(enso_path, "Climatology.png")
            nino_plot = os.path.join(enso_path, "Nino_Figure_Short_with_map_single.png")
            
            if os.path.exists(climatology_plot):
                print("✅ Found Climatology.png")
            else:
                print("❌ Missing Climatology.png")
                
            if os.path.exists(nino_plot):
                print("✅ Found Nino_Figure_Short_with_map_single.png")
            else:
                print("❌ Missing Nino_Figure_Short_with_map_single.png")
            
            # List all generated ENSO files
            print(f"\n📁 Generated ENSO files:")
            if os.path.exists(enso_path):
                for file in os.listdir(enso_path):
                    print(f"  • {file}")
            else:
                print("  No ENSO directory created")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_enso_visualization()
    exit(0 if success else 1)