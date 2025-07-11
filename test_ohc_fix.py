#!/usr/bin/env python3
"""Test OHC plots implementation."""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig
from ocean_emulators.viz_modules.main import run_visualization_pipeline

def test_ohc_plots():
    """Test that OHC plots are generated."""
    print("Testing OHC plots implementation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "ohc_test"
        
        # Create minimal OHC config  
        config = VizConfig.minimal_ohc(str(output_dir))
        
        try:
            # Run the pipeline
            result_path = run_visualization_pipeline(config)
            
            output_path = Path(result_path)
            
            # Check for OHC files
            ohc_png = output_path / "OHC" / "OHC.png"
            ohc_ref_png = output_path / "OHC" / "OHC_ref0_noanomaly.png"
            compare_txt = output_path / "compare_info.txt"
            
            if ohc_png.exists():
                print("✓ OHC.png generated successfully")
            else:
                print("✗ OHC.png MISSING")
                
            if ohc_ref_png.exists():
                print("✓ OHC_ref0_noanomaly.png generated successfully") 
            else:
                print("✗ OHC_ref0_noanomaly.png MISSING")
                
            if compare_txt.exists():
                print("✓ compare_info.txt generated successfully")
            else:
                print("✗ compare_info.txt MISSING")
            
            # List all files for debugging
            all_files = list(output_path.rglob("*"))
            generated_files = [f for f in all_files if f.is_file()]
            print(f"\\nGenerated {len(generated_files)} files:")
            for f in sorted(generated_files):
                rel_path = f.relative_to(output_path)
                print(f"  - {rel_path}")
            
            if ohc_png.exists() and ohc_ref_png.exists() and compare_txt.exists():
                print("\\n✅ OHC plots test PASSED!")
                return True
            else:
                print("\\n❌ OHC plots test FAILED!")
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_ohc_plots()
    if not success:
        sys.exit(1)