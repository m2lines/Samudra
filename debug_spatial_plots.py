#!/usr/bin/env python3
"""Debug spatial plot functions to identify issues."""

import sys
import os
import traceback
sys.path.insert(0, '/workspace/src')

def test_spatial_functions():
    """Test each spatial function individually to find issues."""
    
    print("🔍 Debugging Spatial Plot Functions")
    print("=" * 50)
    
    try:
        # Test basic imports first
        print("📦 Testing imports...")
        from ocean_emulators.viz_modules.spatial_viz import (
            create_missing_salinity_plots,
            create_missing_temperature_plots, 
            create_missing_temperature_profile_plots,
            generate_all_spatial_plots
        )
        print("✅ All spatial functions imported successfully")
        
        # Test function signatures and basic structure
        print("\n🔧 Testing function signatures...")
        
        import inspect
        
        # Check create_missing_salinity_plots
        sig = inspect.signature(create_missing_salinity_plots)
        print(f"✅ create_missing_salinity_plots signature: {sig}")
        
        # Check create_missing_temperature_plots  
        sig = inspect.signature(create_missing_temperature_plots)
        print(f"✅ create_missing_temperature_plots signature: {sig}")
        
        # Check create_missing_temperature_profile_plots
        sig = inspect.signature(create_missing_temperature_profile_plots)
        print(f"✅ create_missing_temperature_profile_plots signature: {sig}")
        
        print("\n🔍 Checking for potential issues in implementations...")
        
        # Look for common issues that could cause hanging
        
        # 1. Check remove_climatology import
        try:
            from ocean_emulators.viz_modules.spatial_viz import remove_climatology
            print("✅ remove_climatology function available")
        except ImportError as e:
            print(f"❌ remove_climatology import issue: {e}")
        
        # 2. Check linear_piecewise_scale
        try:
            from ocean_emulators.viz_modules.spatial_viz import linear_piecewise_scale
            print("✅ linear_piecewise_scale function available") 
        except ImportError as e:
            print(f"❌ linear_piecewise_scale import issue: {e}")
        
        # 3. Check ocean_temperature_profile
        try:
            from ocean_emulators.viz_modules.spatial_viz import ocean_temperature_profile
            print("✅ ocean_temperature_profile function available")
        except ImportError as e:
            print(f"❌ ocean_temperature_profile import issue: {e}")
        
        # 4. Check get_basin_datasets  
        try:
            from ocean_emulators.viz_modules.spatial_viz import get_basin_datasets
            print("✅ get_basin_datasets function available")
        except ImportError as e:
            print(f"❌ get_basin_datasets import issue: {e}")
        
        print("\n📋 Summary:")
        print("  All functions appear to be defined correctly")
        print("  Issue likely occurs during actual execution with real data")
        print("  Possible causes:")
        print("    - Infinite loop in computation")
        print("    - Memory exhaustion")
        print("    - Missing data variables")
        print("    - Coordinate/dimension mismatches")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spatial_functions()
    
    if success:
        print("\n💡 Next steps to debug:")
        print("  1. Check if issue is in generate_all_spatial_plots() call order")
        print("  2. Test with minimal/dummy data")
        print("  3. Add debugging prints to spatial_viz.py functions")
        print("  4. Run functions individually with real data")
    
    sys.exit(0 if success else 1)