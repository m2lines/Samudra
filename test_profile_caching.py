#!/usr/bin/env python3
"""Test profile caching functionality."""

import tempfile
import time
import os
from src.ocean_emulators.viz_modules.main import OceanVisualizationPipeline
from src.ocean_emulators.viz_modules.config import VizConfig

def test_profile_caching():
    """Test that profile caching speeds up repeated runs."""
    
    # Use the test configuration
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = VizConfig(
            output_path=tmp_dir,
            analysis_groups={'timeseries'},  # Only timeseries to minimize test time
            variables=['thetao'],  # Just one variable for faster testing
            timeseries_variables=['thetao']
        )
        
        pipeline = OceanVisualizationPipeline(config)
        
        print("🧪 Testing Profile Caching")
        print("=" * 50)
        
        # Show initial cache status
        print("Initial cache status:")
        pipeline.show_cache_status()
        
        # Load data (same for both runs)
        print("\nLoading data...")
        pipeline.load_data()
        
        # First run: compute profiles from scratch
        print("\n📊 First run (computing profiles from scratch):")
        start_time = time.time()
        pipeline.compute_profiles(use_cache=False)  # Force no cache
        first_run_time = time.time() - start_time
        print(f"⏱️  First run took: {first_run_time:.2f} seconds")
        
        # Show cache status after first run
        print("\nCache status after first run:")
        pipeline.show_cache_status()
        
        # Second run: should use cache
        print("\n⚡ Second run (using cached profiles):")
        start_time = time.time()
        pipeline.compute_profiles(use_cache=True)
        second_run_time = time.time() - start_time
        print(f"⏱️  Second run took: {second_run_time:.2f} seconds")
        
        # Calculate speedup
        if second_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"\n🚀 Speedup: {speedup:.1f}x faster!")
            
            if speedup > 5:  # Should be much faster when using cache
                print("✅ Caching test PASSED - significant speedup achieved!")
                return True
            else:
                print("❌ Caching test FAILED - not enough speedup")
                return False
        else:
            print("❌ Caching test FAILED - second run time was 0")
            return False

if __name__ == "__main__":
    success = test_profile_caching()
    exit(0 if success else 1)