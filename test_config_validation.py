#!/usr/bin/env python3
"""Quick validation test for the config system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocean_emulators.viz_modules.config import VizConfig

def test_config_creation():
    """Test that config objects can be created correctly."""
    print("Testing config creation...")
    
    # Test minimal zos config
    config = VizConfig.minimal_zos_only("/tmp/test_output")
    print(f"✓ Minimal zos config created")
    print(f"  - Variables: {config.variables}")
    print(f"  - Analysis groups: {config.analysis_groups}")
    print(f"  - Timeseries variables: {config.timeseries_variables}")
    
    # Test expected outputs
    expected = config.get_expected_outputs()
    print(f"✓ Expected outputs generated")
    print(f"  - Directories: {expected['directories']}")
    print(f"  - Plots: {expected['plots']}")
    
    # Test minimal timeseries config
    config2 = VizConfig.minimal_timeseries("/tmp/test_output2")
    print(f"✓ Minimal timeseries config created")
    print(f"  - Variables: {config2.variables}")
    print(f"  - Analysis groups: {config2.analysis_groups}")
    
    # Test full config
    config3 = VizConfig.full_analysis("/tmp/test_output3")
    print(f"✓ Full analysis config created")
    print(f"  - Variables: {config3.variables}")
    print(f"  - Analysis groups: {config3.analysis_groups}")
    
    print("\n✓ All config tests passed!")

if __name__ == "__main__":
    test_config_creation()