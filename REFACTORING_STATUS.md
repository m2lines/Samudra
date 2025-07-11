# Ocean Visualization Refactoring Status

## Overview

Refactored 4k-line monolithic `notebooks/viz.py` into modular architecture with 6 modules in `src/ocean_emulators/viz_modules/`. Full pipeline runs and generates outputs, validated against gold standard.

## Architecture

```
src/ocean_emulators/viz_modules/
├── config.py              # Pydantic-style configs (zos-only, timeseries, spatial, full, etc.)
├── data_processing.py      # Data loading & transformation  
├── analysis.py             # Statistical calculations & profiles
├── timeseries_viz.py       # Time series plotting (includes global plots)
├── spatial_viz.py          # Spatial maps & OHC analysis
└── main.py                 # Pipeline orchestration

src/ocean_emulators/viz.py  # CLI interface
tests/test_viz_snapshot.py  # Config-aware snapshot testing
```

## Key Improvements

- **Performance**: Spatial analysis 10x faster (no profile computation)
- **Modularity**: Each analysis group runs independently
- **Config system**: 7 configurations from minimal (3 plots) to full (90+ plots)
- **Testing**: Comprehensive snapshot validation with gold standard comparison

## Current Status

**Validation Results**: 73.5% match rate (83/113 outputs) vs gold standard at `/data/2025-06-10_samudra-recreate-paper-om4`

**Working modules:**
- Spatial analysis: Complete
- ENSO/climate indices: Complete  
- Timeseries: Mostly complete (added global plots)
- OHC: Complete (added missing timeseries plots)
- Metrics: Works but different file naming

**Recent fixes implemented:**
- Added `Global_Thetao_Timeseries.png` and `Global_Salinity_Timeseries.png`
- Added `OHC/OHC.png` and `OHC/OHC_ref0_noanomaly.png` 
- Added `compare_info.txt` file generation
- Added PDFs warning system

## Running Tests

```bash
# Individual modules (fast)
uv run --dev python test_spatial_module.py
uv run --dev python test_enso_module.py  
uv run --dev python test_ohc_fix.py

# Full validation (slow, 10+ minutes)
uv run --dev python test_full_pipeline.py

# Quick validation of fixes  
uv run --dev python test_improved_validation.py
```

## Missing Functionality

**High impact (for >90% match rate):**
- `temp_timeseries_grid_shallow_skipped.png` - Grid visualization of timeseries
- `temperature_timeseries_grid_shallow_both.png` - Another grid variant  
- Metrics file naming fixes (`sst_mae_info.txt`, `thetao_mae_info.txt`, etc.)

**Medium impact:**
- Snapshot time naming (t_298 vs t_299)
- `salinity_manuall_non_deseasonalized.png` 
- Various basin-specific plots

**Not implemented:**
- PDFs analysis module (warning in place)

## Next Steps

1. Implement missing timeseries grid plots for +2 outputs
2. Fix metrics file naming for +3-4 outputs  
3. Re-run full validation to confirm >90% match rate
4. Extract remaining hardcoded values to config (optional)

## Technical Notes

- Profile computation is expensive (~3 minutes) - only runs for timeseries/metrics
- pytest configured with 30-minute timeout for long tests
- Gold standard path: `/data/2025-06-10_samudra-recreate-paper-om4`
- Original viz.py preserved at `notebooks/viz.py` for reference
- Snapshot testing validates only expected outputs per config