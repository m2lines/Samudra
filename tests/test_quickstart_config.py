"""Smoke test for configs/quickstart/.

Verifies the quickstart Samudra config trains end-to-end against the mock-om4
fixture. Dates and data filenames are overridden via CLI:
  - The mock-om4 fixture covers 1975-08-05 .. 1976-03-31 (vs. the real OM4
    slice the notebook fetches at 1958-1961).
  - The fixture writes data.zarr / means.nc / stds.nc; clone_data.py produces
    OM4.zarr / OM4_means.zarr / OM4_stds.zarr.
  - The quickstart disables full rollout inference; the notebook does a
    one-step validation prediction after training.
"""

import json
import logging

import pytest

QUICKSTART_CONFIG = "quickstart/train.yaml"

_MOCK_OVERRIDES = [
    "--epochs",
    "1",
    "--save_freq",
    "1",
    "--train_time.start",
    "1975-08-15",
    "--train_time.end",
    "1975-09-25",
    "--val_time.start",
    "1975-10-20",
    "--val_time.end",
    "1975-11-20",
    # Re-point data filenames at what the mock-om4 fixture writes to disk.
    "--data.sources",
    json.dumps(
        [
            {
                "data_location": "data.zarr",
                "data_means_location": "means.nc",
                "data_stds_location": "stds.nc",
            }
        ]
    ),
]


@pytest.mark.parametrize(
    "data_source,config_name,extra_config_args",
    [("mock-om4", QUICKSTART_CONFIG, _MOCK_OVERRIDES)],
    indirect=True,
)
def test_quickstart__smoke(trainer_pair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair
    trainer.run()
