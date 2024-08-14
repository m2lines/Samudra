"""This is s bit silly. It would be better to be able to create datasets from the schema dynamically, so the schema serve as source of truth, and when they are updated, the test data is too. But for now lets at least test that our test fixtures are compliant."""

from ocean_emulators.dataset_validation import (
    ds_processed_validate,
    ds_input_validate,
    ds_raw_prediction_validate,
    ds_prediction_validate,
)


def test_processed_data(processed_data):
    ds_processed_validate(processed_data)


def test_input_data(input_data):
    ds_input_validate(input_data)


def test_prediction_data(prediction_data):
    ds_prediction_validate(prediction_data)
