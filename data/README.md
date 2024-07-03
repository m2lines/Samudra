# ocean_emulators
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/m2lines/ocean_emulators/main.svg)](https://results.pre-commit.ci/latest/github/m2lines/ocean_emulators/main)

## Producing Input and Prediction Datasets

### Input Datasets

#### QC
To ensure that an input dataset adheres to the most recent checks please always run the following before publishing/uploading:
```python
from ocean_emulators.preprocessing import input_data_test
ds = ...
input_data_test(ds, deep=True) # deep enables long-running tests that check for nan consistency on the entire dataset
```

### Prediction Datasets

#### Postprocessing Raw prediction output
We aim to not upload the 'raw prediction output. Instead use the postprocessor to create an xarray dataset that is formattes as much as possible as the input data:

```python
from ocean_emulators.postprocessing import post_processor
ds_input = ... # load datasets that was used for the training of this model
ds_truth = ds_input.isel(time=...) # select the timesteps that are considered the ground-truth to compare predictions against
ds_raw_prediction = ... # output from the inference/prediction
ds_prediction = post_processor(ds_raw_prediction, ds_truth)
```

#### QC
Before uploading please always run the most recent checks
```python
from ocean_emulators.postprocessing import prediction_data_test
ds_prediction = ... # see above
ds_truth = ...# see above
prediction_data_test(ds_prediction, ds_truth)
```

## Developing this package

Set up a fresh mamba environment (optional but recommended)

```bash
mamba create -n=ocean_emulators_dev python=3.10
mamba activate ocean_emulators_dev
pip install -e ".[dev]"
```

>[!NOTE]
> If you are using conda instead of mamba, you can replace `mamba` with `conda` above

Now install this package with all the developer extra dependencies

```
pip install -e ".[dev]"
```

Before you edit the code make sure all tests pass by running pytest from the root level of this repository
```
pytest
```

### Code Linting

We use [pre-commit.ci](https://results.pre-commit.ci/) to run the CI linting checks.

You can configure pre-commit to run locally on every commit like this:

```
pre-commit install
```

and if you want to run the linting manually do:

```
pre-commit run --all-files
```

>[!TIP]
> You can also commit and bypass these checks (not generally recommended)
> ```
> git commit -m "some message" --no-verify
> ```

## Developing the documentation page

Clone this repository and navigate into the `docs/` folder

Set up a new environment for the docs
```
mamba env create -f environment.yaml
mamba activate ocean_emulators_docs
```

Build the html docs with
```
jupyter-book build .
```

You can then look at them
```
open _build/html/index.html
```
