<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Contributing Guide

Everyone can contribute to Ocean Emulator, and we value everyone's contributions. There are several ways to help,
including:

* Reporting bugs or feature requests in our [issue tracker](https://github.com/Open-Athena/Ocean_Emulator/issues).
* Contributing PRs to our [code base](https://github.com/Open-Athena/Ocean_Emulator).
* Writing or editing documentation. (Yes, typo fixes are welcome!)

This project follows the [M2LInES _Code of Conduct_](https://m2lines.github.io/pages/code-of-conduct/).

## Contributing code with pull requests

<details>
<summary><strong>TL;DR</strong></summary>

```shell
git clone git@github.com:Open-Athena/Ocean_Emulator.git
cd Ocean_Emulator
uv sync --dev
source .venv/bin/activate
uvx pre-commit install
uvx pre-commit run --all-files # also creates config schemas for validation (see below)

# dev
uv run pytest -m "not manual and not cuda"
uv run pytest --benchmark-only --benchmark-autosave
uv run pytest-benchmark compare 0001 0002
./scripts/annotate_license.sh

# push new remote branch to make a PR
git push -u origin <feature-branch>

# sync branch
git pull origin main --rebase
git push --force-with-lease
```

</details>

1. (If you're not a core maintainer), please fork the repository by clicking the **Fork**
   button on [the repository page](https://github.com/Open-Athena/Ocean_Emulator).

2. Clone the repository (via [`ssh` recommended](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)!) and change into the root directory.
   ```shell
   # if you're using a fork, make sure to clone your fork's repo
   git clone https://github.com/Open-Athena/Ocean_Emulator.git
   # preferred method, but requires setting up an ssh key with Github.
   git clone git@github.com:Open-Athena/Ocean_Emulator.git
   # or, using the Github CLI
   gh repo clone Open-Athena/Ocean_Emulator

   # then, change directory
   cd Ocean_Emulator
   ```

3. Install developer dependencies using [`uv`](https://docs.astral.sh/uv/getting-started/):
   ```shell
   uv sync --dev
   ```
   Then, activate the environment that `uv` created:
   ```shell
   source .venv/bin/activate
   ```

4. (If forked) Add the original repository as an upstream remote, so you can sync your changes.
   ```shell
   # via http
   git remote add upstream https://github.com/Open-Athena/Ocean_Emulator.git
   # via ssh
   git remote add upstream git@github.com:Open-Athena/Ocean_Emulator.git
   ```

5. Check out feature branches where you will develop from:
   ```shell
   git checkout -b name-of-change
   ```

6. Perform project lifecycle routines as needed during development:
   ```shell
   # run tests
   uvx pytest -m "not manual and not cuda"
   ```

   For more details on how to run specific tests, please see the section below.

   To validate the PhysicsNeMo-based container locally:
   ```shell
   scripts/container/build_physicsnemo_25_11.sh
   scripts/container/run_cuda_tests_in_image.sh
   ```
   The corresponding CI workflow is `Container PhysicsNeMo 25.11` in
   `.github/workflows/container-physicsnemo.yml` (x86 build + smoke checks, publish, and containerized CPU/GPU tests).

   **Recommended**: For convenience, we've collected lint checks as a [pre-commit](https://pre-commit.com/)
   hook.

   To install the pre-commit hook (which will run before every commit), call:
   ```shell
   uvx pre-commit install
   ```

   To run all checks manually, you can run:
   ```shell
   # run against staged files
   uvx pre-commit run
   # run against all files in the project
   uvx pre-commit run --all-files
   ```

   If you want to commit _without_ running pre-commit checks, you're always free to use the `--no-verify` flag:
   ```shell
   git commit --no-verfiy -m "WIP"
   ```

   Sometimes, you may want to skip _just one check_, but run the rest of the pre-commit. You can accomplish this by
   setting an [environment variable](https://pre-commit.com/#temporarily-disabling-hooks):
   ```shell
   export SKIP=reuse
   uvx pre-commit run
   ```

   If you add new code or docs, our pre-commit check will insist that you add a proper license identifier. To do this
   automatically, please run:
   ```shell
   scripts/annotate_license.sh
   ```

   It's totally ok to make lots of small commits as you develop your feature! Please, make sure to
   write [commit messages](https://cbea.ms/git-commit/) along the way.

   Sometimes, you may change code that changes the project's performance characteristics. To measure the code's current
   local performance, you can run:

   ```shell
   uv run pytest --benchmark-only --benchmark-autosave
   ```

   This will output stats on all our current benchmarks performance in your terminal. To compare this performance to
   previous changes, run:
   ```shell
   uv run pytest-benchmark compare <id1> <id2>
   # For example, comparing change 1 to change 2:
   uv run pytest-benchmark compare 0001 0002
   ```

   Please see the _Benchmarks & Profiling_ section below for more details.

7. Before submitting a pull request, please sync with the main repo via rebase:
   ```shell
   git pull origin main --rebase
   # if working in a fork
   git pull upstream main --rebase
   ```

   If the rebase requires that you force push to your remote feature branch, [we recommend using `--force-with-lease`](https://stackoverflow.com/questions/52823692/git-push-force-with-lease-vs-force):
   ```shell
   git push --force-with-lease
   ```

8. Finally, when you're ready to submit a pull request — say, when all checks have passed — push your change on your
   development branch so you can create a pull request:
   ```shell
   git push -u origin name-of-change
   ```

9. Before you make the final merge, please make sure your commits are tidy and well-named. To do this, you can either
   use the **Squash and merge** button to commit (the default), or merge commits after you've cleaned up the commit
   history. For this, we recommend performing an [interactive rebase](https://about.gitlab.com/blog/2020/11/23/keep-git-history-clean-with-interactive-rebase/):
   ```shell
   git rebase -i <starting-commit>
   ```

10. Celebrate submitting your patch to Ocean Emulator — well done!

## Running Ocean Emulator

### Training the model

To train the model on a single GPU, you can run:

```bash
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
uv run -m ocean_emulators.train configs/samudra_om4/train.yaml --experiment.data_root $DATA_PATH --experiment.name <my-experiment-name>
```

Unless you override `--experiment.output_dir`, this will write to a `.LOCAL` directory.
You can run `uv run -m ocean_emulators.train --help` to see all the options available.

To train on multiple GPUs, you can use skypilot, `torchrun`, or SLURM.

#### SkyPilot

If you use a model that requires Flash Attention, make sure to install the `cuda` extra first, like so:

```bash
uv sync --extra cuda
```
Of course, this will only work on CUDA-enabled machines.

To run a remote training job with SkyPilot, use the following command:

```shell
# export WANDB_API_KEY=<my-key>  # Get your key at https://wandb.ai/authorize
uv run sky launch skypilot/train.sky.yaml  --env WANDB_API_KEY --env-file <my-vars>.env --env NAME <my-experiment-name> --env CONFIG configs/samudra_om4/train.yaml
```

Please read the docstring in the `train.sky.yaml` for more information.

#### torchrun

To use torchrun on a single host with 8 GPUs, use something like:

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 python -m ocean_emulators.train configs/samudra_om4/train.yaml --experiment.data_root $DATA_PATH
```

See the [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html) for other examples.

#### SLURM

For SLURM, you want to allocate the same number of tasks to a given node as you have allocated GPUs to that *node* (not task).
You want to avoid using `--gpus-per-task` or `--gpu-bind` as it restricts the GPU's visibility to a given task, which
prevents cross-GPU communication. So you want something like (for 2 nodes with 4 GPUs each):

```bash
srun --nodes=2 --ntasks-per-node=4 --gres=gpu:4 -- uv run python -m ocean_emulators.train configs/samudra_om4/train.yaml --experiment.data_root $DATA_PATH
```

Each task will see all GPUs on the node, but they know how to choose the correct one for their work.

To learn more about other datasets used during training, please see the _Data Engineering_ section below.

### Evaluating the model

```bash
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
# (then put a checkpoint of the model at path/to/checkpoint)
uv run -m ocean_emulators.eval configs/samudra_om4/eval.yaml --ckpt_path path/to/checkpoint --experiment.data_root $DATA_PATH --experiment.name <my-experiment-name>-eval
```

This produces a `predictions.zarr` file in the output directory (by default `.LOCAL`) with the rollout of the model.

You can run `uv run -m ocean_emulators.eval --help` to see all the options available.

To learn more about other datasets used during evaluation, please see the _Data Engineering_ section below.

To run a remote training job with SkyPilot, use the following command:

```shell
# export WANDB_API_KEY=<my-key>  # Get your key at https://wandb.ai/authorize
uv run sky launch skypilot/eval.sky.yaml  --env WANDB_API_KEY --env-file <my-vars>.env --env NAME <my-experiment-name>-eval --env CONFIG configs/samudra_om4/eval.yaml
```

Please read the `eval.sky.yaml` docstring for more information.

### Visualizing outputs from the model

```bash
uv run -m ocean_emulators.viz configs/viz_om4.yaml --data_root path/to/data --name <my-experiment-name>-viz --runs='[{"name": "my_experiment", "location": "path/to/<my-experiment-name>-eval/predictions.zarr"}]'
```

You can run `uv run -m ocean_emulators.viz --help` to see all the options available.

After making changes to the visualization code, you can run the following command to compare old and new plots:

```bash
uv run -m ocean_emulators.utils.compare path/to/old/viz path/to/new/viz
```

To run a remote viz job with SkyPilot, please use the following command:

```shell
# export WANDB_API_KEY=<my-key>  # Get your key at https://wandb.ai/authorize
uv run sky launch skypilot/viz.sky.yaml \
  --env WANDB_API_KEY \
  --env-file <my-vars>.env \
  --env NAME=<my-experiment-name>-viz \
  --env BASIN_PATH=basin_masks_original.zarr \
  --env RUNS='[{"name": "my_experiment", "location": "/inputs/<my-experiment-name>-eval/predictions.zarr"}]'

```

### Managing SkyPilot Clusters

All of the `sky launch` commands above will create a 1-node cluster with the needed
resources for that job. You can then run (or queue) additional jobs on that same cluster by passing
its name to `sky exec` commands:

```shell
uv run sky exec -c my-cluster-name skypilot/eval.sky.yaml ...
```

SkyPilot will complain if you try to use a cluster with the wrong resources for your job.
Note that we didn't use `sky launch` for this. The `launch` command sets up the cluster
from scratch again, which can break running jobs. Even when using `sky exec`, your local directory
is *immediately* copied up to the cluster which means other jobs running on it will
immediately see that new code. So, we recommend you not change code versions or other local
files before running another job.

When you're done with the cluster you can shut it down:

```shell
uv run sky down my-cluster-name
```

If you like, you can also have it automatically take itself down after it becomes idle:

```shell
# shut down after 30 minutes of idleness
uv run sky autostop --down my-cluster-name -i 30
```

See the [SkyPilot docs](https://docs.skypilot.co/) for more.

### FOMO Model

The FOMO model (in configs/fomo_om4/) requires Flash Attention. Make sure to install the `cuda` extra first, like so:

```bash
uv sync --extra cuda
```

Of course, this will only work on CUDA-enabled machines.

You can then train/eval/etc as described above using the `configs/fomo_om4/*` files.

## Configuration

### Configuration files

Configuration is defined by config.py and values are stored in YAML files within the `configs/`
directory. Configuration files can include other configuration files using the `!include` directive.

Each configuration file is associated with a Pydantic model -- you can generate JSON schemas
for them with `uv run src/ocean_emulators/config_schema.py` (which is run automatically in pre-commit).
To associate a configuration file with a Pydantic model, generate the JSON schema (if it doesn't
already exist) and then add this line to the top of the config file:

```yaml
# yaml-language-server: $schema=path/to/schema.json
```

This is what the `config_schema.py` script uses to determine which model to validate against,
and also enables autocomplete/type checking in VS Code via the [YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml).

### Command line configuration

The train and eval modules accept the configuration file as a positional argument.
You can override arbitraries keys on the command line -- see `--help` for details. When overriding
an object (as opposed to a single scalar value) via the command line, you can either supply JSON
like `--data '{"key": "value"}'` or a YAML file with a leading '@' symbol: `--data @configs/data/file.yaml`.

Training runs create a YAML file in the checkpoint directory with the final configuration used which
you can use to reproduce the run by passing to train e.g. `uv run -m ocean_emulators.train path/to/config.yaml`.

## VS Code Integration

If you're using VS Code, we recommend installing the `ruff` and `mypy` extensions. For the latter,
you'll want to configure it to use pyproject.toml, which you can do with a `.vscode/settings.json` file:

```json
{
    "mypy-type-checker.args": [
        "\"--config=pyproject.toml\""
    ]
}
```

## Testing Ocean Emulator

<details>
<summary><strong>TL;DR</strong></summary>

```bash
# local dev / CI
pytest -m "not manual and not cuda"
# with more CPU cores
pytest -m "not manual and not cuda" -n auto
# all manual tests
pytest -m manual
```

</details>

We use `pytest` as a test runner. All tests in this project have several [_marks_](https://docs.pytest.org/en/stable/how-to/mark.html)
that allow developers to control what tests are run locally. Two marks of particular interest are `cuda` and `manual`.

"cuda" tests are tests that require an NVIDIA GPU to run. If tests use the `device` fixture, then they'll automatically
be configured to run on both GPU and CPU simultaneously. Certain tests, however, can be marked with `@pytest.mark.cuda`
if they need to run on that hardware. To run CUDA-only tests, call:

```bash
pytest -m cuda
```
And to exclude all cuda-marked tests, run:

```bash
pytest -m "not cuda"
```

"manual" tests are not run in continuous integration (CI), but are useful checks during the development process. For
example, evaluating if two model weights are equal is marked `manual`. All manual tests can be run like so:

```bash
pytest -m manual
```

To exclude manual tests, run:

```bash
pytest -m "not manual"
```

**To run the same tests that are run in CI, please run the following:**

```bash
pytest -m "not manual and not cuda"
```

### Running tests in the PhysicsNeMo container

If you want to run tests using the PhysicsNeMo 25.11-based image, build it first:

```bash
BUILD_APPTAINER=0 scripts/container/build_physicsnemo_25_11.sh
```

Run the CPU test set (same marker expression as CPU CI) inside the built image:

```bash
docker run --rm \
  -v "$PWD":/repo \
  -w /workspace \
  ocean-emulator:physicsnemo-25.11 \
  bash -lc '. .venv/bin/activate && cd /repo && python -m pytest -m "not manual and not cuda"'
```

Run CUDA tests in the built image:

```bash
scripts/container/run_cuda_tests_in_image.sh
```

You can pass custom pytest arguments for CUDA runs, for example:

```bash
PYTEST_ARGS="-k test_trainer" scripts/container/run_cuda_tests_in_image.sh
```

### Testing with Multitons

We have a set of singletons in the code which use the "Multiton" helper to prevent tests from interfering with each other.
When writing tests, you can either:

```python3
   def test_foo():
      with MultitonScope():
         # set up whatever singletons you need
         Normalize.init_instance(…)
         assert …
```

Or you can initialize them in a Generator-based fixture:


```python3
   @pytest.fixture()
   def my_fixture():
       with MultitonScope():
           Normalize.init_instance(…)
           yield

   def test_foo(my_fixture):
       assert … # in this code, the Normalize instance is the one from my_fixture
```

### Preventing checking-in secrets

In our pre-commit check, we use a tool developed by Yelp that detects strings that look suspiciously like secrets and
raises alarms. If this is blocking your patch **and you've manually inspected the sources for secrets and vetted that
there are, in fact, none checked in**, the following command will regenerate a metadata file to pass this check:

```shell
uvx detect-secrets scan > .secrets.baseline
```

Please check in the baseline after generating.

## Benchmarking & Profiling

We use `pytest-benchmark` to measure performance regressions in this project. Our intentions are to cultivate a culture
of writing performant programs. To this end, we offer users the following tools:

To run local benchmarks and save their status locally (associated with the current commit), execute:

```shell
uv run pytest --benchmark-only --benchmark-autosave
```

To compare benchmark run 0001 to 0002, you can run:

```shell
uv run pytest-benchmark compare 0001 0002
```

Please check your local `.benchmarks/` directory to see other benchmarks runs for comparison.

To generate a histogram plot of several local benchmark runs, you may use the `--histogram=FILENAME-PREFIX` flag:

```shell
uv run pytest-benchmark compare 'Darwin-CPython-3.10-64bit/*' --histogram
```

Instead of merely benchmarking performance, sometimes you may want to inspect the details of how benchmarks run. This is
useful, for example, for white-box performance optimization. To collect a `cProfile` trace for each benchmark, run:

```shell
uv run pytest --benchmark-only --benchmark-cprofile="tottime_per" --benchmark-cprofile-dump
```

(Please consult [these docs](https://pytest-benchmark.readthedocs.io/en/latest/usage.html#:~:text=%2D%2Dbenchmark%2Dcprofile%3DCOLUMN)
to see all available values for the `--benchmark-cprofile` flag.)

This will generate a `.prof` file located in the `.` directory (by default). You can visualize this trace with `snakeviz`
like so:

```shell
uvx snakeviz <benchmark_path>.prof
```

### Profiling CPU Usage + Memory

We also have a few other profiling tools available in the environment, including:

[py-spy](https://github.com/benfred/py-spy), which captures python + native CPU usage:
```shell
uv run py-spy record --native -o profile.svg -- ./.venv/bin/python  src/ocean_emulators/train.py configs/samudra_vnext/train.yaml
```

[memray](https://github.com/bloomberg/memray), which captures peak memory usage:

```shell
uv run memray run src/ocean_emulators/train.py --config configs/samudra_vnext/train.yaml
uv run memray flamegraph path/to/memray-output.bin
```

And [scalene](https://github.com/joaomdmoura/scalene), which shows per-line python/native CPU usage,
memory usage and GPU (though the latter is a bit deceptive since it is async wrt the highlighted code).

```shell
uv run scalene src/ocean_emulators/train.py configs/samudra_vnext/train.yaml
```

### Profiling CUDA Memory

You can turn on profiling of CUDA memory by setting the `profiler.cuda_snapshot_frequency` to a non-None value
in the config. eg:

```shell
uv run memray run src/ocean_emulators/train.py --config configs/samudra_vnext/train.yaml --profiler.cuda_snapshot_frequency 10
```

This will take a snapshot of the CUDA memory every 10 batches in the output directory. These can be visualized with
https://docs.pytorch.org/memory_viz -- see https://pytorch.org/blog/understanding-gpu-memory-1/ for more details.

## Data Engineering

Here are a few notes on how to replicate the core datasets used in this emulator.

### Cloning Data

We've provided a script to clone training and evaluation data locally (or on the target machine).
This will download the coarse 1-degree data.

```shell
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
```

To use the experimental (compact) view of the dataset, you can use the `--compact_variables` flag:

```shell
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH --compact_variables
```

To see all available options for this script (for example, to set a different chunking scheme), please run:

```shell
uv run scripts/clone_data.py -h
```

If you would like to download the half-degree or other data, you will need an API key for the `emulators`
bucket. We recommend using [rclone](https://rclone.org/) for this.

```shell
# This will prompt you for the API key.
rclone config create nyu-osn s3 provider=Other endpoint=https://nyu1.osn.mghpcc.org/

# This will show you some top-level user directories
rclone lsf nyu-osn:emulators/

# This will copy down the half-degree data
rclone copy --progress --transfers=32 nyu-osn:emulators/jr7309/data/om4_halfdeg $DATA_PATH
```

### Regridding & pre-processing OM4 data

If you've downloaded the data as described above, the data was already preprocessed. If you'd like to run preprocessing
yourself, please read on below:

As of 2025-06-16, we perform these operations on top of Dask clusters inside notebooks, though this is likely to change
in the near future.

To spin up a coiled notebook, please do the following:

1. Make sure coiled in installed. If you've installed all the dev dependencies, then it should be in your local env.
   ```shell
   uv pip install coiled
   ```

2. Log in to coiled. This may require creating an account (recommended: sign in with a Google or Github SSO).
   ```shell
   coiled login
   ```

3. Connect to a cloud provider. Here are a few commands to set up the top three cloud providers:
   ```shell
   coiled setup aws
   coiled setup gcp
   coiled setup azure
   ```

4. Once you have coiled infra ready-to-go, spin up a notebook with a Pangeo docker image via the following script:
   ```shell
   ./scripts/coiled-pangeo-notebook
   # The script only has one command, which you could also just run directly:
   coiled notebook start --container pangeo/pangeo-notebook --disk-size 256
   ```
   > Note: Coiled _does_ come with a `--sync` flag that is supposed to replicate your local environment with the remote
   > notebook. I have found this doesn't work, likely, because my local environment contains a large data cache.
   > You may try running the above command with the `--sync` flag added after switching into the `./notebooks` directory
   > -- I bet that might work.

5. Last, but not least, copy the most up-to-date pre-processing notebook into the Jupyter Lab instance that just opened
   up. This can be found in the `./notebooks/` directory with the name "YYYY-MM-DD-data_preprocess.ipynb". We recommend
   selecting the latest / most recent version.
