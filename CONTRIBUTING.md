# Contributing Guide

Everyone can contribute to Ocean Emulator, and we value everyone's contributions. There are several ways to contribute,
including:

* Reporting bugs or feature requests in our [issue tracker](https://github.com/suryadheeshjith/Ocean_Emulator/issues).
* Contributing to our [code base](https://github.com/suryadheeshjith/Ocean_Emulator).
* Writing or editing documentation. (Yes, typo fixes are welcome!)

This project follows the [M2LInES _Code of Conduct_](https://m2lines.github.io/pages/code-of-conduct/).

## Contribute code with pull requests

<details>
<summary><strong>TL;DR</strong></summary>

```shell
git clone git@github.com:suryadheeshjith/Ocean_Emulator.git
cd Ocean_Emulator
uv sync --dev
source .venv/bin/activate
uvx pre-commit install
uvx pre-commit run --all-files # also creates config schemas for validation (see below)

# dev
uv run pytest -m "not manual and not cuda"
uv run pytest --benchmark-only --benchmark-autosave
uv run pytest-benchmark compare 0001 0002

# push new remote branch to make a PR
git push -u origin <feature-branch>

# sync branch
git pull origin main --rebase
git push --force-with-lease
```

</details>

1. (If you're not a core maintainer), please fork the repository by clicking the **Fork**
   button on [the repository page](https://github.com/suryadheeshjith/Ocean_Emulator).

2. Clone the repository (via [`ssh` recommended](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)!) and change into the root directory.
   ```shell
   # if you're using a fork, make sure to clone your fork's repo
   git clone https://github.com/suryadheeshjith/Ocean_Emulator.git
   # preferred method, but requires setting up an ssh key with Github.
   git clone git@github.com:suryadheeshjith/Ocean_Emulator.git
   # or, using the Github CLI
   gh repo clone suryadheeshjith/Ocean_Emulator

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
   git remote add upstream git@github.com:suryadheeshjith/Ocean_Emulator.git
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

   For more details on how to run specific tests, please see the next section.

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
   export SKIP=ruff
   uvx pre-commit run
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

10. Celebrating submitting your patch to Ocean Emulator — well done!

## Training the model

```bash
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
uv run -m ocean_emulators.train configs/train_om4.yaml --cluster_data_dir $DATA_PATH
```

You can run `uv run -m ocean_emulators.train --help` to see all the options available.

## Evaluating the model

```bash
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
# (then put a checkpoint of the model at path/to/checkpoint)
uv run -m ocean_emulators.eval configs/eval_om4.yaml --ckpt_path path/to/checkpoint --cluster_data_dir $DATA_PATH
```

You can run `uv run -m ocean_emulators.eval --help` to see all the options available.

## Configuration

### Configuration files

Configuration is defined by config.py and values are stored in YAML files within the configs/
directory. Configuration files can include other configuration files using the !include directive.

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

If you're using VS Code, we reccommend installing the `ruff` and `mypy` extensions. For the latter,
you'll want to configure it to use pyproject.toml, which you can do with a `.vscode/settings.json` file:

```json
{
    "mypy-type-checker.args": [
        "\"--config=pyproject.toml\""
    ]
}
```

## How to run tests

<details>
<summary><strong>TL;DR</strong></summary>

```bash
# local dev / CI
pytest -m "not manual and not cuda"
# with more CPU cores
pytest -m "not manual and not cuda" -n auto
# all manual tests
pytest -m manual
# just the model weights test
pytest -k "model and weights" --model1 <path/to/model1.pt> --model2 <path/to/model2.pt>
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

Note that for now you can't run both cuda and non-cuda tests together, see [#87](https://github.com/suryadheeshjith/Ocean_Emulator/issues/87).

"manual" tests are not run in continuous integration (CI), but are useful checks during the development process. For
example, evaluating if two model weights are equal is marked `manual`. All manual tests can be run like so:

```bash
pytest -m manual
```

To exclude manual tests, run:

```bash
pytest -m "not manual"
```

Usually, if a test is marked `manual`, it means that it should be run alone. Further, manual tests often need to be run
with specific arguments. To run a specific subsets of tests — or even a single test — at a time with arguments, please
do the following:
```bash
# single test
pytest -k "models_have_same_weights" --model1 <path/to/model1.pt> --model2 <path/to/model2.pt>
# test that match "model" and are manual
pytest -k "models" -m manual --model1 <path/to/model1.pt> --model2 <path/to/model2.pt>
# all model weights tests
pytest -l "models and weights" --model1 <path/to/model1.pt> --model2 <path/to/model2.pt>
```

**To run the same tests that are run in CI, please run the following:**

```bash
pytest -m "not manual and not cuda"
```

## Testing with Multitons

We have a set of singletons in the code which use the "Multiton" helper to prevent tests from interfering with each other.
When writing tests, you can either:

   def test_foo():
      with MultitonScope():
         # set up whatever singletons you need
         Normalize.init_instance(…)
         assert …

Or you can initialize them in a Generator-based fixture:

   @pytest.fixture()
   def my_fixture():
       with MultitonScope():
           Normalize.init_instance(…)
           yield

   def test_foo(my_fixture):
       assert … # in this code, the Normalize instance is the one from my_fixture

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

This will generate a `.prof` file located in the `` directory (by default). You can visualize this trace with `snakeviz`
like so:

```shell
uvx snakeviz <benchmark_path>.prof
```

We also have a few other profiling tools we use, including:

[py-spy](https://github.com/benfred/py-spy), which captures python + native CPU usage:
```shell
uvx py-spy record --native -o profile.svg -- ./.venv/bin/python  src/ocean_emulators/train.py configs/train_om4.yaml
```

[memray](https://github.com/bloomberg/memray), which captures peak memory usage:

```shell
uvx memray run src/ocean_emulators/train.py --config configs/train_om4.yaml
uvx memray flamegraph path/to/memray-output.bin
```

And [scalene](https://github.com/joaomdmoura/scalene), which captures python/native CPU usage,
memory usage and GPU (though the latter is a bit deceptive since it is async wrt the highlighted code).

```shell
uvx scalene src/ocean_emulators/train.py configs/train_om4.yaml
```
