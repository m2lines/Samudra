<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Contributing Guide

Everyone can contribute to Samudra, and we value everyone's contributions. There are several ways to help,
including:

* Reporting bugs or feature requests in our [issue tracker](https://github.com/m2lines/Samudra/issues).
* Contributing PRs to our [code base](https://github.com/m2lines/Samudra).
* Writing or editing documentation. (Yes, typo fixes are welcome!)

This project follows the [M2LInES _Code of Conduct_](https://m2lines.github.io/pages/code-of-conduct/).

## Contributing code with pull requests

<details>
<summary><strong>TL;DR</strong></summary>

```shell
git clone git@github.com:m2lines/Samudra.git
cd Samudra
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
   button on [the repository page](https://github.com/m2lines/Samudra).

2. Clone the repository (via [`ssh` recommended](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)!) and change into the root directory.
   ```shell
   # if you're using a fork, make sure to clone your fork's repo
   git clone https://github.com/m2lines/Samudra.git
   # preferred method, but requires setting up an ssh key with Github.
   git clone git@github.com:m2lines/Samudra.git
   # or, using the Github CLI
   gh repo clone m2lines/Samudra

   # then, change directory
   cd Samudra
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
   git remote add upstream https://github.com/m2lines/Samudra.git
   # via ssh
   git remote add upstream git@github.com:m2lines/Samudra.git
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

10. Celebrate submitting your patch to Samudra — well done!

## Running Samudra

Please review the [How to Run](run.md) guide to learn how to train, evaluate, and visualize our emulators. To learn
how to customize experiments and hyperparameters, please read our [configuration](config.md) guide.

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

## Testing Samudra

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
         Normalize.init_instance(...)
         assert ...
```

Or you can initialize them in a Generator-based fixture:


```python3
   @pytest.fixture()
   def my_fixture():
       with MultitonScope():
           Normalize.init_instance(...)
           yield

   def test_foo(my_fixture):
       assert ... # in this code, the Normalize instance is the one from my_fixture
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
uv run py-spy record --native -o profile.svg -- ./.venv/bin/python  src/samudra/train.py configs/samudra_vnext/train.yaml
```

[memray](https://github.com/bloomberg/memray), which captures peak memory usage:

```shell
uv run memray run src/samudra/train.py --config configs/samudra_vnext/train.yaml
uv run memray flamegraph path/to/memray-output.bin
```

And [scalene](https://github.com/joaomdmoura/scalene), which shows per-line python/native CPU usage,
memory usage and GPU (though the latter is a bit deceptive since it is async wrt the highlighted code).

```shell
uv run scalene src/samudra/train.py configs/samudra_vnext/train.yaml
```

### Profiling CUDA Memory

You can turn on profiling of CUDA memory by setting the `profiler.cuda_snapshot_frequency` to a non-None value
in the config. eg:

```shell
uv run memray run src/samudra/train.py --config configs/samudra_vnext/train.yaml --profiler.cuda_snapshot_frequency 10
```

This will take a snapshot of the CUDA memory every 10 batches in the output directory. These can be visualized with
https://docs.pytorch.org/memory_viz -- see https://pytorch.org/blog/understanding-gpu-memory-1/ for more details.

## Data Engineering

To learn about how to access or recreate our datasets, please review our [data guide](data.md).
