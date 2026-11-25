<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

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
git clone --filter=blob:none git@github.com:m2lines/Samudra.git
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
   git clone --filter=blob:none https://github.com/m2lines/Samudra.git
   # preferred method, but requires setting up an ssh key with Github.
   git clone --filter=blob:none git@github.com:m2lines/Samudra.git
   # or, using the Github CLI
   gh repo clone m2lines/Samudra -- --filter=blob:none

   # then, change directory
   cd Samudra
   ```

   <details>
   <summary>Why <code>--filter=blob:none</code>?</summary>

   Old commits contain large files, mostly about 240 MB of evaluation GIFs that were later deleted, so a
   full clone downloads far more than the current code. A blobless clone still fetches every commit, so
   history, hashes, and <code>git log</code> work as usual, but it downloads old file contents only when you need
   them (for example, when you check out an old commit). We don't rewrite history to remove these files
   because W&B runs and data releases record commit hashes.

   </details>

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
   scripts/container/build_physicsnemo_26_05.sh
   scripts/container/run_cuda_tests_in_image.sh
   ```
   The corresponding CI workflow is `Container PhysicsNeMo 26.05` in
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

Please review the [How to Run](https://m2lines.github.io/Samudra/docs/run/) guide to learn how to train, evaluate, and visualize our emulators. To learn
how to customize experiments and hyperparameters, please read our [configuration](https://m2lines.github.io/Samudra/docs/config/) guide.

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

The test suite generates about 5 GB of reusable ocean data under `.data_cache` by default. To avoid creating a copy in
every Git worktree, set `SAMUDRA_TEST_DATA_CACHE` to a shared directory before running tests:

```bash
export SAMUDRA_TEST_DATA_CACHE="$HOME/Library/Caches/samudra/test-data"
pytest -m "not manual and not cuda"
```

Samudra adds a versioned subdirectory to this path and uses file locks so concurrent test runs can safely initialize the
shared data. The default remains the worktree-local `.data_cache` when the environment variable is unset. When a change
makes the cached test-data format incompatible, increment `TEST_DATA_CACHE_VERSION` in `tests/conftest.py`.

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

If you want to run tests using the PhysicsNeMo 26.05-based image, build it first:

```bash
BUILD_APPTAINER=0 scripts/container/build_physicsnemo_26_05.sh
```

Run the CPU test set (same marker expression as CPU CI) inside the built image:

```bash
docker run --rm \
  -v "$PWD":/repo \
  -w /workspace \
  ocean-emulator:physicsnemo-26.05 \
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

## Property tests and mutation checks

Most of our tests are ordinary unit tests with hand-picked inputs and expected outputs. A *property test* instead
states something that must hold for every input, such as "renaming a source file never changes the decoded values".
[Hypothesis](https://hypothesis.readthedocs.io/) then generates inputs to search for a counterexample. Each property
needs an *oracle*, an independent way to know the right answer: a small executable model of the system, a trusted
reader (such as opening the source NetCDF directly), or a
[metamorphic relation](https://www.hillelwayne.com/post/metamorphic-testing/) that predicts how the output changes
when the input changes.

This section covers writing, running, and reading property tests. The shared tooling lives in its own project,
`verification/`; its [README](../verification/README.md) explains how campaigns, the mutation gate, and triage work.

### Writing a property

Put property tests in flat files named `*_properties.py`: `tests/test_*_properties.py` for `samudra`, and
`data/tests/test_*_properties.py` for `ocean_preprocessing`. `tests/test_sampler_properties.py` is a compact example.
It simulates every distributed rank in one process and checks that the sampler covers the dataset and replays epoch
changes, with no model, GPU, or process group.

* Name the oracle in the test's docstring, so a reviewer can tell why the expected answer is right.
* Keep generated inputs small and the test deterministic. Mark tests that need a GPU or a human with `cuda` or
  `manual`; property runs exclude both.
* Draw randomness from Hypothesis, `random`, NumPy, or torch's global generator; Hypothesis seeds all four for each
  example, so a failure replays exactly. (`tests/conftest.py` calls `register_torch_random()` for torch; a new
  project whose properties use torch must too.) A generator you construct yourself without a drawn seed makes
  replays differ, and Hypothesis reports the test as `Flaky`.
* When a search finds a bug, fix it and pin the shrunk counterexample with `@example(...)`, so it runs on every PR.
* If a property encodes an invariant the team hasn't agreed on yet, mark it
  `@pytest.mark.invariant(status="proposed")`. Its failures still show up, but triage won't propose them as bug
  reports. Review a proposed property as carefully as production code, since a wrong property produces false bug
  reports.

### Running them

From the repository root:

```bash
SAMUDRA_HYPOTHESIS_PROFILE=property uv run pytest tests/test_*_properties.py verification/tests
```

The `property` profile runs 40 examples per test, as PR CI does. For a longer search, use
`SAMUDRA_HYPOTHESIS_PROFILE=soak`, with `SAMUDRA_HYPOTHESIS_EXAMPLES` setting the count (default 1,000). The data
properties run with the rest of the data suite; see `data/AGENTS.md`.

In CI, each PR runs every property with 40 examples, and the nightly `Property soak` workflow runs them with 1,000.

### Reading a soak report

Open the Actions tab, choose `Property soak`, and select a run. The run's Summary has three kinds of section:

1. **One line per campaign**, such as `data-pinned-pyarrow property campaign: counterexample`. A campaign is one
   project's suite in one environment: `pinned` uses locked dependencies, and `canary` floats within supported
   ranges. `passed` and `counterexample` are results. `invalid` or `timeout` means the run itself broke (collection
   error, crash, nothing ran), and the line gives the reason.
2. **The mutation gate's summary**, described below.
3. **Triage**, per campaign. Each failing property is listed either as "new, needs triage" or with links to matching
   open or closed issues. A run that never finished is flagged at the top, and its failures aren't triaged.

A failure in only the `canary` lane usually means an upstream release changed behavior. A failure in every lane is
more likely our bug.

To investigate a failure, download the campaign's artifact from the bottom of the run page; artifacts expire after
30 days. In `pytest.log`, find the failing test's `Failing test case` (older Hypothesis versions say
`Falsifying example`), and the `@reproduce_failure(...)` line Hypothesis prints after it. Add that decorator to the
test temporarily to replay the exact input locally. Then decide whether the code or the property is wrong. If it's
the code, fix it and pin the input with `@example`. If it's the property, fix the property. To file an issue for a
checked failure, run `python -m samudra_verification.triage <campaign directory> --repo m2lines/Samudra --run-url
<run url>`, which prints the issues it would file. `--apply` files every one of them, so check each failure it
lists, not only the one you investigated, before rerunning with `--apply`.
CI never files, reopens, or comments on issues itself.

### Mutation checks

A passing property only helps if it would fail on buggy code. A *fault* is a named, plausible bug, usually one we
actually shipped, written as a small source substitution, together with the properties that should catch it. The
`Property mutation gate` workflow plants each fault and checks that one of those properties fails.

To add one, append a `Fault` to the catalog next to the tests, a flat `*_faults.py` file such as
`tests/sampler_faults.py`. A new catalog in a directory CI already gates needs no CI changes. Add faults only for bugs that happened or nearly did; a new property doesn't
need one. Run the gate on a project's catalogs from the repository root, with a new output directory each time:

```bash
uv run python -m samudra_verification.gate tests --output /tmp/root-mutations   # root catalogs
```

The gate prints each trial's result, and `summary.md` lists every trial that failed it. A `survived` mutant means
none of its detectors noticed the fault: strengthen a property, or list the property that should catch it. An
`invalid` trial gives its reason, such as a dirty control or a mutant that is broken code. `data/AGENTS.md` has the
command for the data catalogs.

On every PR, CI checks that each fault's anchor still occurs exactly once, so a refactor that moves the code fails
there rather than in the nightly run. It runs a project's gate only when the PR changes a catalog, a fault's target
module, a detector's test file, the project's environment or conftest, or `verification/`. The nightly soak runs every
gate.

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
uv run pytest-benchmark compare 'Darwin-CPython-3.12-64bit/*' --histogram
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
uv run py-spy record --native -o profile.svg -- ./.venv/bin/python  src/samudra/train.py src/samudra/configs/samudra_vnext/train.yaml
```

[memray](https://github.com/bloomberg/memray), which captures peak memory usage:

```shell
uv run memray run src/samudra/train.py --config src/samudra/configs/samudra_vnext/train.yaml
uv run memray flamegraph path/to/memray-output.bin
```

And [scalene](https://github.com/joaomdmoura/scalene), which shows per-line python/native CPU usage,
memory usage and GPU (though the latter is a bit deceptive since it is async wrt the highlighted code).

```shell
uv run scalene src/samudra/train.py src/samudra/configs/samudra_vnext/train.yaml
```

### Profiling CUDA Memory

You can turn on profiling of CUDA memory by setting the `profiler.cuda_snapshot_frequency` to a non-None value
in the config. eg:

```shell
uv run memray run src/samudra/train.py --config src/samudra/configs/samudra_vnext/train.yaml --profiler.cuda_snapshot_frequency 10
```

This will take a snapshot of the CUDA memory every 10 batches in the output directory. These can be visualized with
https://docs.pytorch.org/memory_viz -- see https://pytorch.org/blog/understanding-gpu-memory-1/ for more details.

## Data Engineering

To learn about how to access or recreate our datasets, please review our [data guide](https://m2lines.github.io/Samudra/docs/data/).
