<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# samudra-verification

The shared tooling behind Samudra's property tests and mutation checks. The
[contributor guide](../docs/contributing.md#property-tests-and-mutation-checks) explains how to write and run
property tests. This README explains how the tooling works, for anyone maintaining it.

The project is published as `samudra-verification` and imported as `samudra_verification`. It depends only on
Hypothesis, pytest, and pytest-reportlog, so the `data/` environment can install it without the root package's torch
dependency. Root `uv sync --dev` installs it as a workspace member. `data/` installs it with
`pip install -e ./verification`; it isn't on PyPI.

## Modules

| Module | Role |
| --- | --- |
| `profiles` | Hypothesis profiles (`unit`, `property`, `soak`, `mutation`); `tag_project_items`, which each project's conftest calls to record a test's project and invariant status; and `register_torch_random`, which lets Hypothesis seed torch |
| `evidence` | Reads pytest-reportlog's event stream into per-test outcomes |
| `verdicts` | Decides whether a pytest session can be trusted, and what a trusted one means |
| `spec` | `CampaignSpec` and `Lane`: what identifies a campaign, written to `replay.json` |
| `campaign` | Command-line tool: runs one campaign and writes its artifacts |
| `faults` | `Fault` and `Catalog`, which projects declare in `*_faults.py` files |
| `gate` | Command-line tool: the mutation gate. Also the pytest plugin each trial loads |
| `triage` | Command-line tool: matches failing properties to GitHub issues |
| `artifacts` | Git revision and installed-package records |
| `records` | Strict JSON for the tools' frozen dataclasses: reading rejects missing, unknown and wrongly typed fields |

The tests in `tests/` are themselves property tests. They check, for example, that an empty or all-skipped run can't
pass, that a failing control is never counted as a kill, and that filing issues is idempotent.

## Campaigns

A *campaign* is one run of a project's property suite with a fixed seed and example budget, in one *lane*: `pinned`
(a lock file or pinned environment file) or `canary` (floating within supported version ranges).

| Workflow | When | Examples per property |
| --- | --- | --- |
| CPU tests (`test.yml`), in a separate step | every PR | 40 |
| Data tests (`data-tests.yaml`), with the rest of the data suite | every PR | 40 |
| `Property soak` | nightly at 09:37 UTC, and on demand | 1,000 |
| `Property mutation gate` | every PR checks anchors; a project's gate runs when the PR changes one of its catalogs' inputs (see below); every gate runs in the nightly soak | 40 per property in each trial |

The root CPU test step skips `*_properties.py`, and a separate step runs them with `verification/tests`, so each
property runs once per PR. The soak runs the data suite three times: twice in the `pinned` lane (`data/.test.yml`),
once for each pandas string storage backend, and once in the `canary` lane (`data/.test-canary.yml`), so upstream
breakage shows up early.

To reproduce a CI campaign, artifacts included, run from the repository root:

```bash
uv run --locked python -m samudra_verification.campaign \
  --project samudra --lane pinned --tests tests/test_*_properties.py verification/tests \
  --dependency-file uv.lock --output /tmp/samudra-properties --seed 95920261105 \
  -- -m "not manual and not cuda"
```

`--examples` defaults to 1,000, the soak's budget. Each run needs a new `--output` directory.

### Artifacts

Each campaign directory holds:

* `replay.json`: the `CampaignSpec`, with project, lane, seed, command, git revision, and dependency hash. With the
  same Hypothesis version, it reproduces the run.
* `dependency-input.txt` and `environment.txt`: the dependency file and the installed packages. Data CI also saves
  `conda-explicit.txt`, the exact Mamba solution.
* `pytest.log`, `pytest.jsonl` (pytest-reportlog's events), `result.json`, and `summary.md`.

Every session is first checked for trustworthiness: it finished, collected cleanly, had no setup or teardown errors,
and ran at least one test. An untrustworthy session is `invalid`, with the reason in `result.json`. A trustworthy
campaign either `passed` or found a `counterexample`. The check needs pytest's per-phase results and node IDs, which
pytest-reportlog's events keep and JUnit XML doesn't.

## The mutation gate

A *fault* is a named source substitution in one module, plus the properties (its *detectors*) that should catch it.
Each project lists its faults in a flat `*_faults.py` file that exports a `Catalog`. The gate runs *trials*, each a
fresh pytest process under one condition:

1. A *control* trial for each seed runs every detector on unmodified code and must be clean. A dirty control stops
   the gate, so a bug that already exists is never credited as a kill.
2. For each fault, the gate finds its source anchor, which must occur exactly once, so a refactor that moves the code
   fails loudly. It writes the mutant source and runs one *mutant* trial per seed with that fault's detectors. Each
   trial process executes the mutant into the imported module. Production files are never modified.
3. Trials use the `mutation` profile: 40 examples, with explicit `@example`s, and with no database, no shrinking,
   and no deadline. Each trial is capped at `--timeout` seconds per property it runs. The mutant trials of a catalog
   also share `--budget` seconds, which start once its controls finish.
4. Only an oracle failure kills a mutant: an assertion, `pytest.raises` not raising, or the code under test
   crashing. Hypothesis's own errors (failed health checks, Unsatisfiable, Flaky) and errors showing the mutant is
   broken code (`NameError`, `ImportError`, `SyntaxError`) make the trial invalid. So do setup or collection errors,
   skips, timeouts, and any drift from the expected settings, source hashes, or collected tests. Invalid trials and
   survivors fail the gate.

Each trial records its verdict and reason in `results.json`.

`gate --plan <directory>` runs nothing: it checks every anchor and prints each catalog's inputs (the catalog, its
target modules, and its detectors' test files). The workflow's `plan` job runs it for `tests` and `data/tests` on
every PR, and starts a project's gate job when the PR changes one of those inputs, the project's environment or
conftest, or `verification/`. A new project needs one line in that job's project list. `data/AGENTS.md` has the command for the data catalogs.

## Triage

`python -m samudra_verification.triage <campaign directory>...` reads each campaign's failing properties and gives
each one an identity of project and node ID, so different counterexamples for one property map to one issue.
Failures of `@pytest.mark.invariant(status="proposed")` properties are excluded.

* With `--triage-output <dir>`, it looks up matching open and closed issues (read-only) and writes a JSON and
  Markdown summary per campaign. The nightly soak does this.
* With `--apply`, it files an issue for each property that has none. It never reopens a closed issue, and it refuses
  to file from an incomplete run. No workflow has permission to file issues.
