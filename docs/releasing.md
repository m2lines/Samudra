<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Releasing to PyPI

Samudra is published to [PyPI](https://pypi.org/project/samudra/) as a single
pure-Python wheel by the [`Release`](https://github.com/m2lines/Samudra/actions/workflows/release.yml)
workflow. It authenticates with [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored anywhere.

## Installing the package

Samudra itself is pure Python, so one universal wheel serves every platform.
PyTorch provides the optimized attention kernels used on GPUs.

```bash
# CPU (default)
uv add samudra
pip install samudra

# GPU — adds torchvision; PyTorch SDPA supplies optimized attention
uv add "samudra[cuda]"
pip install "samudra[cuda]"

# Latest nightly dev build
uv add samudra --prerelease=allow
pip install --pre samudra
```

The `cuda` extra does not compile attention extensions. PyTorch's SDPA dispatcher
selects an available optimized CUDA kernel at runtime, with no additional
FlashAttention package or CUDA build toolchain required by Samudra.

Installing exposes a `samudra` console command that mirrors the module entry
points, so you don't need a checkout to run a task:

```bash
# Bundled preset by name (the example configs ship in the wheel):
samudra train samudra_om4/train.yaml --experiment.data_root $DATA_PATH
# Or a path to your own YAML:
samudra eval  path/to/eval.yaml  --ckpt_path path/to/checkpoint
samudra viz   path/to/viz.yaml
```

The example presets under `src/samudra/configs/` are packaged with the wheel, so a
`CONFIG` argument may be either a bundled preset name (e.g. `samudra_om4/train.yaml`)
or a filesystem path — a real path always wins, and a bare name that isn't found on
disk falls back to the bundled preset. Test-fixture configs under `tests/configs/`
are dev-only and excluded from the wheel.

## How versions are cut

The version is owned by [setuptools-scm](https://setuptools-scm.readthedocs.io/):
there is **no** `version = "..."` field to maintain — a git tag *is* the version.
`[tool.setuptools_scm]` in `pyproject.toml` configures it, and `samudra.__version__`
is available at runtime. The release paths differ only in what version reaches
the build:

| Trigger | Mode | Version | Published? |
| --- | --- | --- | --- |
| Push a `v*` tag | `stable` | the tag, e.g. `v1.0.0` → `1.0.0` (setuptools-scm) | ✅ PyPI |
| Weekly `schedule` (Mon 06:00 UTC) | `nightly` | `<next-patch>.dev<YYYYMMDDhhmm>` | ✅ PyPI |
| `workflow_dispatch` → `nightly`/`stable` | as chosen | as above | ✅ PyPI |
| `workflow_dispatch` → `smoke` | `smoke` | `<next-patch>+smoke.<sha>` | ❌ build-only |
| Pull request touching the script/workflow | `smoke` | — | ❌ build-only |
| Local editable install (`uv sync`) | — | `<next-patch>.dev<N>` from git | n/a |

The scheduled dev release is weekly rather than daily — Samudra doesn't turn over
enough in a day to warrant one, and the timestamp still makes each build unique.
The `smoke` mode is build-only: it names what the build *does* (a no-publish
check), not how it's triggered — every mode, `smoke` included, can be started by
hand from **Actions → Release → Run workflow**.

On a tagged commit setuptools-scm derives the version straight from the tag. For
the two synthetic modes, `scripts/package.py` computes the version and hands it
to setuptools-scm via `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SAMUDRA` — it never
edits a tracked file. A nightly's base is one patch above the most recent `v*`
tag (or `fallback_version` before the first tag), and its UTC timestamp keeps
every nightly unique and PEP 440-ordered *above* the last release, so `--pre`
resolves them.

Every publish (stable **and** nightly) waits on the CPU test suite: the release
workflow calls `test.yml` as a required `test` job, so a red suite blocks the
upload. A tag push doesn't otherwise run the tests, which is why the release
workflow invokes them itself.

### Cutting a stable release

```bash
# Just tag and push — no version bump anywhere. Name the commit to tag
# (it defaults to HEAD); see the release checklist below.
git tag -a v0.0.1 <commit> -m "Initial Release"
git push origin v0.0.1
```

The tag push runs `resolve → test → build → publish`, uploading `samudra 0.0.1`
to PyPI. To dry-run first, use **Actions → Release → Run workflow → mode: smoke**;
that builds and runs `twine check` without publishing.

### Choosing a version

Until 1.0, the minor version marks breaking changes:

- **Minor** (`0.1.x` → `0.2.0`): a config field, CLI option or output format is
  removed or renamed, or a bundled preset's defaults change behaviour. Examples
  from 0.2.0: removing `data.hist` and the `movies` visualization step, and
  changing the defaults of the `samudra_om4` preset.
- **Patch** (`0.2.0` → `0.2.1`): fixes and additions that existing configs and
  code don't notice.

Aim to release whenever a user-facing fix or breaking change lands, and at least
monthly while `main` is active. Tagging is cheap, and the weekly nightlies are
not what `pip install samudra` installs.

### Release checklist

1. **Pick the commit.** Use a `main` commit whose CPU *and* GPU checks passed.
   The release workflow runs only the CPU tests before publishing.
2. **Test the exact package you will release**, installed in a clean
   environment. Get it from one of three places:

   - **The latest nightly**, if no file that goes into the wheel has changed
     since the nightly's commit. Check with:

     ```bash
     nightly_sha=$(gh run list -R m2lines/Samudra --workflow release.yml \
       --event schedule --limit 1 --json headSha --jq '.[0].headSha')
     git diff --name-only "$nightly_sha" <sha> -- \
       src pyproject.toml README.md LICENSE LICENSES
     ```

     If this prints nothing, the nightly is equivalent; set
     `PKG="samudra==<nightly version>"`. If it lists only `pyproject.toml`, the
     nightly still counts as long as the diff doesn't touch `[build-system]`,
     `[project]`, `[tool.setuptools]` or `[tool.setuptools_scm]`.
   - **A smoke build of the commit.** Run **Actions → Release → Run workflow →
     mode: smoke** on `main` while it still points at `<sha>` (the run summary
     shows the commit it built). Then download its wheel:

     ```bash
     gh run download <run-id> -R m2lines/Samudra -n samudra-dist -D /tmp/samudra-dist
     PKG=$(ls /tmp/samudra-dist/*.whl)
     ```

   - **A release candidate.** `git tag -a v0.2.0rc1 <sha> -m "..."` and pushing
     it publishes a pre-release to PyPI; set `PKG="samudra==0.2.0rc1"`.

   Then install and exercise it:

   ```bash
   uv venv -p 3.12 /tmp/samudra-rc
   uv pip install --refresh -p /tmp/samudra-rc/bin/python "$PKG"
   /tmp/samudra-rc/bin/python -c "import samudra; print(samudra.__version__)"
   /tmp/samudra-rc/bin/python -c "from samudra.config_base import resolve_config_path as r; assert r('samudra_om4/train.yaml').is_file()"
   /tmp/samudra-rc/bin/samudra --help
   ```

3. **Tag the selected commit and push.** Name the commit explicitly so the tag
   lands on the commit you checked, not on whatever `HEAD` is:

   ```bash
   git tag -a v<version> <sha> -m "Samudra v<version>"
   git push origin v<version>
   ```

   PyPI never accepts the same version twice, so a broken release is fixed with
   a new patch release, not a re-upload.
4. **Check the published package.** Repeat step 2's install and checks in a
   fresh environment with `PKG="samudra==<version>"`.
5. **Publish release notes** as a GitHub Release on the tag. Lead with what
   users must change when upgrading, then fixes, then new features, and link
   each item to its PR. `gh release create <tag> --generate-notes` lists the
   merged PRs to start from.

The `pypi-publish` environment has no required reviewers, so anyone who can
push a `v*` tag publishes. Maintainers can add a required reviewer under
**Settings → Environments → pypi-publish** to make each publish wait for
approval.
