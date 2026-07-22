<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Releasing to PyPI

Samudra is published to [PyPI](https://pypi.org/project/samudra/) as a single
pure-Python wheel by the [`Release`](https://github.com/Open-Athena/Samudra/actions/workflows/release.yml)
workflow. It authenticates with [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored anywhere.

## Installing the package

Samudra itself is pure Python, so one universal wheel serves every platform.
The GPU custom kernels are opt-in.

```bash
# CPU (default) — everything except the compiled GPU kernels
uv add samudra
pip install samudra

# GPU — adds flash-attn, flash-perceiver, and torchvision, which compile
# against your local CUDA + torch at install time
uv add "samudra[cuda]"
pip install "samudra[cuda]"

# Latest nightly dev build
uv add samudra --prerelease=allow
pip install --pre samudra
```

The `cuda` extra builds native kernels, so it needs a CUDA toolchain and a
matching `torch` already present. With `uv` the `[tool.uv]` build settings in
`pyproject.toml` handle this automatically; with plain `pip` you typically want
`pip install --no-build-isolation "samudra[cuda]"` in an environment that
already has `torch`.

## How versions are cut

The version is owned by [setuptools-scm](https://setuptools-scm.readthedocs.io/):
there is **no** `version = "..."` field to maintain — a git tag *is* the version.
`[tool.setuptools_scm]` in `pyproject.toml` configures it, and `samudra.__version__`
is available at runtime. The release paths differ only in what version reaches
the build:

| Trigger | Mode | Version | Published? |
| --- | --- | --- | --- |
| Push a `v*` tag | `stable` | the tag, e.g. `v1.0.0` → `1.0.0` (setuptools-scm) | ✅ PyPI |
| Daily `schedule` (06:00 UTC) | `nightly` | `<next-patch>.dev<YYYYMMDDhhmm>` | ✅ PyPI |
| `workflow_dispatch` → `nightly`/`stable` | as chosen | as above | ✅ PyPI |
| `workflow_dispatch` → `manual` | `manual` | `<next-patch>+manual.<sha>` | ❌ build-only |
| Pull request touching the script/workflow | `manual` | — | ❌ build-only |
| Local editable install (`uv sync`) | — | `<next-patch>.dev<N>` from git | n/a |

On a tagged commit setuptools-scm derives the version straight from the tag. For
the two synthetic modes, `scripts/package.py` computes the version and hands it
to setuptools-scm via `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SAMUDRA` — it never
edits a tracked file. A nightly's base is one patch above the most recent `v*`
tag (or `fallback_version` before the first tag), and its UTC timestamp keeps
every nightly unique and PEP 440-ordered *above* the last release, so `--pre`
resolves them.

### Cutting a stable release

```bash
# Just tag and push — no version bump anywhere:
git tag v1.1.0
git push origin v1.1.0
```

The tag push runs `resolve → build → publish`, uploading `samudra 1.1.0` to
PyPI. To dry-run first, use **Actions → Release → Run workflow → mode: manual**;
that builds and runs `twine check` without publishing.

!!! note "Before the first tag"
    The repository has no `v*` tags yet, so builds fall back to `1.0.0`
    (`fallback_version` in `[tool.setuptools_scm]`, mirrored by `FALLBACK_VERSION`
    in `scripts/package.py` — keep the two in sync). Cutting `v1.0.0` makes the
    tag the single source of truth from then on.

## One-time trusted-publisher setup

Before the first publish, register the repository as a trusted publisher on
PyPI (a maintainer with project-owner rights does this once):

1. Create the project on PyPI, or use [pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   to reserve the name `samudra` before the first upload.
2. On the project's **Settings → Publishing** page, add a GitHub Actions
   publisher with:
     - **Owner**: `Open-Athena`
     - **Repository**: `Samudra`
     - **Workflow name**: `release.yml`
     - **Environment**: `pypi-publish`
3. In the GitHub repo, create an environment named `pypi-publish`
   (**Settings → Environments**). Optionally add required reviewers so stable
   releases need an approval before the publish job runs.

No secrets are needed — the publish job mints a short-lived OIDC token per run.
