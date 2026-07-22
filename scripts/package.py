# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []  # stdlib + git only; the build itself shells out to `uv`.
# ///
"""Build the samudra wheel + sdist for PyPI publication.

Samudra is pure Python, so the build produces a single universal
``samudra-<version>-py3-none-any.whl`` that serves both CPU and GPU users. The
GPU custom kernels (flash-attn, flash-perceiver, torchvision) live in the
``cuda`` optional extra and compile on the user's machine at install time --
nothing GPU-specific goes into this wheel. That means there is exactly one
artifact per build regardless of hardware.

Versioning is owned by setuptools-scm (see ``[tool.setuptools_scm]`` in
pyproject.toml): a plain build derives the version from git tags. This script
only steps in for the two synthetic modes below, which it feeds to the build
via ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SAMUDRA`` -- so nothing here ever
mutates a tracked file.

Publication to PyPI is done by the release workflow
(.github/workflows/release.yml) via ``pypa/gh-action-pypi-publish`` with OIDC
trusted publishing -- this script never uploads anything and never needs a
token.

Three modes:
    nightly  -- version becomes <base>.dev<YYYYMMDDhhmm> (UTC), where <base> is
                one patch above the most recent v* tag. PEP 440 orders the dev
                release above that stable, so `pip install --pre samudra` / `uv`
                prefer it. The timestamp guarantees every nightly is unique even
                on days with no new commits.
    stable   -- version is taken verbatim from --version (the release workflow
                extracts it from the v<version> tag; setuptools-scm would derive
                the identical value from that same tag).
    manual   -- version becomes <base>+manual.<sha>; a build-only smoke. PyPI
                rejects local-version identifiers, so the workflow's publish job
                declines to run in this mode.

Usage:
    python scripts/package.py --mode nightly
    python scripts/package.py --mode stable --version 1.0.0
    python scripts/package.py --mode nightly --resolve-only
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
PYPI_PACKAGE = "samudra"

# setuptools-scm reads the build version from this env var (dist name uppercased
# with non-alphanumerics collapsed to underscores) and skips git entirely.
SCM_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SAMUDRA"

# Stands in for the last released version before the first v* tag exists (0.0.0
# = nothing released yet); dev builds then target 0.0.1. Keep in sync with
# `fallback_version` under [tool.setuptools_scm] so tagless dev installs and
# nightlies resolve to the same base.
FALLBACK_VERSION = "0.0.0"


# ---------- helpers ----------------------------------------------------------


def _check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        print(
            f"ERROR: '{name}' not found. Install with: {install_hint}", file=sys.stderr
        )
        sys.exit(1)


def _git_short_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT
    ).strip()


def _emit_github_output(key: str, value: str) -> None:
    """Append `key=value` to $GITHUB_OUTPUT when running under GitHub Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{key}={value}\n")


# ---------- version resolution -----------------------------------------------


def _bump_patch(version: str) -> str:
    """Return one patch above `version` (e.g. 1.0.0 -> 1.0.1)."""
    import re

    parts = [int(p) for p in re.split(r"[.\-+]", version) if p.isdigit()][:3]
    parts += [0] * (3 - len(parts))
    major, minor, patch = parts
    return f"{major}.{minor}.{patch + 1}"


def _last_release() -> str:
    """Most recent v* tag (without the leading 'v'), or FALLBACK_VERSION.

    FALLBACK_VERSION (0.0.0) stands in when no v* tag exists yet, so the first
    release the tree builds toward is 0.0.1.
    """
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0", "--match=v*"],
            text=True,
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return FALLBACK_VERSION
    return tag.lstrip("v")


def _release_base() -> str:
    """Next patch above the last release -- the version dev builds work toward.

    Mirrors setuptools-scm's guess-next-dev (one patch above the most recent
    tag), computed locally from git so the script carries no build-time
    dependency. It anchors the synthetic nightly/manual versions to the real
    release line without ever needing a hand-maintained version field.
    """
    return _bump_patch(_last_release())


def resolve_version(mode: str, explicit: str | None) -> str:
    """Return the build version for the requested mode.

    nightly -> <base>.dev<YYYYMMDDhhmm>
    stable  -> <explicit>
    manual  -> <base>+manual.<sha>
    """
    if mode == "stable":
        if not explicit:
            raise SystemExit("--version is required for --mode stable")
        return explicit
    if mode == "nightly":
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M")
        return f"{_release_base()}.dev{stamp}"
    if mode == "manual":
        return f"{_release_base()}+manual.{_git_short_sha()}"
    raise SystemExit(f"Unknown mode: {mode}")


# ---------- build ------------------------------------------------------------


def build_dist(version: str) -> None:
    """Build the samudra wheel + sdist into DIST_DIR, pinning setuptools-scm to `version`."""
    _check_tool("uv", "https://docs.astral.sh/uv/")

    # Start from a clean dist/ so stale artifacts never reach the publish step.
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir()

    # Hand the resolved version to setuptools-scm via env so it never touches
    # git (and no tracked file is mutated). Local, untagged dev builds that do
    # NOT set this fall back to setuptools-scm's own git-derived version.
    env = {**os.environ, SCM_ENV_VAR: version}
    print(f"\n--- Building {PYPI_PACKAGE} ({version}) ---")
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--sdist",
            "--out-dir",
            str(DIST_DIR),
            str(REPO_ROOT),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    wheels = sorted(DIST_DIR.glob("*.whl"))
    sdists = sorted(DIST_DIR.glob("*.tar.gz"))
    print(f"\nBuilt {len(wheels)} wheel(s) and {len(sdists)} sdist(s):")
    for f in (*wheels, *sdists):
        print(f"  {f.name}")
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly 1 wheel, got {len(wheels)}")
    if len(sdists) != 1:
        raise RuntimeError(f"Expected exactly 1 sdist, got {len(sdists)}")


# ---------- main -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=["nightly", "stable", "manual"], default="nightly"
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Explicit version. Required for --mode stable; when set for any mode it "
            "is used verbatim. CI's resolve job computes the version once and passes "
            "it here so the build job stamps the identical value."
        ),
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Print the resolved version and emit it to $GITHUB_OUTPUT; do not build.",
    )
    args = parser.parse_args()

    # A passed --version wins for every mode so the build job reuses the exact
    # value the resolve job advertised (a nightly straddling midnight UTC would
    # otherwise recompute a different timestamp here).
    version = args.version if args.version else resolve_version(args.mode, args.version)
    print(f"Mode:    {args.mode}\nVersion: {version}")
    _emit_github_output("version", version)

    if args.resolve_only:
        return

    build_dist(version)
    print(f"\nBuild complete. Wheel + sdist in {DIST_DIR}/")


if __name__ == "__main__":
    main()
