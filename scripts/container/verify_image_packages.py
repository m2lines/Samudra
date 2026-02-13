#!/usr/bin/env python3
"""Verify image-provided package versions satisfy project requirements."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

DEFAULT_PACKAGES = ("torch", "torchvision", "flash-attn", "zarr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=list(DEFAULT_PACKAGES),
        help="Distributions expected to come from the base image",
    )
    return parser.parse_args()


def requirements_by_name(pyproject_path: Path) -> tuple[str, dict[str, SpecifierSet]]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    requires_python = project["requires-python"]

    req_strings: list[str] = list(project.get("dependencies", []))
    req_strings.extend(project.get("optional-dependencies", {}).get("cuda", []))

    requirements: dict[str, SpecifierSet] = {}
    for req_string in req_strings:
        req = Requirement(req_string)
        requirements[canonicalize_name(req.name)] = req.specifier

    return requires_python, requirements


def main() -> int:
    args = parse_args()
    requires_python, requirements = requirements_by_name(args.pyproject)

    current_python = ".".join(str(x) for x in sys.version_info[:3])
    python_ok = SpecifierSet(requires_python).contains(current_python, prereleases=True)
    print(
        f"python installed={current_python} requirement={requires_python} "
        f"status={'OK' if python_ok else 'MISMATCH'}"
    )

    ok = python_ok
    for package in args.packages:
        canonical = canonicalize_name(package)
        spec = requirements.get(canonical)

        if spec is None:
            print(f"{package} requirement=MISSING status=MISMATCH")
            ok = False
            continue

        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            print(f"{package} installed=MISSING requirement={spec} status=MISMATCH")
            ok = False
            continue

        matches = spec.contains(installed, prereleases=True)
        print(
            f"{package} installed={installed} requirement={spec} "
            f"status={'OK' if matches else 'MISMATCH'}"
        )
        ok = ok and matches

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
