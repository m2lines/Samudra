# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Named faults that a subsystem's properties must detect.

A fault is a plausible bug, often one we actually shipped, written as a source
substitution in one module. Each fault names the properties that should detect it.
A catalog is a project's list of faults, in a flat `*_faults.py` file that exports
`CATALOG`.
"""

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

SUFFIX = "_faults.py"


@dataclass(frozen=True)
class Fault:
    name: str
    module: str  # Importable name of the module the substitution applies to.
    before: str  # Source text that must occur exactly once in that module.
    after: str
    detectors: tuple[str, ...]  # Node IDs, relative to the project root.

    def __post_init__(self):
        if not self.detectors:
            raise ValueError(f"{self.name}: a fault needs at least one detector")


@dataclass(frozen=True)
class Catalog:
    project: Path  # Where pytest runs; detectors are relative to it.
    faults: tuple[Fault, ...]

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted({fault.module for fault in self.faults}))


def module_path(module: str) -> Path:
    """The file a module would be imported from, found without importing it."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError(f"Cannot locate module {module}")
    return Path(spec.origin).resolve()


def load_catalog(path: Path) -> Catalog:
    name = "fault_catalog_" + hashlib.sha256(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import catalog {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = module.CATALOG
    if not isinstance(catalog, Catalog) or not catalog.faults:
        raise ValueError(f"Expected a nonempty Catalog in {path}")
    names = [fault.name for fault in catalog.faults]
    if len(set(names)) != len(names):
        raise ValueError(f"Fault names must be unique in {path}")
    return catalog
