# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subsystem-owned named faults and the property oracles assigned to them."""

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mutation:
    name: str
    before: str
    after: str
    property: str


@dataclass(frozen=True)
class Catalog:
    project: Path
    module: str
    target: Path
    properties: Path
    mutations: tuple[Mutation, ...]


def load_catalog(path: Path) -> Catalog:
    name = "mutation_catalog_" + hashlib.sha256(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import catalog {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = module.CATALOG
    if not isinstance(catalog, Catalog) or not catalog.mutations:
        raise ValueError(f"Expected a nonempty Catalog in {path}")
    return catalog
