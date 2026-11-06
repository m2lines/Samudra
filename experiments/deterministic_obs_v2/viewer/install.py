# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Install a verified child catalog and atomically switch the viewer registry."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app-root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    root = a.app_root
    sys.path.insert(0, str(root))
    Catalog = importlib.import_module("data").Catalog

    child = root / ".data-deterministic-061126"
    registry = root / ".data-deterministic-061126-registry"
    live = root / ".data"
    old = live.resolve()
    old_hash = sha(old / "catalog.json")
    assert not child.exists() and not registry.exists()
    source = Catalog(a.source)
    expected = {"deterministic-C-061126", "deterministic-D-061126"}
    assert set(source.all_models) == expected
    before = Catalog(old)
    assert not expected.intersection(before.all_models)
    for name, info in source.meta["files"].items():
        assert sha(a.source / name) == info["sha256"], name
    shutil.copytree(a.source, child)
    for name, info in source.meta["files"].items():
        assert sha(child / name) == info["sha256"], name
    subprocess.run(["cp", "-al", str(old), str(registry)], check=True)
    meta = json.loads((registry / "catalog.json").read_text())
    relative = "../" + child.name
    meta.setdefault("extra_catalogs", []).append(relative)
    meta.setdefault("extra_catalog_hashes", {})[relative] = sha(child / "catalog.json")
    temp = registry / "catalog.json.new"
    temp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(temp, registry / "catalog.json")
    after = Catalog(registry)
    assert set(after.all_models) == set(before.all_models) | expected
    assert sha(old / "catalog.json") == old_hash and live.resolve() == old
    for name in before.meta["files"]:
        assert os.path.samefile(old / name, registry / name)
    for model in expected:
        c = after.for_model(model)
        for mode in [
            "surface",
            "interior",
            "monthly_interior",
            "forecast_interior",
            "heat",
        ]:
            assert c.origins_for(model, mode) == c.meta["origins"]
        for origin in c.meta["origins"]:
            r = c.meta["records"][origin]["models"][model]
            assert c.array(r["surface"]).shape == (73, 2, 180, 360)
            assert c.array(r["interior"]).shape == (19, 4, 180, 360)
            assert c.array(r["monthly_interior"]).shape == (12, 19, 2, 180, 360)
    link = root / ".data-deterministic-061126-link.new"
    link.symlink_to(registry.name)
    os.replace(link, live)
    receipt = dict(
        old=str(old),
        new=str(registry),
        old_models=len(before.all_models),
        new_models=len(after.all_models),
        added=sorted(expected),
        catalog_sha256=sha(child / "catalog.json"),
    )
    (root / ".data-deterministic-061126-installation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
