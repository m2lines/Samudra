# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test a Samudra installed from PyPI with normal dependency resolution.

CI installs from ``uv.lock``, so it never sees the newest upstream releases that
``pip install samudra`` picks up. This script checks what a user gets: the
package imports, search's Parquet output works, bundled presets resolve, the
Colab quickstart config validates, and the public 2° OM4 store opens and loads.

It must run against an installed ``samudra`` (not the checkout's ``src/``), with
``nbformat`` available to load the quickstart config:

    python -I scripts/fresh_install_smoke.py
"""

import importlib
import importlib.metadata
import importlib.util
import math
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
QUICKSTART_BUILDER = SCRIPTS / "build_quickstart_notebook.py"


def _load_quickstart_yaml() -> str:
    """Return the YAML the Colab quickstart writes, from its generator script.

    Loaded by path because ``python -I`` keeps the script directory off
    ``sys.path``; the generator imports ``nbformat`` at module level.
    """
    spec = importlib.util.spec_from_file_location(
        "build_quickstart_notebook", QUICKSTART_BUILDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_quickstart_yaml()


def _check_parquet_round_trip() -> None:
    """Write and read back a small table the way ``samudra.search`` does.

    Search writes results.parquet and epochs.parquet through pandas, which
    imports pyarrow lazily, so importing modules alone can't catch a broken
    pyarrow (e.g. one that needs a newer NumPy than samudra allows).
    """
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        frame = pd.DataFrame({"candidate": ["a", "b"], "score": [0.5, 0.25]})
        parquet_path = Path(tmp) / "results.parquet"
        frame.to_parquet(parquet_path)
        pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
    print("Parquet round trip ok")


def main() -> None:
    import samudra

    package_file = Path(samudra.__file__).resolve()
    assert "site-packages" in package_file.parts, (
        f"samudra imported from {package_file}, not an installed distribution"
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("samudra", "xarray", "zarr", "numpy", "pandas", "torch")
    }
    print("Resolved versions:", ", ".join(f"{k} {v}" for k, v in versions.items()))

    for module in (
        "samudra.train",
        "samudra.eval",
        "samudra.stepper",
        "samudra.viz",
        "samudra.search",
    ):
        importlib.import_module(module)
    print("Imports ok")

    from samudra.config import TrainConfig
    from samudra.config_base import resolve_config_path

    preset = resolve_config_path("samudra_om4/train.yaml")
    assert preset.is_file(), f"bundled preset missing: {preset}"
    print(f"Bundled preset ok: {preset}")

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "quickstart.yaml"
        config_path.write_text(_load_quickstart_yaml())
        cfg = TrainConfig.from_yaml_and_cli([str(config_path)])
    from samudra.utils.location import S3Location

    source = cfg.data.sources[0]
    location = source.data_location
    assert isinstance(location, S3Location), location
    print(f"Quickstart config ok; opening {location}")

    # The step that failed for every fresh install of 0.2.0 (xarray 2026.9 with
    # zarr 2; fixed in #1017).
    ds = location.open()
    train = ds.sel(time=source.train_time.time_slice)
    lat_dim = "lat" if "lat" in ds.dims else "y"
    lon_dim = "lon" if "lon" in ds.dims else "x"
    assert (ds.sizes[lat_dim], ds.sizes[lon_dim]) == (90, 180), dict(ds.sizes)
    assert train.sizes["time"] == 17, dict(train.sizes)

    # Area-weighted: high-latitude cells are smaller. NaN (land) cells drop out
    # of both the sum and the weights. CMIP `zos` has its global mean removed.
    first_zos = train["zos"].isel(time=0)
    mean_zos = float(first_zos.weighted(ds["areacello"].fillna(0)).mean().load())
    assert math.isfinite(mean_zos) and abs(mean_zos) < 0.05, mean_zos
    print(f"Public OM4 store ok: area-weighted mean zos {mean_zos:.4f} m (~0)")
    ds.close()

    _check_parquet_round_trip()


if __name__ == "__main__":
    main()
