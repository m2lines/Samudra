# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Run pinned CM4 quarter-degree preparation; promotion requires explicit gates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/scratch/jr7309/cm4-quarterdeg-20261023")
CODE = ROOT / "source"
INPUT = Path("/scratch/jr7309/cm4-pr928-smoke-20261015/inputs")
RAW = "s3://emulators/ai2_colab/2024-11-01-CM4-pre-industrial-control-simulation"
PILOT = "s3://m2lines-pubs/Samudra/experiments/cm4-quarterdeg-pilot-20261023/CM4.zarr"
FINAL = "s3://m2lines-pubs/Samudra/v2026-09/cm4_quarterdeg/CM4.zarr"
EXPECTED_VERSIONS = {
    "xarray": "2026.2.0",
    "xgcm": "0.9.0",
    "xesmf": "0.9.2",
    "dask": "2026.7.1",
    "distributed": "2026.7.1",
    "numpy": "2.4.6",
    "zarr": "2.18.2",
    "s3fs": "2026.6.0",
    "numcodecs": "0.15.1",
}


def filesystem(profile: str) -> Any:
    import s3fs

    return s3fs.S3FileSystem(profile=profile)


def record(name: str, payload: dict[str, Any]) -> None:
    (ROOT / "records" / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def run(command: list[str], *, profile: str) -> None:
    print("COMMAND", json.dumps(command), flush=True)
    subprocess.run(
        command,
        cwd=CODE / "data",
        env=dict(os.environ, AWS_PROFILE=profile),
        check=True,
    )


def prepare(pilot: bool) -> None:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    assert versions == EXPECTED_VERSIONS, versions
    if not pilot:
        gate = json.loads((ROOT / "records" / "pilot-qualified.json").read_text())
        assert gate["passed"] and gate["source"] == PILOT
        assert gate["processing_commit"] == (ROOT / "commit.txt").read_text().strip()
        assert gate["rust_loader_matches_xarray_exactly"]
    out = PILOT if pilot else FINAL
    fs = filesystem("cm4-publish")
    prefix = out.removeprefix("s3://").rsplit("/", 1)[0]
    if fs.exists(prefix):
        raise FileExistsError(f"Refusing to overwrite destination: {prefix}")
    inputs = json.loads((INPUT / "input_manifest.json").read_text())
    for key in ("supergrid", "quarterdeg"):
        item = inputs[key]
        with (INPUT / item["local_file"]).open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest()
                == item["staged_file_sha256"]
            )
    args = [
        sys.executable,
        "-u",
        "-m",
        "ocean_preprocessing",
        "cm4",
        RAW + "/ocean_5daily.zarr",
        RAW + "/ocean_static_no_mask_table.zarr",
        str(INPUT / "ocean_hgrid.nc"),
        str(INPUT / "gaussian_grid_720_by_1440.nc"),
        "--time_bounds_path=" + RAW + "/time_bnds.nc",
        "--output_path=" + out,
        "--output_profile=cm4-publish",
        "--compress",
        "--skip_spatial_filtering",
        "--skip_flattening",
        "--skip_validation",
        "--cluster=local",
        "--n_workers=" + ("8" if pilot else "32"),
        "--threads_per_worker=1",
        "--memory_limit=" + ("14GB" if pilot else "12GB"),
    ]
    if pilot:
        args.append("--small_run")
    started = time.time()
    run(args, profile="cm4-source")
    record(
        "pilot-preprocess" if pilot else "preprocess",
        {
            "commit": (ROOT / "commit.txt").read_text().strip(),
            "versions": versions,
            "job_id": os.environ["SLURM_JOB_ID"],
            "output": out,
            "command": args,
            "seconds": time.time() - started,
            "inputs": inputs,
        },
    )
    for suffix in ("_means.zarr", "_stds.zarr"):
        if fs.exists(out[:-5] + suffix):
            raise FileExistsError(out[:-5] + suffix)
    started = time.time()
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "ocean_preprocessing.make_norm_datasets",
            out,
            "--start",
            "0151-01-06",
            "--end",
            "0151-02-20" if pilot else "0351-01-01",
        ],
        profile="cm4-publish",
    )
    record(
        "pilot-norm" if pilot else "norm",
        {
            "seconds": time.time() - started,
            "job_id": os.environ["SLURM_JOB_ID"],
            "source": out,
        },
    )
    run(
        [
            sys.executable,
            "-u",
            str(CODE / "experiments/cm4-quarterdeg-preprocessing/audit.py"),
            "pilot" if pilot else "full",
        ],
        profile="cm4-publish",
    )

    if pilot:
        subprocess.run(
            [
                "bash",
                str(CODE / "experiments/cm4-quarterdeg-preprocessing/build-native.sh"),
            ],
            check=True,
        )
        local = ROOT / "pilot-data"
        local.mkdir(exist_ok=True)
        for name in ("CM4.zarr", "CM4_means.zarr", "CM4_stds.zarr"):
            fs.get(
                PILOT.rsplit("/", 1)[0].removeprefix("s3://") + "/" + name,
                str(local / name),
                recursive=True,
            )
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT / "native"),
        )
        subprocess.run(
            [
                "apptainer",
                "exec",
                "/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif",
                "/workspace/.venv/bin/python",
                "-u",
                str(CODE / "experiments/cm4-quarterdeg-preprocessing/check_rust.py"),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    assert sys.argv[1] in ("pilot", "full")
    prepare(sys.argv[1] == "pilot")
