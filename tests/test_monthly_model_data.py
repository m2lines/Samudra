# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import zarr  # type: ignore[import-untyped]

from samudra.constants import build_om4_layout
from samudra.experiments.monthly_model_data import ModelMonths


def fixture(tmp_path):
    lat, lon = np.array([-20.0, 20.0]), np.array([0.5, 1.5, 2.5, 3.5])
    z = zarr.open_group(str(tmp_path / "CM4.zarr"), mode="w")

    def put(name, value):
        return z.create_dataset(name, data=np.asarray(value))

    put("lat", np.broadcast_to(lat[:, None], (2, 4)))
    put("lon", np.broadcast_to(lon[None], (2, 4)))
    put("dz", build_om4_layout().depth_thickness)
    put("wetmask", np.ones((19, 2, 4), dtype=bool))
    t = put("time", np.arange(1, 61) * 5)
    t.attrs.update(units="days since 0151-01-01", calendar="noleap")
    b = put("time_bnds", np.stack([np.arange(60) * 5, np.arange(1, 61) * 5], -1))
    b.attrs.update(units="days since 0151-01-01", calendar="noleap")
    for i, name in enumerate(["uo", "vo", "thetao", "so"]):
        put(name, np.full((60, 19, 2, 4), i + 1.0, dtype=np.float32))
    put("zos", np.full((60, 2, 4), 5.0, dtype=np.float32))
    for name in ["tauuo", "tauvo", "hfds"]:
        put(name, np.full((60, 2, 4), 11.0, dtype=np.float32))
    zarr.consolidate_metadata(z.store)
    for suffix, value in [("means", 1.0), ("stds", 2.0)]:
        s = zarr.open_group(str(tmp_path / f"OM4_{suffix}.zarr"), mode="w")
        for name in ["tauuo", "tauvo", "hfds"]:
            s.create_dataset(name, data=np.array(value))
        zarr.consolidate_metadata(s.store)
    ref = dict(
        names=np.array(
            [f"{v}_{i}" for v in ["uo", "vo", "thetao", "so"] for i in range(19)]
            + ["zos"]
        ),
        lat=lat,
        lon=lon,
        mean=np.ones(77),
        std=np.full(77, 2.0),
    )
    return z, ref


def test_cm4_uses_own_fields_forcings_and_native_month(tmp_path):
    _, ref = fixture(tmp_path)
    reader = ModelMonths(tmp_path / "CM4.zarr", "cm4", ref, tmp_path)
    sample = reader.sample(0, "cpu")
    assert sample["first"].startswith("0151-")
    assert sample["fields"].shape[2:] == (77, 2, 4)
    torch.testing.assert_close(sample["fields"][0, 0, 38], torch.ones(2, 4))
    torch.testing.assert_close(
        sample["forcing"], torch.full_like(sample["forcing"], 5.0)
    )
    assert sample["weights"].sum().item() == pytest.approx(1)
    assert sample["fields"].shape[1] == 19 + len(sample["weights"])


def test_cm4_rejects_missing_wet_chunk_and_changed_grid(tmp_path):
    z, ref = fixture(tmp_path)
    reader = ModelMonths(tmp_path / "CM4.zarr", "cm4", ref, tmp_path)
    z["thetao"][:] = np.nan
    with pytest.raises(ValueError, match="Nonfinite model target"):
        reader.sample(0, "cpu")
    ref["lat"] = ref["lat"] + 1
    with pytest.raises(AssertionError):
        ModelMonths(tmp_path / "CM4.zarr", "cm4", ref, tmp_path)


def test_packed_frames_match_native_and_detect_corruption(tmp_path):
    import hashlib
    import json

    z, ref = fixture(tmp_path)
    native = ModelMonths(tmp_path / "CM4.zarr", "cm4", ref, tmp_path)
    values = np.concatenate(
        [z[v][:] for v in ["uo", "vo", "thetao", "so"]] + [z["zos"][:][:, None]], axis=1
    )
    packed = tmp_path / "packed"
    packed.mkdir()
    np.save(packed / "frames.npy", values)
    proof = dict(
        source=str(tmp_path / "CM4.zarr"),
        source_metadata_sha256=hashlib.sha256(
            (tmp_path / "CM4.zarr" / ".zmetadata").read_bytes()
        ).hexdigest(),
        shape=list(values.shape),
        indices=list(range(len(values))),
        frame_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in values],
        verification="every packed frame read back and byte-hash matched to source",
    )
    (packed / "COMPLETE.json").write_text(json.dumps(proof))
    cache = ModelMonths(tmp_path / "CM4.zarr", "cm4", ref, tmp_path, packed=packed)
    a, b = native.sample(0, "cpu"), cache.sample(0, "cpu")
    for key in ["fields", "forcing", "contexts", "mask", "weights"]:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    modified = np.load(packed / "frames.npy", mmap_mode="r+")
    modified[:] = 100
    modified.flush()
    fresh = ModelMonths(tmp_path / "CM4.zarr", "cm4", ref, tmp_path, packed=packed)
    with pytest.raises(ValueError, match="checksum changed"):
        fresh.sample(0, "cpu")
