# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numbers

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.config import EvalConfig, TrainConfig
from samudra.eval import Eval
from samudra.train import Trainer
from samudra.utils.data import CanonicalSource, XarrayCanonicalReader
from samudra.utils.llc import canonicalize_llc_datasets
from samudra.utils.multiton import MultitonScope
from tests.llc_fixtures import coarse_llc_dataset, write_coarse_llc_dataset


def canonicalize(data, **kwargs):
    return canonicalize_llc_datasets(
        data,
        data,
        data,
        prognostic_vars_key=kwargs.get("prognostic", "all"),
        boundary_vars_key=kwargs.get("boundary", "all"),
    )


def test_compact_llc_channels_statistics_and_geography(tmp_path):
    original = write_coarse_llc_dataset(tmp_path)
    with xr.open_zarr(tmp_path / "data.zarr") as stored:
        data, means, stds, layout = canonicalize(stored)
        assert stored.U.dims == ("time", "k", "y", "x")
        assert "U_0" not in stored
        assert layout.grid_type == "gaussian"
        assert len(layout.prognostic_var_names) == 205
        assert "Eta" not in layout.boundary_var_names
        np.testing.assert_array_equal(data.lat, original.lat)
        np.testing.assert_array_equal(data.lon, original.lon)
        np.testing.assert_array_equal(data.areacello, original.rA)
        assert data.attrs == original.attrs
        for name in ("U", "V", "Theta", "Salt"):
            for level in (0, 10, 50):
                channel = f"{name}_{level}"
                assert data[name].dims == ("time", "lev", "lat", "lon")
                np.testing.assert_array_equal(
                    data[name].isel(lev=level), original[name].isel(k=level)
                )
                assert means[channel].ndim == stds[channel].ndim == 0
                assert means[channel].item() == original[f"{name}_mean"][level].item()
                assert stds[channel].item() == original[f"{name}_std"][level].item()
        for name in ["Eta", *layout.boundary_var_names]:
            np.testing.assert_array_equal(data[name], original[name])
            assert means[name].ndim == stds[name].ndim == 0
        assert "Depth" not in data.data_vars
        assert "repaired" not in data.data_vars


def test_llc_masks_use_wet_area_and_keep_coastal_cells():
    original = coarse_llc_dataset()
    data, _, _, _ = canonicalize(original)
    for level in range(51):
        np.testing.assert_array_equal(
            data[f"mask_{level}"], original.wet_area[level] > 0
        )
    assert not data.mask_0[0, 0].item()
    assert data.mask_0[1, 2].item()
    assert not data.mask_10[1, 2].item()
    assert 0 < original.land_frac[0, 0, 1].item() < 1
    assert data.mask_0[0, 1].item()


def test_llc_single_channel_selection_ignores_unused_fields():
    original = coarse_llc_dataset().drop_vars(["U", "U_mean", "U_std", "oceFWflx"])
    data, means, stds, _ = canonicalize(
        original, prognostic="single_1", boundary="single_1"
    )
    assert "Theta" in data and "oceQnet" in data
    assert "U" not in data and "Eta" not in data
    assert "U_0" not in means and "U_0" not in stds


@pytest.mark.parametrize("dimension", ["face", "i_g", "j_g"])
def test_llc_rejects_native_layout(dimension):
    with pytest.raises(ValueError, match="native face/tile"):
        canonicalize(coarse_llc_dataset().expand_dims({dimension: [0]}))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_llc_rejects_invalid_statistics(value):
    original = coarse_llc_dataset()
    original["U_std"][2] = value
    with pytest.raises(ValueError, match="U_std.*finite.*positive"):
        canonicalize(original)


def test_llc_rejects_missing_forcing():
    with pytest.raises(KeyError, match="oceFWflx"):
        canonicalize(coarse_llc_dataset().drop_vars("oceFWflx"))


def test_llc_rejects_wrong_depth_count():
    with pytest.raises(ValueError, match="51 depth levels"):
        canonicalize(coarse_llc_dataset().isel(k=slice(0, 19)))


def test_llc_rejects_statistics_with_spatial_dimensions():
    original = coarse_llc_dataset().drop_vars("U_mean")
    original["U_mean"] = (("k", "one"), np.ones((51, 1)))
    with pytest.raises(ValueError, match="U_mean must have dimensions"):
        canonicalize(original)


def test_llc_rejects_nonfinite_wet_area():
    original = coarse_llc_dataset()
    original["wet_area"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="wet_area must be finite"):
        canonicalize(original)


def test_llc_train_and_eval_smoke(tmp_path, backend):
    """Train on a compact store, reload the checkpoint, and verify a saved rollout."""
    physical = write_coarse_llc_dataset(tmp_path, n_time=12, ny=8, nx=8)
    values = TrainConfig.from_yaml("samudra_llc/train.yaml").model_dump(mode="json")
    values.update(
        backend=backend,
        epochs=1,
        batch_size=1,
        steps=[1],
        save_freq=1,
        rollout_validation={"model_steps": 2, "frequency": 1, "steps_forward": 1},
    )
    values["experiment"].update(
        name="train", base_output_dir=str(tmp_path), data_root=str(tmp_path)
    )
    values["experiment"]["wandb"]["mode"] = "disabled"
    values["model"]["unet"].update(ch_width=[2, 2], dilation=[1, 1], n_layers=[1, 1])
    values["model"].update(use_bfloat16=False, checkpointing=None)
    values["data"].update(
        input_steps=2,
        output_steps=1,
        loading={"type": "cpu", "num_workers": 0},
        inference_loading={"num_workers": 0},
    )
    values["data"]["sources"][0].update(
        data_location="data.zarr",
        data_means_location="data.zarr",
        data_stds_location="data.zarr",
        train_time={"start": "2011-09-13T00:00:00Z", "end": "2011-09-13T05:00:00Z"},
        val_time={"start": "2011-09-13T06:00:00Z", "end": "2011-09-13T11:00:00Z"},
        inference_time={"start": "2011-09-13T06:00:00Z", "end": "2011-09-13T11:00:00Z"},
    )
    with MultitonScope():
        trainer = Trainer(TrainConfig.model_validate(values))
        initial_weights = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.model.named_parameters()
        }
        assert len(trainer.prognostic_var_names) == 205
        assert len(trainer.boundary_var_names) == 4
        trainer.run()
        checkpoint_path = trainer.ckpt_paths.latest_checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["train_progress"]["optimizer_steps"] == 4
        assert np.isfinite(checkpoint["best_validation_score"])
        assert all(
            torch.isfinite(value).all() for value in checkpoint["model"].values()
        )
        assert any(
            not torch.equal(initial, checkpoint["model"][name])
            for name, initial in initial_weights.items()
        )

    evaluation = EvalConfig.model_validate(
        {
            "backend": backend,
            "save_zarr": True,
            "num_model_steps_forward": 2,
            "ckpt_path": str(checkpoint_path),
            "model": values["model"],
            "data": values["data"],
            "experiment": {**values["experiment"], "name": "eval"},
        }
    )
    with MultitonScope():
        metrics = Eval(evaluation).run()
    assert any("rmse" in name for name in metrics)
    assert all(
        np.isfinite(float(value))
        for value in metrics.values()
        if isinstance(value, numbers.Real)
    )
    with xr.open_zarr(
        evaluation.experiment.output_dir / "predictions.zarr"
    ) as predictions:
        for name in ("U", "V", "Theta", "Salt", "Eta"):
            wet = physical.wet_area.to_numpy() > 0
            if name == "Eta":
                wet = wet[0]
            output = predictions[name].to_numpy()
            assert np.isfinite(output[:, wet]).all(), name
            assert np.isnan(output[:, ~wet]).all(), name
        np.testing.assert_array_equal(
            predictions.time, physical.time.isel(time=slice(8, 12))
        )
        np.testing.assert_allclose(predictions.lat.isel(x=0), physical.lat)
        np.testing.assert_allclose(predictions.lon.isel(y=0), physical.lon)
        np.testing.assert_array_equal(predictions.areacello, physical.rA)


def test_compact_llc_reads_all_levels_once_without_dask(tmp_path, monkeypatch):
    from zarr.storage import DirectoryStore  # type: ignore[import-untyped]

    original = write_coarse_llc_dataset(tmp_path)
    reads = []
    getitem = DirectoryStore.__getitem__

    def counted_getitem(store, key):
        if key.split("/")[0] in {"U", "V", "Theta", "Salt"} and not key.split("/")[
            -1
        ].startswith("."):
            reads.append(key)
        return getitem(store, key)

    def forbid_chunking(*args, **kwargs):
        pytest.fail("The non-Dask reader must not turn on Dask")

    monkeypatch.setattr(DirectoryStore, "__getitem__", counted_getitem)
    monkeypatch.setattr(xr.Dataset, "chunk", forbid_chunking)
    monkeypatch.setattr(xr.DataArray, "chunk", forbid_chunking)
    with xr.open_zarr(tmp_path / "data.zarr", chunks=None) as stored:
        data, means, stds, layout = canonicalize(stored)
        source = CanonicalSource.from_datasets(
            data,
            means,
            stds,
            data_layout=layout,
            prognostic_var_names=layout.prognostic_var_names,
            boundary_var_names=layout.boundary_var_names,
        )
        assert reads == []
        assert isinstance(source.reader, XarrayCanonicalReader)
        indices = np.array([[3, 1], [3, 6]], dtype=np.int64)
        channels = tuple(
            f"{name}_{level}"
            for name in ("U", "V", "Theta", "Salt")
            for level in range(51)
        ) + ("Eta",)
        expected = np.stack(
            [
                original[name].values[indices, level]
                for name in ("U", "V", "Theta", "Salt")
                for level in range(51)
            ]
            + [original.Eta.values[indices]],
            axis=2,
        )
        np.testing.assert_array_equal(source.read(indices, channels), expected)
        assert sorted(reads) == sorted(
            f"{name}/{time}.0.0.0"
            for name in ("U", "V", "Theta", "Salt")
            for time in (1, 3, 6)
        )
        assert source.reader.data.U.chunks is None
        exported = source.to_xarray_dataset()
        assert "U_0" in exported and "U" not in exported
        np.testing.assert_array_equal(
            exported.U_50.isel(time=3), original.U.isel(time=3, k=50)
        )
