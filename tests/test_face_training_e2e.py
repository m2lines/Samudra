"""A whole face advanced end to end, small enough to run on a CPU.

Every other test here covers one piece: the geometry, the blender, the reader,
the sharding, the denominators. This one runs the pieces together through the
real `Trainer` -- catalog, face context, group chunk reader, chunked
forward/backward, seam blend, replay write-back -- because the ways they fail
together are not the ways they fail apart. The face-scale version of this is
what caught the timestamp-decoding mismatch between the reader and
`DataSource`, and it needed 100 GB to find it; this needs a few hundred MB.

Shrunk 45x from the real geometry but structurally identical: 6x6 tiles of
`TILE + 2 * OVERLAP` over a square face, cut out of ONE packed cache with
`data.llc_tiles`, with land that differs tile to tile.
"""

import json
import logging

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import GradientLossConfig, TrainConfig
from ocean_emulators.tiling import face_tile_windows
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope

EXTENT = 96
TILE = 16
OVERLAP = 2
SIZE = TILE + 2 * OVERLAP  # 20, divisible by 4 for the test model's 2 stages
PROGNOSTIC = ["Theta_0", "Theta_1"]  # one 3D variable, one depth pair
BOUNDARY = ["oceQnet"]


def _write_face_cache(root, times: int = 240) -> None:
    """A packed cache shaped like the real face cache, 45x smaller."""
    time = xr.cftime_range("1975-08-05", periods=times, freq="6h", calendar="julian")
    generator = np.random.default_rng(0)
    shape = (times, len(PROGNOSTIC), EXTENT, EXTENT)
    prognostic = generator.standard_normal(shape).astype(np.float32)
    boundary = generator.standard_normal(
        (times, len(BOUNDARY), EXTENT, EXTENT)
    ).astype(np.float32)

    # Land that differs between tiles, which is the whole reason the loss
    # denominator has to come from the face rather than from a rank's batch.
    mask = np.ones((len(PROGNOSTIC), EXTENT, EXTENT), dtype=bool)
    mask[:, :TILE, :TILE] = False  # one corner tile entirely land
    mask[:, EXTENT // 2 :, EXTENT - TILE :] = False

    xarray = xr.Dataset(
        {
            "prognostic": (("time", "prognostic_channel", "y", "x"), prognostic),
            "boundary": (("time", "boundary_channel", "y", "x"), boundary),
            "prognostic_mean": (
                ("prognostic_channel",),
                np.zeros(len(PROGNOSTIC), dtype=np.float32),
            ),
            "prognostic_std": (
                ("prognostic_channel",),
                np.ones(len(PROGNOSTIC), dtype=np.float32),
            ),
            "boundary_mean": (
                ("boundary_channel",),
                np.zeros(len(BOUNDARY), dtype=np.float32),
            ),
            "boundary_std": (
                ("boundary_channel",),
                np.ones(len(BOUNDARY), dtype=np.float32),
            ),
            "prognostic_mask": (("prognostic_channel", "y", "x"), mask),
            "boundary_mask": (
                ("boundary_channel", "y", "x"),
                np.ones((len(BOUNDARY), EXTENT, EXTENT), dtype=bool),
            ),
        },
        coords={
            "time": time,
            "y": np.arange(EXTENT, dtype=np.int32),
            "x": np.arange(EXTENT, dtype=np.int32),
        },
        attrs={
            "cache_format": "llc-train-ready-v1",
            "llc_face": 1,
            "prognostic_channel_names_json": json.dumps(PROGNOSTIC),
            "boundary_channel_names_json": json.dumps(BOUNDARY),
        },
    )
    xarray.to_zarr(
        root / "face.zarr",
        encoding={
            "prognostic": {"chunks": (1, len(PROGNOSTIC), TILE, TILE)},
            "boundary": {"chunks": (1, len(BOUNDARY), TILE, TILE)},
        },
    )
    # `DataConfig.build` checks these exist before the packed branch skips
    # them, so they have to be on disk even though nothing reads them.
    (root / "means.zarr").mkdir()
    (root / "stds.zarr").mkdir()


def _face_config(root, **overrides) -> TrainConfig:
    windows = face_tile_windows(1, extent=EXTENT, tile=TILE, overlap=OVERLAP)
    args = {
        "--backend": "cpu",
        "--epochs": "1",
        "--save_freq": "1",
        "--batch_size": "1",
        "--gradient_accumulation_steps": "1",
        "--experiment.data_root": str(root),
        "--experiment.name": "face_e2e",
        "--experiment.base_output_dir": str(root / "out"),
        "--experiment.prognostic_vars_key": "single_2",
        "--experiment.boundary_vars_key": "single",
        "--data.data_location": "face.zarr",
        "--data.data_means_location": "means.zarr",
        "--data.data_stds_location": "stds.zarr",
        "--data.llc_tiles": json.dumps([list(w) for w in windows]),
        "--data.num_workers": "0",
        "--data.hist": "0",
        "--data.valid_mask": "true",
        "--replay.enabled": "true",
        "--replay.grouped": "true",
        "--replay.buffer_size": "2",
        "--replay.steps_per_epoch": "3",
        "--replay.refresh_every_n_microbatches": "2",
        "--replay.blend_before_backward": "false",
        "--replay.checkpoint_buffer": "false",
        "--face_parallel.enabled": "true",
        "--face_parallel.tiles_per_chunk": "6",
        "--face_parallel.read_threads": "2",
        "--surface_snapshot": "true",
        # The stock test config's window predates this cache; state both.
        "--train_time.start": "1975-08-05",
        "--train_time.end": "1975-09-15",
        "--val_time.start": "1975-09-16",
        "--val_time.end": "1975-09-25",
        "--one_step_val_num": "1",
        "--short_autoregressive_val_num": "0",
        "--long_autoregressive_val_num": "0",
    }
    args.update(overrides)
    flat = ["configs/test/train_default.yaml"]
    for key, value in args.items():
        flat += [key, value]
    config = TrainConfig.from_yaml_and_cli(flat)
    config.inference_epochs = []
    # The real face loss: both gradient terms and a channel weight, so the
    # fixed denominators are exercised on the hard term too. Set here rather
    # than on the command line because the loss field is a union and
    # pydantic-settings hands it the raw string.
    config.loss = GradientLossConfig(
        type=["gradient_h", "gradient_z"],
        metric="mse_mae",
        lambda_h=0.1,
        lambda_z=0.1,
        channel_weights={"Theta": 2.0},
    )
    return config


@pytest.fixture(scope="module")
def face_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("face")
    _write_face_cache(root)
    return root


def test_a_whole_face_trains_end_to_end(face_root, caplog) -> None:
    """The load-bearing test: the real Trainer, all 36 tiles, on one rank."""
    caplog.set_level(logging.INFO)
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()

        assert trainer.fp_ctx is not None
        assert trainer.fp_ctx.local_tiles == tuple(range(36))
        assert trainer.group_frame_reader is not None
        # 36 distinct chunks per array; one read per tile would be far more.
        assert trainer.group_frame_reader.chunks_per_frame == 72
        assert (
            trainer.group_frame_reader.naive_chunks_per_frame
            > trainer.group_frame_reader.chunks_per_frame
        )
        # The face advanced without a runaway tile forcing a reseed.
        assert trainer._diverged_writebacks == 0
        assert trainer._loss_denominator_is_fixed

    text = caplog.text
    assert "Face-parallel rank 0/1" in text
    assert "Face loss normalization installed" in text


def test_the_replay_rows_hold_the_whole_face(face_root) -> None:
    """A row is the face, not a tile: 36 states on one cursor, on the CPU."""
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()
        assert trainer.replay_buffer is not None
        for entry in trainer.replay_buffer.entries:
            assert entry.state.shape[0] == 36
            assert entry.state.device.type == "cpu"
            assert torch.isfinite(entry.state).all()


def test_the_stored_face_agrees_with_itself_in_every_seam(face_root) -> None:
    """What blending is FOR. After a step the tiles are written back
    reconciled, so a cell two tiles share reads the same either way -- which is
    also the precondition the next step's residual blend relies on.
    """
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()

        tiles = face_tile_windows(1, extent=EXTENT, tile=TILE, overlap=OVERLAP)
        state = trainer.replay_buffer.entries[0].state.to(torch.float32)
        worst = 0.0
        compared = 0
        for a, (_, ai0, ai1, aj0, aj1) in enumerate(tiles):
            for b, (_, bi0, bi1, bj0, bj1) in enumerate(tiles[a + 1 :], start=a + 1):
                j0, j1 = max(aj0, bj0), min(aj1, bj1)
                i0, i1 = max(ai0, bi0), min(ai1, bi1)
                if j1 <= j0 or i1 <= i0:
                    continue
                left = state[a, :, j0 - aj0 : j1 - aj0, i0 - ai0 : i1 - ai0]
                right = state[b, :, j0 - bj0 : j1 - bj0, i0 - bi0 : i1 - bi0]
                worst = max(worst, float((left - right).abs().max()))
                compared += 1
        assert compared > 0
        # bfloat16 storage, so exact equality is not on offer; anything above
        # rounding means the blend is not reaching the seams.
        assert worst < 1e-2, f"{compared} seams compared, worst disagreement {worst}"


def test_validation_keeps_a_batch_normalized_loss(face_root) -> None:
    """The training loss divides by the FACE's cell counts, which makes each
    chunk a share rather than a mean -- right for accumulating gradients, and
    wrong for anything reported. Validation and checkpoint selection must keep
    the ordinary normalization or `best_validation_ckpt` ranks on a number
    that is a constant multiple of the metric it claims to be.

    Checked by behaviour rather than by poking at attributes, because
    `channel_weights` wraps the loss and the denominator sits on the inner
    object.
    """
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()
        assert trainer.train_loss_fn is not trainer.loss_fn

        tiles = len(trainer.fp_ctx.local_tiles)
        generator = torch.Generator().manual_seed(3)
        shape = (tiles, len(PROGNOSTIC), SIZE, SIZE)
        pred = torch.randn(shape, generator=generator)
        target = torch.randn(shape, generator=generator)
        mask = trainer.tile_wet_masks.to(dtype=torch.float32)

        def score(loss_fn, spans):
            return sum(
                loss_fn(pred[a:b], target[a:b], sample_weight=mask[a:b])
                for a, b in spans
            )

        whole = [(0, tiles)]
        thirds = [(0, tiles // 3), (tiles // 3, 2 * tiles // 3), (2 * tiles // 3, tiles)]

        # The training loss is a share, so its chunks add up.
        assert torch.allclose(
            score(trainer.train_loss_fn, thirds),
            score(trainer.train_loss_fn, whole),
            rtol=1e-5,
        )
        # The reported loss is a mean, so they do not -- which is exactly why
        # it must not be the one the chunked step accumulates.
        assert not torch.allclose(
            score(trainer.loss_fn, thirds),
            score(trainer.loss_fn, whole),
            rtol=1e-2,
        )


def test_without_face_parallel_both_losses_are_the_same_object(face_root) -> None:
    """Nothing changes for a run that is not a face."""
    with MultitonScope():
        trainer = Trainer(
            _face_config(face_root, **{"--face_parallel.enabled": "false"})
        )
        assert trainer.fp_ctx is None
        assert trainer.train_loss_fn is trainer.loss_fn
        assert not trainer._loss_denominator_is_fixed


def test_grouped_validation_scores_only_the_ranks_own_tiles(face_root) -> None:
    """`ownership_masks` is built from the LAYOUT, which is the whole face
    even when a rank holds a slice of it. Narrowing it is not cosmetic: the
    per-tile wet masks are the rank's, so a face-wide ownership tensor does
    not broadcast against them -- and it only raises once validation runs,
    an epoch after the mistake.
    """
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()  # replay_groups is built by init_data_loaders
        group = trainer.replay_groups[0]
        weight = trainer._grouped_val_weight(group)
        assert weight.shape[0] == group.num_tiles
        assert weight.shape[0] == len(trainer.fp_ctx.local_tiles)
        # Overlaps are halved between neighbours, so the weight is not all-ones
        # anywhere two tiles meet.
        assert float(weight.min()) == 0.0


def test_validation_forwards_are_chunked_like_training(face_root, monkeypatch) -> None:
    """Validation steps a rank's whole tile set, where training steps a chunk.
    On a real face that is nine 752^2 tiles through one forward -- ~17 GB for a
    single GroupNorm output, which OOMed an 80 GB card an epoch after the
    training step had been running fine. Both paths must use the same split.
    """
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()

        model = trainer.model
        model = getattr(model, "module", model)
        seen: list[int] = []
        original = model.predict_step

        def recording(inputs):
            seen.append(inputs.shape[0])
            return original(inputs)

        monkeypatch.setattr(model, "predict_step", recording)
        tiles = len(trainer.fp_ctx.local_tiles)
        inputs = torch.zeros(
            tiles, trainer.num_in, SIZE, SIZE, device=trainer.device
        )
        out = trainer._predict_in_chunks(model, inputs)

        assert out.shape[0] == tiles
        assert seen == [len(chunk) for chunk in trainer.fp_ctx.chunks]
        assert max(seen) <= trainer.face_parallel_cfg.tiles_per_chunk


def test_unchunked_prediction_is_untouched_without_a_face(face_root) -> None:
    with MultitonScope():
        trainer = Trainer(
            _face_config(face_root, **{"--face_parallel.enabled": "false"})
        )
        model = getattr(trainer.model, "module", trainer.model)
        inputs = torch.zeros(2, trainer.num_in, SIZE, SIZE, device=trainer.device)
        assert trainer._predict_in_chunks(model, inputs).shape[0] == 2


def test_face_validation_loss_is_the_mean_over_owned_wet_cells(face_root) -> None:
    """On one rank the face scorer's loss is exactly the ordinary batch mean
    under the ownership weight -- the number grouped validation has always
    reported -- even though it is scored one tile at a time against fixed
    denominators. So no validation curve moves when a face is sharded.
    """
    with MultitonScope():
        trainer = Trainer(_face_config(face_root))
        trainer.run()
        group = trainer.replay_groups[0]
        weight = trainer._grouped_val_weight(group)
        tiles = weight.shape[0]

        generator = torch.Generator().manual_seed(11)
        shape = (tiles, len(PROGNOSTIC), SIZE, SIZE)
        blended = torch.randn(shape, generator=generator, device=trainer.device)
        label = torch.randn(shape, generator=generator, device=trainer.device)

        whole = trainer.loss_fn(blended, label, sample_weight=weight)
        metrics = trainer.face_scorer.score(blended, label)
        assert torch.allclose(metrics.loss_per_channel, whole, rtol=1e-5, atol=1e-7)


def _validation_config(face_root, **overrides):
    return _face_config(
        face_root,
        **{
            "--one_step_val_num": "2",
            "--short_autoregressive_val_num": "1",
            "--short_autoregressive_val_length": "3",
            "--long_autoregressive_val_num": "1",
            "--long_autoregressive_val_length": "6",
            "--long_autoregressive_val_start_epoch": "1",
            **overrides,
        },
    )


def test_face_validation_reports_the_whole_face_for_every_horizon(face_root) -> None:
    """One-step, short and long rollouts all run through the face path and
    report a finite face loss and a pooled RMSE."""
    with MultitonScope():
        trainer = Trainer(_validation_config(face_root))
        trainer.run()
        one_step = trainer.validate_one_epoch(1)
        rollouts = trainer.validate_autoregressive_one_epoch(1)

    for key in ("val/mean/one-step-loss", "val/mean/one-step-rmse"):
        assert np.isfinite(one_step[key]) and one_step[key] > 0, key
    for label in ("short", "long"):
        for metric in ("loss", "rmse"):
            key = f"val/mean/{label}-autoregressive-{metric}"
            assert np.isfinite(rollouts[key]) and rollouts[key] > 0, key


def test_face_rollout_matches_the_established_grouped_rollout(
    face_root, monkeypatch
) -> None:
    """The face rollout reads through the chunk reader and prepares tensors
    with the training dataset; the grouped rollout it replaces for faces
    reads each tile through `InferenceDataset`. On one rank both see the
    whole face, so they must roll out the same states -- and the loss, which
    both score under the ownership weight, must agree at every step.
    """
    from ocean_emulators.train import AutoregressiveValSpec

    with MultitonScope():
        trainer = Trainer(_validation_config(face_root))
        trainer.run()
        group = trainer.replay_groups[0]
        spec = AutoregressiveValSpec(
            label="short", num_steps=4, num_runs=1, seed_offset=0, weight=1.0
        )
        sources = trainer.autoregressive_val_sources()
        model = getattr(trainer.model, "module", trainer.model)
        trainer.model.eval()

        captured = {}
        from ocean_emulators.aggregator.validate import rollout

        original = rollout.RolloutValidationAggregator.get_logs

        def capture(self, label):
            captured.setdefault("curves", []).append(self.loss_by_step().clone())
            return original(self, label)

        monkeypatch.setattr(rollout.RolloutValidationAggregator, "get_logs", capture)
        with torch.no_grad():
            trainer._run_face_autoregressive_validation(model, spec, sources, group)
            trainer._run_grouped_autoregressive_validation(
                model, spec, sources, group, 1
            )

    face, grouped = captured["curves"]
    assert torch.allclose(face, grouped, rtol=1e-4, atol=1e-6), (face, grouped)
