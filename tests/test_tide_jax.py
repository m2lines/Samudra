import numpy as np
import pytest
import torch

from ocean_emulators.constants import LoaderVersion
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ocean_emulators import tide_jax  # noqa: E402


class DictMaterializer:
    def __init__(self, values, *, blob_placement="cpu", output_placement=None):
        self.values = values
        self.requests = []
        self.blobs = []
        self.blob_placement = blob_placement
        self.output_placement = output_placement or blob_placement

    def placement_for_blob(self, blob: tide_jax.TideJaxBlob):
        self.blobs.append(blob)
        return self.blob_placement

    def jax_device_for_blob(self, _blob: tide_jax.TideJaxBlob, placement):
        if placement == "cpu":
            return jax.devices("cpu")[0]
        return jax.devices()[0]

    def output_device(self):
        if self.output_placement == "cpu":
            return jax.devices("cpu")[0]
        return jax.devices()[0]

    def materialize(
        self,
        leaf: tide_jax.TideLeaf,
        *,
        tensor_placement: tide_jax.TensorPlacement = "cpu",
        jax_device=None,
    ):
        self.requests.append((leaf, tensor_placement, jax_device))
        return jax.device_put(
            jnp.asarray(self.values[(leaf.kind, leaf.step)]), jax_device
        )


class ShapeOnlyBatch:
    def __len__(self):
        return 4

    def get_raw_step0_parts(self):
        return (
            torch.zeros((2, 2, 4, 5)),
            torch.zeros((2, 1, 4, 5)),
            torch.zeros((2, 2, 4, 5)),
        )

    def get_raw_boundary(self, step):
        raise AssertionError(f"unexpected later-step materialization for step {step}")

    def get_raw_label(self, step):
        raise AssertionError(f"unexpected later-step materialization for step {step}")


def _spec() -> tide_jax.TideJaxShapeSpec:
    return tide_jax.TideJaxShapeSpec(
        step0_prognostic_shape=(2, 2, 4, 5),
        step0_boundary_shape=(2, 1, 4, 5),
        step0_label_shape=(2, 2, 4, 5),
        boundary_shape=(2, 1, 4, 5),
        label_shape=(2, 2, 4, 5),
    )


def test_shape_spec_from_batch_does_not_materialize_later_steps():
    spec = tide_jax.shape_spec_from_batch(ShapeOnlyBatch())

    assert spec.boundary_shape == (2, 1, 4, 5)
    assert spec.label_shape == (2, 2, 4, 5)


def test_plain_jax_slice_runs_inside_an_opaque_blob():
    spec = _spec()

    def step0():
        return tide_jax.raw_step0_prognostic(spec)[..., 1:3, 2:5] + 1.0

    program = tide_jax.trace_tide_jax(step0)
    values = {
        ("raw_step0_prognostic", 0): np.arange(
            np.prod(spec.step0_prognostic_shape), dtype=np.float32
        ).reshape(spec.step0_prognostic_shape)
    }
    materializer = DictMaterializer(values)

    out = program.eval(materializer)

    np.testing.assert_allclose(
        out, values[("raw_step0_prognostic", 0)][..., 1:3, 2:5] + 1.0
    )
    assert materializer.blobs
    assert "slice" in materializer.blobs[0].primitive_names
    assert materializer.requests[0][1] == "cpu"


def _stats(*, normalize_before_mask: bool = True) -> tide_jax.TideJaxStats:
    prognostic_mask = np.ones((2, 4, 5), dtype=np.bool_)
    prognostic_mask[0, 0, 0] = False
    prognostic_mask[1, 1, 1] = False
    boundary_mask = np.ones((4, 5), dtype=np.bool_)
    boundary_mask[0, 1] = False
    return tide_jax.TideJaxStats(
        prognostic_mean=np.asarray([10.0, 20.0], dtype=np.float32),
        prognostic_std=np.asarray([2.0, 4.0], dtype=np.float32),
        boundary_mean=np.asarray([100.0], dtype=np.float32),
        boundary_std=np.asarray([5.0], dtype=np.float32),
        prognostic_mask=prognostic_mask,
        boundary_mask=boundary_mask,
        normalize_before_mask=normalize_before_mask,
        masked_fill_value=np.float32(-9.0),
    )


def _normalize_and_mask_np(
    tensor: np.ndarray, stats: tide_jax.TideJaxStats, kind: tide_jax.TideTensorKind
) -> np.ndarray:
    def normalize(data):
        if kind == "prognostic":
            mean = stats.prognostic_mean
            std = stats.prognostic_std
        else:
            mean = stats.boundary_mean
            std = stats.boundary_std
        stat_index = np.arange(data.shape[1]) % mean.shape[0]
        out = (data - mean[stat_index][None, :, None, None]) / std[stat_index][
            None, :, None, None
        ]
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def apply_mask(data):
        if kind == "prognostic":
            mask_index = np.arange(data.shape[1]) % stats.prognostic_mask.shape[0]
            mask = stats.prognostic_mask[mask_index][None, :, :, :]
        else:
            mask = stats.boundary_mask[None, None, :, :]
        return np.where(mask, data, stats.masked_fill_value).astype(np.float32)

    if stats.normalize_before_mask:
        return apply_mask(normalize(tensor))
    return normalize(apply_mask(tensor))


def _with_rust_loader(train_config):
    return train_config.model_copy(
        update={
            "data": train_config.data.model_copy(
                update={
                    "loader_version": str(LoaderVersion.OM4_RUST_V0.value),
                    "num_workers": 0,
                }
            )
        }
    )


def test_tide_jax_traces_step0_leaves_and_evaluates_jax_ops():
    spec = _spec()
    stats = _stats()

    def step0():
        prognostic = tide_jax.normalize_and_mask(
            tide_jax.raw_step0_prognostic(spec),
            stats,
            "prognostic",
        )
        boundary = tide_jax.normalize_and_mask(
            tide_jax.raw_step0_boundary(spec),
            stats,
            "boundary",
        )
        input_ = jnp.concatenate((prognostic, boundary), axis=1)
        label = tide_jax.normalize_and_mask(
            tide_jax.raw_step0_label(spec),
            stats,
            "prognostic",
        )
        return jnp.where(input_ > 10.0, input_, -input_), label + 1.0

    program = tide_jax.trace_tide_jax(step0)

    assert [(leaf.kind, leaf.step) for leaf in program.leaves] == [
        ("raw_step0_prognostic", 0),
        ("raw_step0_boundary", 0),
        ("raw_step0_label", 0),
    ]

    prognostic = np.arange(
        np.prod(spec.step0_prognostic_shape), dtype=np.float32
    ).reshape(spec.step0_prognostic_shape)
    boundary = np.arange(np.prod(spec.step0_boundary_shape), dtype=np.float32).reshape(
        spec.step0_boundary_shape
    )
    label = np.arange(np.prod(spec.step0_label_shape), dtype=np.float32).reshape(
        spec.step0_label_shape
    )
    out_input, out_label = program.eval(
        DictMaterializer(
            {
                ("raw_step0_prognostic", 0): prognostic,
                ("raw_step0_boundary", 0): boundary,
                ("raw_step0_label", 0): label,
            }
        )
    )
    input_ = np.concatenate(
        (
            _normalize_and_mask_np(prognostic, stats, "prognostic"),
            _normalize_and_mask_np(boundary, stats, "boundary"),
        ),
        axis=1,
    )

    np.testing.assert_allclose(
        out_input, np.where(input_ > 10.0, input_, -input_), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        out_label, _normalize_and_mask_np(label, stats, "prognostic") + 1.0
    )


def test_tide_jax_traces_later_step_boundary_and_label():
    spec = _spec()
    stats = _stats()
    prev = jax.ShapeDtypeStruct((2, 2, 4, 5), jnp.float32)

    def later_step(prev_prediction):
        boundary = tide_jax.normalize_and_mask(
            tide_jax.raw_boundary(spec, 1),
            stats,
            "boundary",
        )
        label = tide_jax.normalize_and_mask(
            tide_jax.raw_label(spec, 1),
            stats,
            "prognostic",
        )
        input_ = jnp.concatenate((prev_prediction, boundary), axis=1)
        return input_, label

    program = tide_jax.trace_tide_jax(later_step, prev)

    assert [(leaf.kind, leaf.step) for leaf in program.leaves] == [
        ("raw_boundary", 1),
        ("raw_label", 1),
    ]

    prev_value = np.ones(prev.shape, dtype=np.float32)
    boundary = np.full(spec.boundary_shape, 2.0, dtype=np.float32)
    label = np.full(spec.label_shape, 3.0, dtype=np.float32)
    out_input, out_label = program.eval(
        DictMaterializer({("raw_boundary", 1): boundary, ("raw_label", 1): label}),
        jnp.asarray(prev_value),
    )

    np.testing.assert_allclose(
        out_input,
        np.concatenate(
            (prev_value, _normalize_and_mask_np(boundary, stats, "boundary")), axis=1
        ),
    )
    np.testing.assert_allclose(
        out_label, _normalize_and_mask_np(label, stats, "prognostic")
    )


def test_tide_jax_rejects_dynamic_control_flow():
    spec = _spec()

    def unsupported(flag):
        return jax.lax.cond(
            flag,
            lambda _: tide_jax.raw_step0_prognostic(spec),
            lambda _: tide_jax.raw_step0_prognostic(spec),
            operand=None,
        )

    with pytest.raises(NotImplementedError, match="cond"):
        tide_jax.trace_tide_jax(
            unsupported,
            jax.ShapeDtypeStruct((), jnp.bool_),
        )


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
def test_tide_jax_matches_rust_batch(train_config):
    pytest.importorskip("tide")
    rust_config = _with_rust_loader(train_config)

    with MultitonScope():
        torch_trainer = Trainer(train_config)
        torch_trainer.init_data_loaders(cur_step=train_config.steps[0])
        torch_batch = torch_trainer.train_loader[0]

    with MultitonScope():
        trainer = Trainer(rust_config)
        trainer.init_data_loaders(cur_step=rust_config.steps[0])
        batch = trainer.train_loader[0]

        spec = tide_jax.shape_spec_from_batch(batch)
        stats = tide_jax.stats_from_batch(batch)

        def step0():
            prognostic = tide_jax.normalize_and_mask(
                tide_jax.raw_step0_prognostic(spec),
                stats,
                "prognostic",
            )
            boundary = tide_jax.normalize_and_mask(
                tide_jax.raw_step0_boundary(spec),
                stats,
                "boundary",
            )
            label = tide_jax.normalize_and_mask(
                tide_jax.raw_step0_label(spec),
                stats,
                "prognostic",
            )
            return jnp.concatenate((prognostic, boundary), axis=1), label

        step0_program = tide_jax.trace_tide_jax(step0)
        out_input, out_label = step0_program.eval(
            tide_jax.RustTrainBatchMaterializer(batch, tensor_placement="cpu")
        )

        np.testing.assert_allclose(
            out_input, torch_batch.get_input(0).numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            out_label, torch_batch.get_label(0).numpy(), rtol=1e-6, atol=1e-6
        )

        prev_prediction = torch.randn_like(torch_batch.get_label(0))
        prev = jax.ShapeDtypeStruct(tuple(prev_prediction.shape), jnp.float32)

        def later_step(prev_prediction):
            boundary = tide_jax.normalize_and_mask(
                tide_jax.raw_boundary(spec, 1),
                stats,
                "boundary",
            )
            label = tide_jax.normalize_and_mask(
                tide_jax.raw_label(spec, 1),
                stats,
                "prognostic",
            )
            return (
                jnp.concatenate((prev_prediction, boundary), axis=1),
                label,
            )

        later_program = tide_jax.trace_tide_jax(later_step, prev)
        out_input, out_label = later_program.eval(
            tide_jax.RustTrainBatchMaterializer(batch, tensor_placement="cpu"),
            jnp.asarray(prev_prediction.numpy()),
        )

        np.testing.assert_allclose(
            out_input,
            torch_batch.merge_prognostic_and_boundary(prev_prediction, 1).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out_label, torch_batch.get_label(1).numpy(), rtol=1e-6, atol=1e-6
        )
