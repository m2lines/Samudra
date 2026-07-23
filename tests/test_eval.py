# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import samudra.eval as eval_module
from samudra.config import EvalConfig, StandaloneEvalConfig
from samudra.eval import Eval
from samudra.utils.location import LocalLocation


def test_eval_runs_with_built_dependencies(monkeypatch, tmp_path):
    model = MagicMock()
    inference_dataset = MagicMock()
    inference_aggregator = MagicMock()
    inference_aggregator.get_summary_logs.return_value = {"loss": 1.5}
    inference_aggregator_factory = MagicMock(return_value=inference_aggregator)
    tensor_map = MagicMock()
    normalize = MagicMock()
    run_rollout = MagicMock()
    monkeypatch.setattr(eval_module, "run_rollout", run_rollout)

    checkpoint_path = Path("checkpoint.pt")
    evaluator = Eval(
        model=model,
        inference_dataset=inference_dataset,
        inference_aggregator_factory=inference_aggregator_factory,
        num_model_steps_forward=25,
        tensor_map=tensor_map,
        normalize=normalize,
        data_container=MagicMock(),
    )

    assert evaluator.standalone_inference(
        output_dir=tmp_path,
        model_path=checkpoint_path,
        save_zarr=True,
    ) == {"inference/loss": 1.5}
    model.eval.assert_called_once_with()
    inference_aggregator_factory.assert_called_once_with()
    run_rollout.assert_called_once_with(
        model=model,
        dataset=inference_dataset,
        inf_aggregator=inference_aggregator,
        epoch=0,
        output_dir=tmp_path,
        model_path=checkpoint_path,
        num_model_steps_forward=25,
        save_zarr=True,
        tensor_map=tensor_map,
        normalize=normalize,
    )


def test_eval_config_build_does_not_accept_overrides():
    assert list(inspect.signature(EvalConfig.build).parameters) == ["self"]


def test_standalone_eval_cli_sets_nested_data_root(tmp_path):
    config = StandaloneEvalConfig.from_yaml_and_cli(
        [
            "configs/samudra_om4/eval.yaml",
            "--eval.data_root",
            str(tmp_path),
        ]
    )

    assert config.eval.resolved_data_root == LocalLocation(path=tmp_path)
