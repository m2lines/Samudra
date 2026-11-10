# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Final EMA one-step skill versus persistence, including 2–4-cell error power."""

import json

import pandas as pd
import torch

from samudra.config import EvalConfig, ExperimentConfig
from samudra.eval import Eval


def shortwave_power(error: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    """Per-channel 2–4-cell zonal power, using only fully wet periodic rows.

    A channel with no fully wet row has no estimate and returns NaN. Land
    boundaries are never filled and transformed as if they were ocean noise.
    """
    size = error.shape[-1]
    frequencies = torch.fft.rfftfreq(size, device=error.device)
    band = (frequencies >= 0.25) & (frequencies <= 0.5)
    multiplicity = torch.full_like(frequencies, 2)
    multiplicity[0] = 1
    if size % 2 == 0:
        multiplicity[-1] = 1
    power = torch.fft.rfft(error, dim=-1, norm="ortho").abs().square()
    row_power = (power * multiplicity * band).sum(-1) / size
    rows = wet.all(-1)
    return row_power.where(rows, 0).sum(-1) / rows.sum(-1)


@torch.no_grad()
def main():
    cfg = EvalConfig.from_yaml_and_cli()
    if cfg.data.input_steps != 2 or cfg.data.output_steps != 1:
        raise ValueError("This diagnostic requires two inputs and one output")
    cfg.experiment = ExperimentConfig.model_validate(
        {
            **cfg.experiment.model_dump(mode="json"),
            "name": cfg.experiment.name + "-one-step",
        }
    )
    cfg.data.sources[0] = cfg.data.sources[0].model_copy(
        update={"inference_time": cfg.data.sources[0].val_time}
    )
    cfg.observations = None
    evaluation = Eval(cfg)
    cfg.save_yaml(cfg.experiment.output_dir / "config.yaml")
    dataset = evaluation.inference_dataset.to(evaluation.device)
    evaluation.model.eval()
    channels = tuple(evaluation.prognostic_var_names)
    count = len(channels)
    wet = dataset.ctx.label_mask
    error_mse = torch.zeros(count, dtype=torch.float64, device=evaluation.device)
    persistence_mse = torch.zeros_like(error_mse)
    model_power = torch.zeros_like(error_mse)
    persistence_power = torch.zeros_like(error_mse)
    try:
        for index in range(len(dataset)):
            prognostic, boundary, target = dataset[index]
            decoding = evaluation.model.forward_once(prognostic, boundary, dataset.ctx)
            prediction = evaluation.model._assemble_prediction(prognostic, decoding)
            error = (prediction - target) * wet
            persistence_error = (prognostic[:, -count:] - target) * wet
            if not torch.isfinite(error).all():
                raise ValueError(f"Nonfinite prediction at validation window {index}")
            error_mse += error.double().square().mean((0, 2, 3))
            persistence_mse += persistence_error.double().square().mean((0, 2, 3))
            model_power += shortwave_power(error[0], wet)
            persistence_power += shortwave_power(persistence_error[0], wet)
        if len(dataset) == 0 or (persistence_mse <= 0).any():
            raise ValueError("Need validation forecasts and positive persistence MSE")
        frame = pd.DataFrame(
            {
                "channel": channels,
                "model_mse": (error_mse / len(dataset)).cpu().numpy(),
                "persistence_mse": (persistence_mse / len(dataset)).cpu().numpy(),
                "mse_over_persistence": (error_mse / persistence_mse).cpu().numpy(),
                "shortwave_error_power": (model_power / len(dataset)).cpu().numpy(),
                "shortwave_persistence_power": (persistence_power / len(dataset))
                .cpu()
                .numpy(),
                "shortwave_power_over_persistence": (model_power / persistence_power)
                .cpu()
                .numpy(),
                "fully_wet_rows": wet.all(-1).sum(-1).cpu().numpy(),
            }
        )
        frame["field"] = frame.channel.str.replace(r"_\d+$", "", regex=True)
        frame.to_csv(cfg.experiment.output_dir / "one_step_skill.csv", index=False)
        scalars = {}
        for field, group in frame.groupby("field"):
            scalars[f"one_step/{field}/mse_over_persistence"] = float(
                group.model_mse.sum() / group.persistence_mse.sum()
            )
        evaluation.wandb_logger.log(scalars, step=None)
        metadata = {
            "checkpoint": cfg.ckpt_path,
            "validation_forecasts": len(dataset),
            "validation_start": str(cfg.data.sources[0].val_time.start),
            "validation_end": str(cfg.data.sources[0].val_time.end),
            "units": "state-normalized, full-grid MSE with land zeroed",
            "spectra": "zonal FFT; wavelengths 2–4 grid cells; fully wet rows only; NaN with no supported rows",
            "field_skill": scalars,
        }
        (cfg.experiment.output_dir / "provenance.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(metadata, indent=2), flush=True)
    finally:
        evaluation.wandb_logger.finish()


if __name__ == "__main__":
    main()
