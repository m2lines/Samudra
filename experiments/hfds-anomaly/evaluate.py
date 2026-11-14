"""Evaluate both predeclared checkpoint rules, with durable per-stage receipts."""

import copy
import json
import sys
from pathlib import Path

import torch
import yaml

from samudra.config import EvalConfig
from samudra.eval import Eval
from samudra.utils.multiton import MultitonScope


def main():
    from runtime import install

    install()
    from local_cache import stage

    from samudra.utils.location import LocalLocation
    from samudra.utils.logging import handle_logging, handle_warnings

    cached_source = stage()
    root = Path("/orcd/pool/008/jrusak/om4-hfds-ablation-20261021")
    arm = sys.argv[1]
    train = yaml.safe_load((root / "configs" / f"{arm}.yaml").read_text())
    obs = yaml.safe_load((root / "code/src/samudra/configs/data/obs.yaml").read_text())
    for label, filename in [
        ("selected", "best_validation_ckpt.pt"),
        ("final-ema", "ema_ckpt.pt"),
    ]:
        out = root / "evaluations" / f"{arm}-{label}"
        if (out / "COMPLETE.json").exists():
            continue
        ckpt = root / "outputs" / f"hfds-{arm}" / "saved_networks" / filename
        # Resolve the native checkpoint directory rather than assume its name.
        matches = list((root / "outputs" / f"hfds-{arm}").rglob(filename))
        assert len(matches) == 1, matches
        ckpt = matches[0]
        cfg = dict(
            debug=False,
            save_zarr=True,
            disk_mode=True,
            num_model_steps_forward=25,
            backend="cuda",
            ckpt_path=str(ckpt),
            data=train["data"],
            model=train["model"],
            experiment=copy.deepcopy(train["experiment"]),
            observations=obs,
        )
        cfg["experiment"].update(
            base_output_dir=str(root / "evaluations"), name=f"{arm}-{label}"
        )
        cfg["experiment"]["wandb"]["group"] = "hfds-anomaly-eval-20261021"
        (root / "configs" / f"eval-{arm}-{label}.yaml").write_text(yaml.safe_dump(cfg))
        with MultitonScope():
            resolved = EvalConfig.model_validate(cfg)
            resolved.data.sources[0].data_location = LocalLocation(path=cached_source)
            resolved.prepare_output_dirs()
            handle_logging(resolved.debug, resolved.experiment.output_dir)
            handle_warnings()
            state = torch.load(ckpt, map_location="cpu")
            print("CHECKPOINT_VERIFIED", str(ckpt), list(state), flush=True)
            del state
            ev = Eval(resolved)
            result = ev.run()
            scalars = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in result.items()
                if isinstance(v, (float, int, str, bool))
                or isinstance(v, torch.Tensor)
                and v.numel() == 1
            }
            out.mkdir(parents=True, exist_ok=True)
            (out / "COMPLETE.json").write_text(
                json.dumps(dict(checkpoint=str(ckpt), metrics=scalars), indent=2)
            )
            ev.wandb_logger.finish()
            del ev
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
