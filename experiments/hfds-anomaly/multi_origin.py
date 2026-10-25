"""Matched quarterly-origin native rollouts, retaining actual timestamps."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from runtime import install

from samudra.config import EvalConfig, Om4TimeConfig
from samudra.eval import Eval
from samudra.utils.logging import handle_logging, handle_warnings
from samudra.utils.multiton import MultitonScope

ROOT = Path("/scratch/jr7309/om4-hfds-ablation-20261021")


def scalar_logs(logs):
    out = {}
    for k, v in logs.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            v = v.item()
        if isinstance(v, (int, float, str, bool)):
            if isinstance(v, float) and not np.isfinite(v):
                raise ValueError(f"Nonfinite metric {k}: {v}")
            out[k] = v
    return out


def main():
    install()
    arm, label = [
        ("with", "selected"),
        ("without", "selected"),
        ("with", "final-ema"),
        ("without", "final-ema"),
    ][int(sys.argv[1])]
    raw = yaml.safe_load((ROOT / "configs" / f"eval-{arm}-{label}.yaml").read_text())
    raw["observations"] = None
    raw["experiment"]["base_output_dir"] = str(ROOT / "multi-origin")
    raw["experiment"]["name"] = f"{arm}-{label}"
    raw["experiment"]["wandb"]["group"] = "hfds-multi-origin-20261025"
    cfg = EvalConfig.model_validate(raw)
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    checkpoint = Path(cfg.ckpt_path)
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with MultitonScope():
        ev = Eval(cfg)
        full_source = ev.source
        times = list(full_source.time.values)
        manifest = []
        try:
            for year in range(2015, 2022):
                for month in (1, 4, 7, 10):
                    requested = f"{year:04d}-{month:02d}-01"
                    origin = next(
                        i
                        for i, t in enumerate(times)
                        if (t.year, t.month, t.day) >= (year, month, 1)
                    )
                    start, end = origin - 1, origin + 72
                    assert start >= 0 and end < len(times)
                    out = cfg.experiment.output_dir / requested
                    record = dict(
                        requested_origin=requested,
                        initialization=str(times[origin]),
                        input_times=[str(t) for t in times[start : origin + 1]],
                        target_times=[str(t) for t in times[origin + 1 : end + 1]],
                        lead_days=[
                            (t - times[origin]).total_seconds() / 86400
                            for t in times[origin + 1 : end + 1]
                        ],
                        checkpoint_sha256=digest,
                    )
                    receipt = out / "COMPLETE.json"
                    if receipt.exists():
                        prior = json.loads(receipt.read_text())
                        assert (
                            prior["checkpoint_sha256"] == digest
                            and prior["target_times"] == record["target_times"]
                        )
                        manifest.append(record)
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    ev.source = full_source.slice_time(
                        Om4TimeConfig.model_validate(
                            dict(
                                start=str(times[start]).split()[0],
                                end=str(times[end]).split()[0],
                            )
                        )
                    )
                    assert len(ev.source.time) == 74
                    ev.metadata = ev.source.metadata
                    ev.output_dir = out
                    ev.init_inference_store()
                    print(
                        "ORIGIN_BEGIN",
                        arm,
                        label,
                        requested,
                        record["initialization"],
                        flush=True,
                    )
                    logs = ev.standalone_inference()
                    record["metrics"] = scalar_logs(logs)
                    receipt.write_text(json.dumps(record, indent=2))
                    manifest.append(record)
                    print("ORIGIN_COMPLETE", arm, label, requested, flush=True)
            (cfg.experiment.output_dir / "COMPLETE.json").write_text(
                json.dumps(
                    dict(
                        arm=arm,
                        selection=label,
                        checkpoint_sha256=digest,
                        origins=manifest,
                    ),
                    indent=2,
                )
            )
        finally:
            ev.finish()


if __name__ == "__main__":
    main()
