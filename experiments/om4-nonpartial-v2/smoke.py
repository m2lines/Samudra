# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual Rust loader, optimizer, validation and checkpoint writer."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("arm")
a = p.parse_args()
cfg = yaml.safe_load((a.root / a.arm / "train.yaml").read_text())
cfg["epochs"] = 1
cfg["data"]["sources"][0]["train_time"]["end"] = "1975-04-30"
cfg["experiment"]["base_output_dir"] = str(a.root / "qualification-output")
cfg["experiment"]["wandb"]["mode"] = "disabled"
cfg["validation_image_log_freq"] = 1
cfg["backend"] = "cuda"
path = a.root / a.arm / "smoke.yaml"
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
subprocess.run([sys.executable, "-m", "samudra.train", str(path)], check=True)
