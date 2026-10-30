<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Frozen experiment source

These archives contain the exact source bundles transferred to Torch for the October 30, 2026 noise and information-probe experiments. Each archive expands into `code/`; its sibling `SOURCE.json` records a SHA-256 checksum for each file and the experiment source identifier. Training data, containers, credentials, and checkpoints are not included.

- [Noise experiment source](noise/code.tar.gz) and [manifest](noise/SOURCE.json).
- [Information-probe source](information-probe/code.tar.gz) and [manifest](information-probe/SOURCE.json).

The main entry points inside the archives are `scripts/run_stochastic_noise_ablation.py`, `scripts/run_stochastic_information_campaign.py`, and `scripts/probe_stochastic_velocity_information.py`. Their Slurm scripts record the container invocation. Paths to external datasets and completed checkpoints are in the experiment reports and run receipts.

The report branch contains standalone analysis scripts. Its installed Samudra package predates some dependencies of these experiments; use the complete frozen source bundle when reproducing training or model evaluation. Extracting only the updated model and trainer into this branch would omit required data and physical-state modules.

Individual files retain their original copyright and license notices. The archives contain files under Apache-2.0, MIT, CC-BY-4.0, and CC0-1.0.

## Viewer export

The newer [viewer exporter](../../../../scripts/export_stochastic_viewer_rollouts.py)
runs with this frozen training source on `PYTHONPATH`. Explicit checkpoints require
their expected arm, update count, saved protocol and existing evaluation directory.
It verifies saved member outputs before exporting additional fields and dates.
`--export-om4-initialization` also exports matching OM4 initialization gold.
The [Slurm wrapper](../../../../scripts/slurm_stochastic_viewer_rollouts.sbatch)
retains the THP and cuDNN compatibility settings.

Convert the completed export into an immutable viewer catalog with:

```bash
python scripts/prepare_stochastic_viewer.py \
  --source /path/to/verified-export \
  --output /path/to/new-viewer-catalog \
  --profile learned-noise-2k
```

The catalog contains the mean of eight trajectories and members 1 and 2,
three observation-initialized 30-day trajectories, and separate OM4 initialization
examples when exported. The converter verifies checksums, checkpoint identity,
coordinates, masks and five-day time intervals. Deployment registers this child
catalog alongside existing models; it does not replace their arrays.
