<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Scorecard

The scorecard answers two questions about a finished rollout: is the model better than doing nothing, and is it better
than the best model we had before? Every error is reported as *skill* against a reference forecast computed from the
truth:

- **persistence**, which holds the rollout's initial state fixed and is hard to beat at short leads;
- **climatology**, the pentad-of-year mean over the training years, which is hard to beat at long leads.

Skill is 1 − MSE_model / MSE_reference: 1 is perfect, 0 is no better than the reference, and negative is worse. The
scorecard also checks that the rollout stays physical, and compares every row with the *champion* (the best-known model on
the same track) paired by calendar year. A short list of hard checks forms the *gate*. The design and its decisions are in
issue #986.

## Running it

Build a track's climatology once, from its training split:

```bash
python -m samudra.metrics.scorecard.climatology samudra_om4_v2/eval.yaml \
    --data=@src/samudra/configs/scorecard/data_om4_v2026-09_onedeg.yaml \
    --scorecard.track=om4-v2026-09-onedeg-thermo_dynamic_all \
    --scorecard.climatology_location=/path/to/climatology.zarr
```

Then score any rollout an eval wrote (`<base_output_dir>/<name>/predictions.zarr`):

```bash
python -m samudra.metrics.scorecard samudra_om4_v2/eval.yaml \
    --experiment.name=my_eval --experiment.base_output_dir=/path/to/runs \
    --data=@src/samudra/configs/scorecard/data_om4_v2026-09_onedeg.yaml \
    --scorecard.track=om4-v2026-09-onedeg-thermo_dynamic_all \
    --scorecard.climatology_location=/path/to/climatology.zarr \
    --scorecard.champion=https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/scorecard/<track>/champion
```

Or add the same `scorecard` block to an eval config, and `samudra eval` scores the rollout when it finishes. The command
writes `<output_dir>/scorecard/` and exits with status 1 when the gate fails. A 1° rollout takes about 8 minutes on a CPU
node.

## Reading it

`scorecard.md` is the summary: the gate, physical plausibility, and skill by variable and depth layer. The figures show
skill by depth and lead, drift by year, and variance by depth. `scorecard.parquet` holds every row (see
`samudra.metrics.scorecard.card.COLUMNS`), and the `series_*.parquet` tables it was built from let it be rebuilt without
the fields.

## The champion registry

Each track's champion lives under `s3://m2lines-pubs/Samudra/experiments/scorecard/<track>/champion/`, as the scorecard
directory above. A new champion is promoted by a pull request reviewed by a second maintainer with write access to the
bucket.

## API

::: samudra.metrics.skill

::: samudra.metrics.scorecard
