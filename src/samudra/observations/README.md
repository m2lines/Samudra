<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation/model-data baseline

Read [`baseline.py`](baseline.py): the model construction, losses, source cycle,
replay updates, validation and cooldown are ordinary commented Python. Edit that
code to change the experiment. The CLI supplies paths and operational controls;
there is no YAML, config builder or generic task framework on this path.

## Model and targets

The deterministic U-global model initializes two states from 19 five-day SST/SSH
and forcing histories, with explicit validity and geographic/seasonal context.
Each state contains 77 channels: U, V, temperature and salinity at 19 depths,
plus SSH. The ConvNeXt U-Net uses widths 128/192/256/384, affine InstanceNorm,
task-specific input adapters and zero-initialized geometry conditioning. It has
62,967,680 parameters and no extra recurrent memory channels. OM4 and CM4 also
initialize from surface histories. Their dense interior targets never become
initializer inputs.

All sources use observation-fitted **state** normalization. Unobserved velocity
slots use mean zero/std one; the five deepest T/S slots repeat the deepest
observed scales. OISST sets the surface-temperature scale. These conventions
do not create labels in unobserved channels. The fitting domain remains the
historical ±60° domain and repeated-history sample weighting; inputs, training
losses and evaluation have **global** finite wet support.

Observations use the learned eight-channel ERA5-to-three-channel forcing adapter.
Model sources retain their own tauuo/tauvo/hfds, scaled by the existing OM4
forcing statistics. CM4 keeps its no-leap calendar, interval-end labels and
native wet mask. Source calendars are never joined into a single timeline.

The observation forecast loss is 0.8 monthly T/S + 0.1 five-day SST + 0.1 SSH,
with 0.1 interior reconstruction and 0.1 missing-surface completion auxiliaries.
Model data supervise all physical fields with variable-balanced MSE and the
original reconstruction/completion auxiliaries. Artificial hiding modifies
inputs, not labels. Later ocean observations enter the separate reconstruction
branch only; they never correct or reset a forecast. No extra multitask tasks,
diffusion objectives or latent penalties are included.

## Data preparation

Start from the identified stores `oisst.zarr`, `duacs.zarr`, `argo-iap.zarr`,
and `era5-surface.zarr`. These contain daily SST/ADT, monthly IAP T/S and hourly
ERA5. The scripts preserve conservative remapping, finite-coverage thresholds,
strict five-day means, ERA5 accumulation timing, depth interpolation and salinity
conversion. Provider downloads and transfers are separate operations.

Run from this repository with Samudra installed, or `PYTHONPATH=src`. IAP
preparation additionally needs the pinned `gsw==3.6.23` preprocessing dependency.

```bash
python data/prepare_observation_baseline.py grid \
  --om4 /data/om4_onedeg_v3/OM4.zarr --output /data/obs/grid.npz
python data/prepare_observation_baseline.py daily \
  --source /data/products --grid /data/obs/grid.npz --output /data/obs/daily \
  --product sst --year 1993
# Repeat daily preparation for sst, adt, era5 and iap for 1993–2014.
python data/prepare_observation_baseline.py months --root /data/obs
python data/prepare_observation_normalization.py /data/obs
```

Normalization is a separate artifact (`normalization.npz`, `statistics.npz`,
`normalization.json`), fitted only to the 243 May 1993–July 2013 training months.
An existing prepared archive can write these artifacts elsewhere with `--output`.
The trainer accepts that directory through `--normalization`, and an existing
daily archive through `--daily`. It never refits normalization while training.
Retain daily shards: replay must reconstruct intervals anchored at the original
forecast origin, rather than concatenate differently anchored monthly samples.

OM4 uses the unblurred 1° `om4_onedeg_v3` with its original means/stds stores;
eligible histories and targets lie in the late-OM4 1975–July 2013 period.
CM4 uses `s3://m2lines-pubs/Samudra/v2026-09/cm4_onedeg/CM4.zarr`, with its first
180 model years eligible and the final twenty held out. The readers check the
common grid, masks and interval bounds. Windows crossing missing intervals are
excluded, including OM4 leap-boundary gaps. A replay chain cannot cross a gap.

## Train, resume, evaluate

```bash
python -m samudra.observations.baseline train \
  --observations /data/obs --output /runs/obs --mix obs
python -m samudra.observations.baseline train \
  --observations /data/obs --om4 /data/om4_onedeg_v3 \
  --output /runs/om4-obs --mix om4-obs
python -m samudra.observations.baseline train \
  --observations /data/obs --om4 /data/om4_onedeg_v3 --cm4 /data/CM4.zarr \
  --output /runs/om4-cm4-obs --mix om4-cm4-obs
```

Each recipe runs 2,000 main updates and 1,000 observation-only cooldown updates,
with eight accumulated examples per update. Main source cycles are `obs`,
`obs→OM4`, and `obs→OM4→obs→CM4`; these mean 100/0/0, 50/50/0, and 50/25/25
percent obs/OM4/CM4. Total updates match, not observation exposure. Seed is 1729,
AdamW uses LR 1e-4 and weight decay 0.01, and gradients clip at one. Cooldown
retains Adam moments, resets EMA to raw weights, and linearly reduces LR to 1e-6.
EMA has a 100-update half-life.

Each source has its own replay pool. Four examples are fresh and four continue
cached states (falling back to fresh only when no eligible entry exists). Pools
sample uniformly across occupied lead-month buckets, then entries. Two entries
per bucket bound full-resolution physical-state RAM; a full float32 pool is
about 0.9 GiB per source. States expire 250 global updates after their original
initialization; replaying does not refresh their age. They remain detached CPU
tensors in their original precision. Replay backpropagates through one target
month, and never beyond the saved state. Source/calendars stay fixed throughout
each chain. Target bins stop at 73 five-day steps; a terminal calendar month that
cannot be fully supervised within that horizon is ineligible.

The next month's state precedes its first overlapping five-day bin. For January
31, seven bins predict January, but the next snapshot is taken after six calls
at day 30. February recomputes the overlapping bin. This preserves complete
calendar-month T/S supervision without retaining a gradient graph across months.
Cooldown uses only observation updates and observation replay.

Restart an identical command to resume `latest.pt`; `--checkpoint` selects an
explicit trusted checkpoint. `--stop-after` stops after an absolute update count
without changing the 3k scientific plan. SIGTERM/USR1 finish the current update
and checkpoint. Model/Adam/EMA, source exposure, random streams, replay entries,
ages and training phase are restored. There is no mid-update resume. Checkpoints
use Python serialization and must come from a trusted run.

W&B defaults to online (`--entity`, `--project`); use `--wandb-mode disabled` for
a qualification without network logging. The same scalar records go to local
`metrics.jsonl`. Source-specific losses, LR, clipping norm, fresh/replay exposure,
maximum sampled lead and annual channel metrics are recorded.

## Annual checkpoint selection

Validation is one continuous 365-day rollout from **November 1, 2013**, with one
initialization, prescribed future ERA5 and no observed-ocean resets. It ends
November 1, 2014, before the 2015–2022 test cohort. Validation every 250 updates
uses EMA weights and Samudra's spatial area-weighted RMSE kernel.

The selection key is `rollout_val/365d/normalized_rmse/channel_mean`. RMSE is
divided by observation-training standard deviation, **not climatology error**.
We average times within each observed channel, then average 29 channels equally:
five-day SST/SSH, monthly T at depths 1–13 and monthly S at depths 0–13. This
matches Samudra's equal channel/time reduction when cadences agree, while avoiding
giving surfaces six times the weight solely because they are sampled more often.
Monthly T at depth zero is an optional diagnostic, not a second SST vote. The
monthly predictions are exact calendar-weighted means of the continuous forecast.
Raw-unit RMSE and support counts are logged separately. No U/V or unobserved deep
target is invented. Nonfinite forecasts at observed cells fail loudly.

`best.pt` is the best annual validation checkpoint across both phases;
`main.pt` is the uncooled endpoint and `endpoint.pt` is the final cooled model.
A run stopped before its first validation has `latest.pt` but no selected model.
Only a fully completed 3k run writes `COMPLETE.json`.

```bash
python -m samudra.observations.baseline evaluate \
  --observations /data/obs --output /runs/obs --mix obs \
  --checkpoint /runs/obs/best.pt
```

Evaluation reconstructs the same source/model definitions and scores EMA weights.
It does not select on or automatically evaluate the held-out test cohort.

## Qualification and provenance

The focused tests cover model continuation and future-input isolation, calendar
overlap, missingness, native calendars, all source cycles, replay aging, annual
RMSE parity, cooldown and exact next-update resume. The real-data qualification
script uses small explicitly labeled update counts and never produces baseline
training results. Full training and comparative reports are follow-up work.

Run manifests pin code hashes, normalization and model Zarr metadata. Normalization
inputs are content-hash verified on startup. Daily-archive file names/sizes/mtimes
and preparation manifests are fingerprinted for change detection. Model field
chunks are checked for support/finiteness as read; this is not a full content
read-back of every model/daily archive chunk. Retain versioned, immutable input
stores and the preprocessing manifests. Checkpoints reject changed manifests.
