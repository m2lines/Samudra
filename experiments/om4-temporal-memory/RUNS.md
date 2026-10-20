# Execution record — October 16, 2026

Host chosen: **Engaging**, using individual L40S GPUs. Beta production was fully
allocated at initial inspection and requires full four-GPU nodes. Engaging queue
estimates are advisory; idle counts included nodes unavailable to general jobs.

Remote root: `/orcd/data/abodner/002/jrusak/om4-temporal-memory-20261016`.
Current immutable code snapshot: `code-v3`; original snapshots are retained.

| Job | Purpose | Submission state |
|---|---|---|
| 26066769 | Rescale isolated diagnostic inputs to frozen archive statistics; CPU replay gate | Completed, exit 0; normalized RMSE 0.000371621 |
| 26066770 | Full normalized training-input preparation | Running after successful replay |
| 26066771 | Six-origin diagnostics, report, four optimizer smoke tests, online W&B qualification | Pending GPU allocation; CPU replay passed |
| 26066772_0 | Full temporal gradients | Depends on preparation and qualification |
| 26066772_1 | Detached physical feedback | Same dependencies |
| 26066772_2 | Persistent latents, full temporal gradients | Same dependencies |
| 26066772_3 | Persistent latents, temporal gradients only through latents | Same dependencies |
| 26066808 | CPU completion callback for diagnostics | afterany:26066771 |
| 26066810 | CPU completion callback for training array | afterany:26066772 |

Durable monitoring registrations (parent thread `01a144d3-d415-7072-8cf0-ff06eb2db174`):

- Diagnostics: `5a2c6689-6e4a-4dd5-95fd-942e11b8c667`, monitor `/root/diagnostics_monitor`.
- Training: `0fd7466c-68df-4313-a8c8-9969a852dd1f`, monitor `/root/training_monitor`.

Callback authentication checked from login host; compute-node callback delivery
must be verified when the dependent CPU notification job runs. Timed reminders
provide fallback. The scripts and reports contain no callback tokens.

## Qualification correction

- CPU preparation 26065159 completed.
- CPU replay 26065822 failed: local Engaging normalization did not reproduce the
  archived official forecast (normalized RMS difference 0.3602 in local-stat units).
- Dependent diagnostics 26065824 and training 26065825 were cancelled without execution.
  Their callbacks ran and were retired. Full CPU preparation 26065823 was stopped.
- Engaging and archive raw U/T/S/SSH/hfds first-input planes had matching SHA-256 hashes.
  Normalization scalars differed. Surface U std: local 0.103232786 versus archive 0.126944527.
- A local CPU controlled probe with the same weights/inputs gave replay RMS 0.197514
  using local statistics versus 0.000371618 using archive statistics, both measured in
  archive-normalized units. The preset acceptance threshold is 0.001; this is close
  replay, not bit equality. Cluster CPU replay subsequently passed with normalized
  RMS 0.000371621 (maximum absolute normalized difference 0.0185997).
  The GPU replay check remains part of the qualification chain.
- Shared source and normalization stores were not modified. The corrected run uses
  frozen `archive-normalization.json` in isolated `*-data-archive` stores.
- CPU probe 26066387 was cancelled after the local comparison resolved its question.

## Completion criteria

Diagnostics require successful replay, six origin JSON files, provenance, report,
summary CSV, plot, and `QUALIFIED`. Training requires all four arms to finish the
same 70 epochs with online W&B URLs, then selected-EMA held-out evaluations with
`COMPLETE`, `EVAL_COMPLETE`, and `heldout.json` for each arm. A scheduler exit or
`NEEDS_RESUME` is not scientific completion. Resume within the authorized 70-epoch
scope when necessary. The final comparison must wait for all four arms.

## Checkpoint replay repair (v4)

Diagnostics 26066771 failed after 37 seconds during the second gradient computation:
a generator context manager used by checkpoint recomputation could only be entered
once. No diagnostic report was produced; training array 26066772 was cancelled
without execution. Replaced that helper with a reusable context that saves and
restores BatchNorm buffers on every entry. Six local tests pass, including a new
repeated-gradient comparison against uncheckpointed execution in train/eval modes;
Ruff checks pass. Scientific settings are unchanged.

Current immutable remote snapshot: `code-v4`.

- Diagnostics **26067262**: confirmed RUNNING on node3205.
- Training **26067267_0..3**: depends on successful diagnostics and preparation 26066770.
- Preparation **26066770**: remains RUNNING; no restart needed.
- Diagnostic callback **26067273**, webhook `5b4689da-27c8-4459-a1e0-ca6e683baae1`.
- Training callback **26067274**, webhook `24e3f139-c421-4605-bba9-1593551d65ed`.
- Both callback jobs verified pending on their dependencies; both monitors reactivated.
- Consumed v3 webhook registrations revoked.

## Diagnostic completion

26067262 completed successfully in 5m47s. All six origins, four optimizer smoke
checks, and online W&B qualification passed. Qualification run:
https://wandb.ai/ocean_emulators/default/runs/odh222fy . COMPLETE and QUALIFIED
verified; small report artifacts retrieved to `results/diagnostics`.
Diagnostic webhook retired. Training monitoring remains active; preparation was
still running at the completion check and training remained pending.

The reported GPU provenance replay RMSE 0.000396981 mixes physical units and is
NOT normalized RMSE. The normalized replay gate passed (<0.001), but its exact
value was not persisted by diagnose.py. The local report explains this distinction.
Report-only clarifications were added locally after retrieval; immutable cluster
code-v4 and the training settings were not changed.

## Single-target overfit moved to Engaging

User prohibited restarting the local GPU test after it disrupted the machine,
and explicitly requested running it on Engaging. Local test remains stopped.

- Job `26096610`, snapshot `code-overfit-v1`, confirmed pending (Priority).
- One GPU, 8 CPUs, 48 GiB host memory, 3-hour allocation.
- Same seed15 random initialization, full official v2 architecture, two input
  timestamps, MSE on first target only (2014-10-20), no autoregressive feedback.
- 5000 updates, cosine LR 6e-4 to 1e-7, fixed normalized wet-cell MSE.
- Uses already qualified archive-normalized diagnostic input window0 directly;
  no dependence on lost local /tmp input files.
- Atomic resumable model/optimizer checkpoints every100 updates. Separate output
  `$ROOT/overfit-one`; preserve interrupted local outputs as their own record.
- Callback job `26096657` afterany26096610, webhook
  `4947c088-cb08-44cf-8d44-e3ce4cc0432d`, six-hour reminder.
- Completion requires COMPLETE plus step5000 history/maps; evaluate both train-
  and eval-mode error and texture, produce learning curves and before/after maps.
  Finishing5000 updates does not establish zero error. Continue analysis from
  actual residuals; no scientific conclusion yet.

## Required post-run relocation (user instruction)

After all campaign writers finish (four training arms and their evaluations,
single-target overfit, resumes if any, and dependent callback jobs), move the
campaign run artifacts out of the group-shared-data filesystem.

Source: `/orcd/data/abodner/002/jrusak/om4-temporal-memory-20261016`.
Destination: remote `~/orcd/pool/om4-temporal-memory-20261016` (resolve on Engaging).
Check quota/capacity before copying. Preserve code snapshots, configs, training
outputs/checkpoints, diagnostics, maps, logs, analysis and experiment-specific
prepared data. Verify a full relative-file checksum manifest at destination
before deleting source copies. Preserve permissions on private callback files;
retire completed credentials instead of exposing them. Do not move or delete
shared datasets such as `/orcd/data/abodner/002/jrusak/om4_onedeg_v3`.
Update this record, reports, and any remaining monitoring paths after relocation.
Do not mark the campaign fully handled until this authorized cleanup completes.

Installed policy: `/home/jder/.codex/skills/engaging-train/SKILL.md`.
If Engaging SSH fails, stop and ask the user to refresh the Engaging login.

## Overfit test moved to Torch

User requested RTX/H200 scheduling on Torch. Submitted RTX job `20119912`
(account torch_pr_347_lzanna, partition rtx6000_lzanna, one GPU,8CPU48G,2h).
Inputs: four qualified normalized timestamps copied from Engaging diagnostic
window0, staged under `/scratch/jr7309/om4-temporal-memory-20261016`.
Code in `code/`, output `overfit-one/`, logs `logs/overfit-20119912.out/.err`.
Existing SIF `26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21` imported harness
successfully. Scientific settings unchanged; record runtime difference versus
Engaging. GPU utilization sampled every30s into `gpu-20119912.csv`.

Callback job `20119915` verified afterany20119912. Webhook
`e2932f56-a22d-4d35-bdab-9894f6a0993b`, quiet monitor `/root/diagnostics_monitor`,
6h reminder. Engaging duplicate26096610 and callback26096657 cancellation issued;
old webhook4947c088-cb08-44cf-8d44-e3ce4cc0432d revoked. Main4 stay on Engaging:
Torch queue estimates offered no demonstrated advantage over progressing runs.
Post-run Engaging relocation remains required for the campaign artifacts.

## Superseding migration instruction: checkpoint boundary, scratch, all user runs

User now requests migration at the upcoming planned checkpoint stop, BEFORE
resuming the four arms, rather than after all70epochs and rather than to pool.
Do not submit continuation jobs from the old shared-data root.

At callback for26067267 /26067274: verify all four writers exited and latest
checkpoints/NEEDS_RESUME exist; wait for callback26067274 to finish writing logs.
Inventory user-owned run artifacts beyond this campaign. Initial candidates:
- /orcd/data/abodner/002/jrusak/om4-temporal-memory-20261016
- /orcd/data/abodner/002/jrusak/llc-quarter-bench-20260928
- /orcd/data/abodner/002/cody/jrusak-kvikio-nsys
Destination: remote ~/orcd/scratch/<same-run-basename>.
All currently queued/running user jobs were the four arms plus callback26067274;
recheck at migration time. Leave shared OM4 and LLC source datasets in place.

Quota observed2026-10-17: scratch464/1024GB; pool536.8/1024GB. Refresh before
copying; account for sparse files. Copy preserving sparse files and permissions,
verify full relative file lists/sizes/checksums and symlink targets, then remove
only verified source run directories. Preserve verification manifests in scratch.
Update root paths in NEW continuation scripts, callback configuration, mutable
run metadata/references as appropriate; keep archived code provenance intact.
Resume all four arms from existing optimizer/EMA checkpoints, same W&B IDs and
70epoch endpoint, using scratch log/output paths. Register replacement callbacks
and verify bring-up. Retire obsolete callback registrations. This migration and
resume are explicitly authorized; no further permission needed unless ambiguity
about shared-data ownership or capacity prevents safe execution.

Torch20119912 completed5000 overfit updates (15m24s), COMPLETE verified.
Results retrieved into results/overfit-one, plots/report generated. Fine-scale
residual texture persists despite no temporal feedback; nonzero final MSE.
Callback20119915 failed network delivery; fallback reminder recovered completion.

Destination clarification from user: "my space", following ORCD guidance.
At checkpoint-boundary migration, use scratch for the active temporal-memory
campaign and its working inputs; use pool for completed older benchmark/profiling
artifacts. This supersedes the instruction to put all artifacts in scratch.
Migration remains BEFORE resuming current arms, with checksum verification.
Latest check: all four running at epochs36–37, elapsed29h03m; expected checkpoint
stop after about another5h47m. No live-directory move yet.

## Checkpoint boundary and verified-copy continuation (2026-10-18)

Login refreshed. Old array26067267 completed cleanly at34h50m; callback26067274
finished and user queue was empty. Checkpoints: full epoch42/group25,
detach epoch43/group1, latent_full epoch42/group81, latent_only epoch43/group24
(zero-based epoch counters). All latest.pt files present, ~1.3GB each.

Copy+fullSHA256 verification CPU job26279094:
- active campaign -> ~/orcd/scratch/om4-temporal-memory-20261016
- LLC benchmark -> ~/orcd/pool/llc-quarter-bench-20260928
- profiling -> ~/orcd/pool/jrusak-kvikio-nsys
Manifest folder ~/orcd/scratch/om4-migration-20261018.
Originals retained: auto-review rejected recursive deletion due to ambiguity
about exact targets, especially the directory under cody/. Do not bypass;
request exact-target removal approval AFTER verified copies are concrete.

Resume array26279118 depends afterok26279094 and COPIES_VERIFIED marker.
Harness ~/orcd/scratch/engaging-resume-scratch.sbatch; old immutable code-v4
retained; same runIDs, optimizer/EMA, scientific settings. All new outputs in
scratch. Logs ~/orcd/scratch/om4-resume-26279118_<arm>.out.
Copy callback26279143, webhook2fd22502-97eb-40d8-9dae-789fbf72bcc7.
Resume callback26279144, webhook69e0afde-1409-45fd-8607-96085bf4fd6a.
Both have1h fallback reminders; old training webhook24e3f139 retired.
Follow-up must verify actual copy success, resumed optimizer logging, final
70epochs+heldout completion, and obtain approval for remaining source cleanup.

## Scope correction: exclude Cody space entirely

User explicitly: "Don't copy or touch anything in Cody's space. Just runs under
my dir in002." Stopped26279094 while its log showed VERIFY of the first campaign;
it had not reached the Cody entry. No further access to Cody space is authorized.
Old resume26279118 and callbacks26279143/26279144 canceled; old webhooks revoked.

Replacement copy+SHA256 verification+authorized deletion job26280108 uses ONLY:
/orcd/data/abodner/002/jrusak/om4-temporal-memory-20261016 -> ~/orcd/scratch/same-name
/orcd/data/abodner/002/jrusak/llc-quarter-bench-20260928 -> ~/orcd/pool/same-name
Source deletion explicitly approved for user's run dirs, allowed after verified
copy; script enforces source parent == /orcd/data/abodner/002/jrusak.
Continuation26280150 depends afterok26280108 and COPIES_VERIFIED marker.
Copy webhook4796ea4e-6519-4693-a9a1-a74272e5f606, reminder1h.
All earlier plans mentioning Cody directory are superseded and must not execute.
Replacement copy callback26280158; continuation callback26280159.
Resume webhook b1d7efe0-c681-45f9-9ea9-ef20c936f88e, reminder1h.

Migration26280108 verified COMPLETED exit0 (11m16s). Both full checksum manifests
matched; both original run directories under002/jrusak removed and absence
confirmed. Active campaign now ~/orcd/scratch/om4-temporal-memory-20261016;
completed benchmark now ~/orcd/pool/llc-quarter-bench-20260928. Shared datasets
unchanged. No Cody-space operations in replacement job. Copy webhook retired.
Continuation26280150 pending at this check; startup and completion monitoring
remain active on b1d7efe0-c681-45f9-9ea9-ef20c936f88e.

Maintenance correction: scheduler reservation monthly_maint runs2026-10-20
00:00–21:00 Eastern. At2026-10-18 21:18,36h request could not fit before it.
Updated pending array26280150 TimeLimit to24h in place; same checkpoints/code/
callbacks/scientific endpoint. Existing B:USR1@600 handler checkpoints near limit.
Remaining estimated training22–23h plus heldout evaluation; additional resume
may be required if limit reached. Test-only12/24h estimates overlapped maintenance
and must not be presented as reliable. No compute reservation requested/used.

User requests capacity review at24h before maintenance if not started.
Rearmed continuation webhook b1d7efe0-c681-45f9-9ea9-ef20c936f88e specifically for
2026-10-19 00:00 Eastern (2026-10-19 04:00 UTC), replacing6h timer.
On that reminder: check each26280150 arm. For any still pending, investigate
Torch RTX/H200 and Beta production scheduling, environment/checkpoint portability,
transfer time and total expected time-to-finish. Preserve progressing arms and
avoid duplicate training. No reservations without user approval, no local GPU
training, and stop for login refresh if required. Report concrete alternatives;
checkpoint/data transfer plus queue/runtime matters, not idleGPU count alone.

At the2026-10-19 midnight Eastern cutoff, all four26280150 tasks were RUNNING
for~2h41m and had resumed optimizer logging at epochs46–47. No cross-cluster
move required under user's conditional request. Six-hour monitoring rearmed.

## Final four-arm completion

All26280150 tasks COMPLETED exit0; all four COMPLETE/EVAL_COMPLETE/heldout.json
verified. Continuation times22h39m46s,21h29m01s,22h03m28s,21h30m44s. Retrieved
70validation records/config/heldout metrics per arm into results/comparison.
Generated results/comparison/REPORT.md, summary.csv, comparison.png/pdf using
compare_arms.py. All70epochs completed; validation-selected EMA epochs29/30/31/31.
At day720 detach vsfull MSE−20.83%; latent_only−21.46%; latent_full−0.31%.
One paired seed and correlated origins; no historical-information mechanism proof.
All scheduled follow-ups retired after verifying completed workload artifacts.
Active campaign artifacts are in the user's scratch space; original002/jrusak
run directories already checksum-verified and removed, shared datasets unchanged.

## Fine-latent and epoch-70 comparison extension (2026-10-19)

User authorized fine-latent, fine-latent-detach, global/quiet-region maps, and
selected-versus-70epoch EMA comparison for all arms. Source staged separately as
scratch root/code-fine-v1, preserving original code-v4. Source checksums in its
SOURCE_SHA256SUMS. Eight CPU mechanism tests passed locally; no local GPU work.

GPU pilot26429562 gates production via FINE_QUALIFIED + afterok dependency.
New training array26429592 (fine_latent/fine_latent_detach), begin2026-10-20
21:05Eastern after verified monthly maintenance00:00–21:00. Continuation26429593
resumes afterok26429592; each segment36h limit, checkpoint exit after125400s.
Existing-four-arm selected/final evaluation array26429563 started immediately.
Logs under scratch root/logs-fine. All new outputs under personal scratch.

Quiet evaluation webhook b58a58dd-3b27-4670-ba91-4e32d6e1fb62, initial20m reminder,
monitor /root/checkpoint_maps_monitor. Training webhook
45543461-72b0-46b4-8c49-da00fc89bde9, initial30m reminder, monitor
/root/fine_training_monitor. Parentthread01a144d3-d415-7072-8cf0-ff06eb2db174.
Automatic approval review rejected remote callback credential transfer; tokens
remain local. Active local systemd services om4-epoch70-watch-26429563 and
om4-fine-watch-26429593 poll Slurm every5min and invoke the local callback sender
on termination or SSH failure. An SSH failure ends that watcher; monitoring must
request login refresh. Durable reminders cover watcher/reboot/network failures.
No remote callback jobs or credential copies were created.

Completion requires both new arms70epochs plus selected/epoch70 evaluations,
all four existing-arm comparison evaluations, matching selected metrics against
prior records, map generation and inclusion in the experiment/results PR938.
This entry is a submission record, not a claim of successful training/results.

Evaluation26429563_0..3 verified COMPLETED exit0, runtime1m49s–2m57s. All8
selected/final evaluations retrieved; selected hashes and entire metric records
exactly match original report. Published results/checkpoint-comparison with24
maps, regional metrics, final-vs-selected summary and checksums. Fixed plotting
region-mask broadcasting (`box = box & ...`); figures use this corrected logic.
Final EMA day720 worsens14–18% acrossarms; detachment advantage remains~22%.
Fine pilot26429562 completed exit0 in1m47s; FINE_QUALIFIED verified. Both new
training arms remain queued until after maintenance; no fine-arm result yet.
Only the completed eval callback is retired; fine-training monitoring remains.
