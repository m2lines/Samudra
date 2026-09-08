---
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

name: scrub-main-ci
description: Daily investigation and repair of main CI failures, checking existing GitHub issues and PRs before starting a fix.
schedule_cron: "0 0 9 ? * *"
schedule_tz: America/New_York
---

# Main CI scrub

Advance the `main-ci` goal in `AGENTS.md`. Run daily at 09:00
America/New_York through the repository scrub harness. The harness discovers this
skill on `origin/main`; its cron controls the next daily run, while the footer
below controls follow-up turns within the current run.

## Inspect current failures

Use live GitHub Actions data for `m2lines/Samudra` and a clean checkout of current
`origin/main`. Preserve unrelated work by using a separate branch or worktree.
Record the main SHA and inspect all main workflows, including builds, containers,
tests, benchmarks, documentation, and releases.

Start with the last seven days of runs and the latest run of each workflow even
if it is older. Paginate or query by workflow until the window is covered; a
single limited list can miss runs. A useful starting query is:

```bash
gh run list --repo m2lines/Samudra --branch main --limit 100 \
  --json databaseId,workflowName,event,headSha,status,conclusion,createdAt,url
```

For unsuccessful completed runs, including startup failures and timeouts, inspect
the current attempt's jobs, failing steps, annotations, and logs. Use
`gh run view RUN_ID --repo m2lines/Samudra --json jobs`
and `gh api repos/m2lines/Samudra/actions/jobs/JOB_ID/logs` when needed. Group
repeated failures by workflow, job, failing test or command, and diagnostic error;
an aggregate status job often only reports an upstream failure.

Check later applicable runs before acting. A passing build-only PR run does not
verify main-only container tests, and a skipped job does not verify recovery.
Distinguish active regressions, recurring flakes, recovered historical failures,
pending runs, and intentional cancellation or approval gates. If a workflow has
no usable results, report the coverage gap instead of calling it green.

## Check existing work before making changes

For each distinct active or recurring failure, search both GitHub issues and PRs
using the error text, failing test or file, workflow/job name, and likely cause.
Read open candidates' descriptions and recent comments; inspect matching PR
diffs, head SHAs, and current-head checks. Search recently closed issues and
merged or closed PRs too, since a fix may have landed after the failed run.
Confirm the candidate's diagnosis and affected check match; keyword hits alone
do not establish an existing repair.

- **Matching open PR:** use it as the repair's tracking record. Avoid a competing
  PR. If it already contains the fix, verify its checks and wait for applicable
  main validation. Continue a scrub-owned repair branch when further changes are
  needed; contribute to another author's branch only when that work is authorized.
- **Matching issue:** read its current plan and linked PRs. If someone is actively
  fixing it, avoid duplicating their work. An open issue alone is not a reason to
  stop: if no repair is underway and a scoped fix is clear, implement it in a PR
  linked to that issue.
- **Merged fix or closed issue:** compare the fix with the failing SHA and later
  applicable main runs. A pre-fix failure may only need verification; a matching
  failure after the fix needs investigation and a link to the earlier work.
- **No match:** investigate and fix it, or create one issue with evidence and a
  concrete next action if a fix cannot be completed.

Recheck for newly opened or updated matching work immediately before creating a
PR or issue. Persist useful findings in the matching record only when they add
new evidence, change the diagnosis, or provide an actionable next step; do not
post identical daily comments. If an issue or PR has gone quiet, reassess its
plan and current failure evidence instead of treating it as a permanent exemption.

## Repair and validate

Distinguish repository regressions from runner provisioning, unavailable services,
and transient failures. `InsufficientInstanceCapacity` before any GPU tests ran
is AWS provisioning; check the existing capacity issue and recent successful
launches before proposing an instance change. It must not hide other failures in
the same container workflow.

For a plausibly transient test/build failure, preserve the log and environment
details and attempt a focused local reproduction when practical. A CI rerun is
appropriate only after checking for another active attempt and reviewing the
failed jobs' side effects. Retry failed jobs at most once per run during this
scrub, and use existing issue comments or run attempts to avoid daily retry loops.
Release or publication jobs require their existing retry authorization; do not
rerun them merely to make the dashboard green. A successful retry can still be
evidence of a recurring flake.

For a confirmed, unclaimed repository problem, make a focused repair from current
main, run the failing check and appropriate regression/quality checks, then commit,
push, and open a PR. Keep unrelated causes in separate PRs. Include the failing
run/job links, diagnosis, validation results, and related issues or PRs in the
description. Do not disable tests, weaken assertions, add blanket skips, or mask
failures with `continue-on-error` to obtain a green status. Do not merge repair
PRs or change branch protection as part of this scrub.

Match validation to the failing environment. Container packaging failures need
tests against the built image's own files, without a checkout mount that could
supply missing assets. CPU validation on this GPU host can use
`CUDA_VISIBLE_DEVICES= uv run pytest -m "not manual and not cuda"`. Data-package
tests run from `data/` with `uv run --extra test pytest ...`. Report architecture,
CUDA, service, or permission limits explicitly when they prevent verification.

If blocked, reuse a matching issue or PR, or file one issue after the duplicate
check. Include the run URL, SHA, failing step/error, reproduction or retry results,
and the exact next action. Respect existing repository permissions and approval
requirements; the scrub does not grant access to credentials or external systems.

## Finish or follow up

Summarize what was inspected, each distinct failure's disposition, repair/tracking
links, and checks completed or still pending. When everything is healthy or
already tracked with no new evidence, keep the result brief and avoid GitHub writes.

A run is done when every identified failure is recovered, has a validated repair
PR, or has a concrete handoff in an existing/new issue or PR. An unchanged known
failure can be left for the next daily run without repeated reminders. A repair
PR being open does not mean main has recovered; verify the relevant main jobs
after the fix lands on a later scrub.

End each spawned scrub turn with exactly one footer:

```text
HARNESS_SCRUB_LOOP {"needs_followup_at":null}
```

Use a future RFC 3339 timestamp with an offset when current work needs another
turn: about two minutes for active work, or the expected completion time for CI
or another external wait. If an issue handoff fails, schedule a follow-up to retry
it. Once the run's work or handoff is complete, use `null`; the daily cron remains
active independently.
