---
name: samudra-viewer
description: Maintain the Samudra observation-results Panel viewer and its fomo-bot user service. Use for viewer UI, data-display, deployment, or service fixes from the bot's supplied Samudra checkout.
---

<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra viewer

## Locations and ownership

- Repository: `https://github.com/m2lines/Samudra.git`
- Integration and deployment branch: **`codex/observation-viewer`**.
- Each change uses its own **`codex/viewer-<change>`** feature branch.
- The bot receives a fresh Samudra working checkout with each message. Use that
  checkout for edits; create a feature branch from the current integration branch.
- App: `apps/observation_viewer/` (standalone uv project).
- Live checkout: `/home/fomo-bot/samudra_viewer`.
- Live data: `/home/fomo-bot/samudra_viewer/apps/observation_viewer/.data`.
  This verified report bundle is ignored by Git and required by the service.
- Owner: `fomo-bot`; user service: `samudra-observation-viewer.service`.
  It listens on all IPv4 interfaces at port **61015** and restarts on failure.
- Installed unit: `/home/fomo-bot/.config/systemd/user/samudra-observation-viewer.service`.
- Private configuration: `/home/fomo-bot/.config/samudra-viewer/environment`.
  `BOKEH_ALLOW_WS_ORIGIN` is a comma-separated host:port list. Read this local
  file when the forwarded URL or origin is needed. **Do not put the public URL,
  hostname, or private environment file in Git, commit messages, or PR text.**
- Skill source: `apps/observation_viewer/deploy/samudra-viewer/SKILL.md`.
  Installed copy: `/home/fomo-bot/.codex/skills/samudra-viewer/SKILL.md`.

## Edit, commit, push, then deploy

Run as `fomo-bot`. The user's requested workflow is to publish changes to this
branch before updating the live service; do not edit the live checkout directly.

1. In the supplied working checkout, inspect `git status` and the origin URL.
   Fetch the integration branch and create an isolated feature branch:

   ```bash
   git fetch origin codex/observation-viewer
   git switch -c codex/viewer-<change> origin/codex/observation-viewer
   ```

   Use a unique feature name for each task. Preserve unrelated work. Never
   develop in the live checkout, discard another task's changes, or force-push
   the integration branch.

2. Read `apps/observation_viewer/README.md` and make the requested changes in
   this working checkout. Use `/home/fomo-bot/.local/bin/uv` if uv is not on PATH.
   Install and check only this lightweight project, not the repository's
   training environment:

   ```bash
   uv sync --frozen --project apps/observation_viewer --group dev
   uv run --project apps/observation_viewer ruff check apps/observation_viewer
   uv run --project apps/observation_viewer ruff format --check apps/observation_viewer
   uv run --project apps/observation_viewer pytest \
     -c apps/observation_viewer/pyproject.toml apps/observation_viewer/tests
   ```

   For interactive changes, exercise actual browser controls. The browser tests
   accept `SAMUDRA_VIEWER_TEST_URL` and `CHROMIUM_EXECUTABLE`. Set
   `SAMUDRA_VIEWER_DATA` to the live data directory above when testing from the
   working checkout. A temporary localhost server on a free port can use that
   read-only bundle; do not compete with the production port or run inference.

3. Review the diff for accidental data, environment files, or public hostnames.
   Stage and commit the intended files. Publish the feature branch with
   `git push -u origin HEAD`; the **Viewer tests** workflow runs its self-contained
   suite without production data. Browser tests are part of the same suite and
   run when a test server and matching data are configured; skipped browser tests
   do not count as browser validation.

   Before integrating, fetch `codex/observation-viewer` again. If it advanced,
   rebase the feature branch onto it, resolve conflicts, and rerun the suite on
   that combined tree. Publish with a normal fast-forward push:
   `git push origin HEAD:refs/heads/codex/observation-viewer`.
   If rejected, repeat integration and testing; never force the push. Verify
   that the remote branch points to the tested commit before deployment. If Git
   author identity is missing, use repository-local `user.name=fomo-bot` and
   `user.email=266121006+fomo-bot@users.noreply.github.com`.

4. Pull the published branch into the separate live checkout and sync its
   runtime. Confirm the live checkout is clean and on the expected branch first:

   ```bash
   cd /home/fomo-bot/samudra_viewer
   git pull --ff-only origin codex/observation-viewer
   /home/fomo-bot/.local/bin/uv sync --frozen --no-dev --project apps/observation_viewer
   ```

   Check that `git rev-parse HEAD` equals the commit just pushed. Keep `.data`
   and the private environment file intact; never use `git clean -fdx` here.
   If the unit changed, install the versioned unit and run daemon-reload:

   ```bash
   install -m 644 apps/observation_viewer/deploy/samudra-observation-viewer.service \
     /home/fomo-bot/.config/systemd/user/samudra-observation-viewer.service
   systemctl --user daemon-reload
   ```

   If this skill changed, copy its source folder to the installed skill folder.
   Then restart and verify:

   ```bash
   systemctl --user restart samudra-observation-viewer
   systemctl --user is-active samudra-observation-viewer
   journalctl --user -u samudra-observation-viewer -n 30 --no-pager
   curl --fail --silent --output /dev/null http://127.0.0.1:61015/app
   ```

   Verify a live browser session and an affected interaction, including the
   WebSocket connection; HTTP success alone does not prove the viewer works.
   A WS 403 usually means the browser's exact host:port is missing from the
   private origin list. Preserve the list and add only the requested origin.
   If the user bus is unavailable in a non-login shell, set
   `XDG_RUNTIME_DIR=/run/user/$(id -u)` for the systemctl/journalctl command.

## Scientific and operational constraints

The following missing-data restriction applies to `fomo-bot`. It does not
restrict agents running as `jder` from acquiring or generating data when jder
authorizes that work.

When a request to `fomo-bot` needs variables, depths, times, samples, or other
information that the available viewer data does not contain, reply with exactly what data
is missing and what is available. Do not search other stores, machines, chats,
or archives for it, download additional data, run inference, or substitute
another quantity. Only acquire or generate missing data if the user explicitly
asks for that as a separate or additional task.

You may offer to create a GitHub issue in `m2lines/Samudra`, assigned to `jder`,
for the missing data. Describe the missing fields, depths, times, or coverage,
the viewer capability they would enable, and acceptance criteria for the data
deliverable. Create the issue only when the user approves or explicitly asks;
do not ask again if that authorization is already present. Keep private
deployment URLs and configuration out of the issue.

The app displays saved observation-report arrays; it does not run training or
inference. Preserve masks, units, five-day forecast intervals, checkpoint
provenance, and the distinction between IAP monthly context and instantaneous
truth. Keep data out of Git and retain the local catalog/receipts. The fixed
2:1 aspect applies to map frames, not the full figure including axes/color bars.

Report the pushed/deployed commit, validation, and service state. If publication
or deployment fails, stop at that failed step and diagnose it; keep the existing
service running where possible instead of repeatedly restarting it.
