---
name: samudra-viewer
description: Maintain the Samudra observation-results Panel viewer and its fomo-bot user service. Use for viewer UI, data-display, deployment, or service fixes from the bot's supplied Samudra checkout.
---

<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra viewer

## Locations and ownership

- Repository: `https://github.com/m2lines/Samudra.git`
- Development and deployment branch: **`codex/observation-viewer`**.
- The bot receives a fresh Samudra working checkout with each message. Use that
  checkout for edits; switch to the branch before changing files.
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
   Fetch and switch to the branch, then fast-forward it:

   ```bash
   git fetch origin codex/observation-viewer
   git switch codex/observation-viewer
   git pull --ff-only origin codex/observation-viewer
   ```

   If the local branch does not exist, use
   `git switch -c codex/observation-viewer FETCH_HEAD` after fetching. Preserve
   unrelated work. If branch checkout or fast-forward fails, inspect the cause;
   do not discard changes, force-push, or deploy a divergent tree.

2. Read `apps/observation_viewer/README.md` and make the requested changes in
   this working checkout. Use `/home/fomo-bot/.local/bin/uv` if uv is not on PATH.
   Install and check only this lightweight project, not the repository's
   training environment:

   ```bash
   uv sync --frozen --project apps/observation_viewer --group dev
   uv run --project apps/observation_viewer ruff check apps/observation_viewer
   uv run --project apps/observation_viewer ruff format --check apps/observation_viewer
   uv run --project apps/observation_viewer pytest \
     -c apps/observation_viewer/pyproject.toml apps/observation_viewer/tests/test_data.py
   ```

   For interactive changes, exercise actual browser controls. The browser tests
   accept `SAMUDRA_VIEWER_TEST_URL` and `CHROMIUM_EXECUTABLE`. Set
   `SAMUDRA_VIEWER_DATA` to the live data directory above when testing from the
   working checkout. A temporary localhost server on a free port can use that
   read-only bundle; do not compete with the production port or run inference.

3. Review the diff for accidental data, environment files, or public hostnames.
   Stage the intended files, commit, then
   `git push origin HEAD:refs/heads/codex/observation-viewer`.
   Verify the remote branch points to the new commit before deployment. If Git
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

The app displays saved observation-report arrays; it does not run training or
inference. Preserve masks, units, five-day forecast intervals, checkpoint
provenance, and the distinction between IAP monthly context and instantaneous
truth. Keep data out of Git and retain the local catalog/receipts. The fixed
2:1 aspect applies to map frames, not the full figure including axes/color bars.

Report the pushed/deployed commit, validation, and service state. If publication
or deployment fails, stop at that failed step and diagnose it; keep the existing
service running where possible instead of repeatedly restarting it.
