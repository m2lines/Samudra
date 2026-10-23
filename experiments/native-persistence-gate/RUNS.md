<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native gated Samudra run

- Source: `6a19fc3c63d56eaeaf756eb734b2a928b1782e97`, built on historical control
  revision `a20c444849ab39575089f8e653bb128dd2b473a6`.
- Beta job `216469`, four B200 GPUs requested, production beta/standard,
  24-hour allocation; one gated run only.
- Root: `/projects/ny/lz1955/multiscale/jrusak/runs/native-persistence-gate-20261023`.
  Source archive checksum checked in batch; full-sized GPU qualification runs
  before the ordinary distributed trainer. Training configuration and eval
  configuration are copied to root; source tree is pinned separately.
- Existing ungated control: Beta `210664`, W&B
  https://wandb.ai/ocean_emulators/default/runs/uaei2vbr;
  source/config and completed epoch-70 artifact/log presence verified live.
- New W&B project: ocean_emulators/tocean. New URL pending actual initialization.
- Local focused tests and config audit passed. All commit hooks passed, including
  Ruff, mypy, schema checks and REUSE. GPU runtime validation pending queue start.
- Callback `eb07fd47-69ac-4699-ad72-ed6417db825c`, original thread
  `01a163c0-be00-72a1-bc5f-c5b22b471dbd`. Exit callback installed; private directory
  mode0700 and credential0600; login-node connectivity check passed. Fallback
  reminder covers node failure/queued cancellation, which an exit trap cannot.
- Completion requires training plus post-training sweep and comparison against
  the existing control; do not infer success from callback alone.
