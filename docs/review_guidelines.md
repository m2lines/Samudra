# Review Guidelines

This checklist captures recurring review themes from recent Ocean Emulator PR reviews.
Use it to keep review feedback consistent and to separate blocking issues from nits.

## Severity Levels

- `BLOCKER`: Must be resolved before merge. Use this for correctness, safety, or clear regression risk.
- `SUGGESTION`: Should usually be addressed, but may be deferred with rationale.
- `:sheep:`: Nit. Style/readability/cleanup only; non-blocking unless it hides a real bug.

If a `:sheep:` comment reveals a correctness, performance, or maintainability hazard, promote it to `BLOCKER`.
You do not need explicit `BLOCKER:`/`SUGGESTION:` prefixes; conversational wording is fine.

## Blocking Checklist

- [ ] **Correctness and invariants:** Shape semantics, masking/normalization behavior, step semantics, and edge cases are correct and explicitly guarded where needed.
- [ ] **No silent behavior drift:** Default behavior and compatibility are preserved, or changes are explicit and justified.
- [ ] **Abstractions are honest:** APIs do not hide permanent caching, mutation, or side effects behind vague names.
- [ ] **Performance/device safety:** No accidental extra CPU/GPU transfers, repeated per-forward recompute, unnecessary copies, or eager loads in hot paths.
- [ ] **Config/type/schema safety:** Configuration has one clear path for behavior, avoids ambiguous fallbacks, and uses strong typing/validation where possible.
- [ ] **Tests prove behavior:** Tests cover the changed behavior and likely regressions; they do not only restate implementation internals.
- [ ] **Failure and observability:** Error/log messages are actionable; failure modes are explicit rather than silent.
- [ ] **Non-obvious logic is explained:** Surprising orderings, dimension conventions, and assumptions are documented.

## `:sheep:` Nit Checklist

- [ ] Naming/readability polish that does not change behavior.
- [ ] Optional refactors for local clarity or deduplication.
- [ ] Small doc/comment clarifications.
- [ ] Optional type-hint tightening.
- [ ] Minor test cleanup or fixture cleanup.
- [ ] File organization or constant placement cleanup.

## Reviewer Workflow

1. Triage severity first: identify blockers before nits.
2. Focus on high-risk paths first: `src/ocean_emulators/utils/data.py`, `src/ocean_emulators/datasets.py`, `src/ocean_emulators/config.py`, training/eval loops, and tests.
3. For each blocker, explain risk and expected fix in one concrete sentence.
4. Mark non-blocking polish as `:sheep:` and avoid blocking merge on it.
5. If deferring a suggestion, ask for a follow-up issue/PR note.
