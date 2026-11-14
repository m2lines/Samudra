# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Match a campaign's failing properties to GitHub issues; only --apply writes to GitHub.

One issue per property, not per random seed. Closed issues stay closed: a human
must decide whether a recurrence merits reopening or a different invariant.
Collection errors and runner failures remain CI failures, not product bug reports,
but the summary states them, and --apply refuses to file from an incomplete run.
"""

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

from .evidence import Outcome, PytestRun, read_run
from .spec import CampaignSpec


def reports(run: PytestRun, run_url):
    """Extract stable identities and bounded counterexamples from actual failures."""
    found: dict[str, dict[str, str]] = {}
    if not run.complete:
        return (
            found  # Interrupted evidence is an execution problem, not a product report.
        )
    for case in run.cases:
        if case.proposed or case.outcome != Outcome.FAILED:
            continue
        identity = f"{case.project}:{case.nodeid}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        marker = f"<!-- property-failure:{digest} -->"
        found[marker] = {
            "title": f"Property failure: {identity}"[:240],
            "body": (
                f"{marker}\n\nProperty: `{identity}`\n\n"
                f"Run and replay artifacts: {run_url}\n\n"
                "Review the oracle before classifying this as a production bug. "
                "The artifact contains the seed, revision, environment, native pytest events, "
                "Hypothesis statistics and shrunk counterexample.\n\n"
                # Indented code keeps even generated backticks inside the code block.
                + "\n".join("    " + line for line in case.details[:16000].split("\n"))
            ),
        }
    return found


def gh(*args, input=None):
    return subprocess.check_output(["gh", *args], input=input, text=True)


def lookup_issues(repo, invoke=gh):
    pages = json.loads(
        invoke(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/issues?state=all&per_page=100",
        )
    )
    return [issue for page in pages for issue in page if "pull_request" not in issue]


def triage_summary(candidates, existing, run_url):
    """Keep every recurrence visible without creating or reopening any issues."""
    failures = []
    for marker, report in candidates.items():
        matches = [
            {
                "number": issue["number"],
                "state": issue["state"],
                "url": issue["html_url"],
            }
            for issue in existing
            if marker in (issue.get("body") or "")
        ]
        failures.append(
            {"marker": marker, "title": report["title"], "existing_issues": matches}
        )
    return {
        "run_url": run_url,
        "total": len(failures),
        "new": sum(not failure["existing_issues"] for failure in failures),
        "recurrences": sum(bool(failure["existing_issues"]) for failure in failures),
        "issues_created": 0,
        "failures": failures,
    }


def render_summary(summary):
    lines = [
        f"# Property triage: {html.escape(summary['campaign'])}",
        "",
        f"Run: {summary['run_url']}",
        "",
        *incomplete_warning(summary["pytest"]),
        f"{summary['total']} failing properties; {summary['new']} new; "
        f"{summary['recurrences']} recurrences. No issues created.",
        "",
    ]
    for failure in summary["failures"]:
        links = (
            ", ".join(
                f"[#{issue['number']} ({issue['state']})]({issue['url']})"
                for issue in failure["existing_issues"]
            )
            or "new, needs triage"
        )
        lines.append(f"- <code>{html.escape(failure['title'])}</code>: {links}")
    counts = summary["pytest"]
    lines.extend(
        [
            "",
            f"Pytest: {counts['failed']} failures, {counts['error']} errors, "
            f"{counts['skipped']} skipped, {counts['collection_errors']} collection "
            f"errors; {counts['proposed']} proposed invariants excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


def incomplete_warning(counts):
    if counts["complete"]:
        return []
    return [
        f"**Incomplete run:** pytest never finished, so the {counts['failed']} "
        "failures recorded before it stopped are not triaged. Check the "
        "campaign's `result.json` and `pytest.log`.",
        "",
    ]


def file_new_reports(candidates, repo, invoke=gh):
    """File each property once, preserving closed issues and generated text."""
    if not candidates:
        return
    existing = lookup_issues(repo, invoke)
    for marker, report in candidates.items():
        if any(marker in (issue.get("body") or "") for issue in existing):
            continue
        # Structured stdin preserves generated content without shell interpolation.
        invoke(
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            report["title"],
            "--body-file",
            "-",
            input=report["body"],
        )


def pytest_counts(run: PytestRun):
    return {
        **run.counts(),
        "collection_errors": len(run.collection_errors),
        "complete": run.complete,
        "proposed": sum(
            case.proposed and case.outcome == Outcome.FAILED for case in run.cases
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaigns", nargs="+", type=Path, help="Campaign artifact directories"
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--triage-output",
        type=Path,
        help="Write read-only triage JSON and Markdown to <dir>/<campaign directory name>",
    )
    args = parser.parse_args()
    if args.triage_output and args.apply:
        parser.error("--triage-output is read-only and cannot be combined with --apply")
    names = [campaign.name for campaign in args.campaigns]
    if len(set(names)) != len(names):
        parser.error("Campaign directories must have distinct names")
    runs = {
        campaign: read_run(campaign / "pytest.jsonl") for campaign in args.campaigns
    }
    candidates = {
        campaign: reports(run, args.run_url) for campaign, run in runs.items()
    }
    if args.apply:
        if incomplete := [str(c) for c, run in runs.items() if not run.complete]:
            raise SystemExit(
                f"Refusing to file issues from incomplete runs: {incomplete}"
            )
        file_new_reports(
            {m: r for found in candidates.values() for m, r in found.items()}, args.repo
        )
        return
    if not args.triage_output:
        print(
            json.dumps(
                [report for found in candidates.values() for report in found.values()],
                indent=2,
            )
        )
        return
    # One listing of the repository's issues serves every campaign.
    existing = lookup_issues(args.repo) if any(candidates.values()) else []
    for campaign, run in runs.items():
        summary = triage_summary(candidates[campaign], existing, args.run_url)
        summary["campaign"] = CampaignSpec.read(campaign).label
        summary["pytest"] = pytest_counts(run)
        output = args.triage_output / campaign.name
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "summary.md").write_text(render_summary(summary))
        print(render_summary(summary))


if __name__ == "__main__":
    main()
