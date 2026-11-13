# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare read-only triage and manual issue reports; only --apply writes to GitHub.

One issue per property, not per random seed. Closed issues stay closed: a human
must decide whether a recurrence merits reopening or a different invariant.
Collection errors and runner failures remain CI failures, not product bug reports.
"""

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

from .reports import Outcome, read_run


def reports(path, run_url):
    """Extract stable identities and bounded counterexamples from actual failures."""
    found: dict[str, dict[str, str]] = {}
    run = read_run(path)
    if not run.complete:
        return (
            found  # Interrupted evidence is an execution problem, not a product report.
        )
    for case in run.cases:
        if case.contract_probe or case.outcome != Outcome.FAILED:
            continue
        if "verification_suite" not in case.properties:
            raise ValueError(
                f"{case.nodeid} has no verification_suite; its conftest must call "
                "tag_verification_items"
            )
        suite = case.properties["verification_suite"]
        identity = f"{suite}:{case.nodeid}"
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
            if "pull_request" not in issue and marker in (issue.get("body") or "")
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
        f"# Property triage: {html.escape(summary.get('lane', 'manual'))}",
        "",
        f"Run: {summary['run_url']}",
        "",
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
    if counts := summary.get("pytest"):
        lines.extend(
            [
                "",
                f"Pytest: {counts['failed']} failures, {counts['error']} errors, "
                f"{counts['skipped']} skipped; {counts['contract_probes']} contract probes excluded.",
            ]
        )
    return "\n".join(lines) + "\n"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--lane",
        default="manual",
        help="Regression/backend or canary lane shown in triage",
    )
    parser.add_argument(
        "--triage-output",
        type=Path,
        help="Write read-only triage JSON and Markdown to a new directory",
    )
    args = parser.parse_args()
    candidates = reports(args.report, args.run_url)
    if args.triage_output:
        if args.apply:
            parser.error(
                "--triage-output is read-only and cannot be combined with --apply"
            )
        summary = triage_summary(
            candidates, lookup_issues(args.repo) if candidates else [], args.run_url
        )
        summary["lane"] = args.lane
        run = read_run(args.report)
        summary["pytest"] = {
            **run.counts(),
            "collection_errors": len(run.collection_errors),
            "complete": run.complete,
            "contract_probes": sum(
                case.contract_probe and case.outcome == Outcome.FAILED
                for case in run.cases
            ),
        }
        args.triage_output.mkdir(parents=True, exist_ok=False)
        (args.triage_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (args.triage_output / "summary.md").write_text(render_summary(summary))
        print(render_summary(summary))
        return
    if not args.apply:
        print(json.dumps(list(candidates.values()), indent=2))
        return
    file_new_reports(candidates, args.repo)


if __name__ == "__main__":
    main()
