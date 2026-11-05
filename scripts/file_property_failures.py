# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare deduplicated issue reports from JUnit; only --apply writes to GitHub.

One issue per property, not per random seed. Closed issues stay closed: a human
must decide whether a recurrence merits reopening or a different invariant.
Collection errors and runner failures remain CI failures, not product bug reports.
"""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def reports(path, run_url):
    """Extract stable identities and bounded counterexamples from actual failures."""
    found = {}
    for case in ET.parse(path).iter("testcase"):
        if any(
            prop.get("name") == "contract_probe" and prop.get("value") == "true"
            for prop in case.findall("./properties/property")
        ):
            continue
        failure = case.find("failure")
        if failure is None:
            continue
        identity = f"{case.get('classname')}.{case.get('name')}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        marker = f"<!-- property-failure:{digest} -->"
        found[marker] = {
            "title": f"Property failure: {identity}"[:240],
            "body": (
                f"{marker}\n\nProperty: `{identity}`\n\n"
                f"Run and replay artifacts: {run_url}\n\n"
                "Review the oracle before classifying this as a production bug. "
                "The artifact contains the seed, revision, environment, JUnit, "
                "Hypothesis statistics and shrunk counterexample.\n\n"
                # Indented code keeps even generated backticks inside the code block.
                + "\n".join(
                    "    " + line for line in (failure.text or "")[:16000].split("\n")
                )
            ),
        }
    return found


def gh(*args, input=None):
    return subprocess.check_output(["gh", *args], input=input, text=True)


def file_new_reports(candidates, repo, invoke=gh):
    """File each property once, preserving closed issues and generated text."""
    if not candidates:
        return
    pages = json.loads(
        invoke(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/issues?state=all&per_page=100",
        )
    )
    existing = [
        issue for page in pages for issue in page if "pull_request" not in issue
    ]
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
    parser.add_argument("junit", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    candidates = reports(args.junit, args.run_url)
    if not args.apply:
        print(json.dumps(list(candidates.values()), indent=2))
        return
    file_new_reports(candidates, args.repo)


if __name__ == "__main__":
    main()
