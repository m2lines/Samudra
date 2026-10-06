<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# CI runner authentication

All EC2 launcher jobs use `workflows/ec2-runner.yml`: GPU tests, benchmarks,
and the PhysicsNeMo x86 build, GPU tests, and ARM build/tests. Add future EC2
launchers through this workflow so ECR Public base-image pulls use AWS
credentials instead of the anonymous quota.

The workflow assumes `vars.EC2_LAUNCH_ROLE` through GitHub OIDC and logs Docker
into `public.ecr.aws` before invoking the checked-out `ec2-gha` Docker action.
A remote Docker action would be built during action preparation, before login.
Both the action checkout and its bootstrap scripts use the same pinned revision.
Update both `ref` and `action_ref` together when upgrading `ec2-gha`.

The launch role needs these permissions in addition to its existing EC2 and
runner permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "ecr-public:GetAuthorizationToken",
    "sts:GetServiceBearerToken"
  ],
  "Resource": "*"
}
```

The role policy is managed outside this repository. Missing permissions fail the
login step; there is no anonymous fallback. ECR Public authentication always uses
`us-east-1`, independently of the EC2 region. See the AWS documentation for
[registry authentication](https://docs.aws.amazon.com/AmazonECR/latest/public/public-registry-auth.html)
and [token permissions](https://docs.aws.amazon.com/cli/latest/reference/ecr-public/get-login-password.html).

AWS credentials authenticate ECR pulls only. Published Samudra images on GHCR
continue to use the existing GitHub-token login; the NVIDIA PhysicsNeMo base
image is pulled from NGC. Jobs that do not pull ECR images do not need AWS access.

## Authorize AWS tests for a pull request

Fork PRs cannot provision AWS runners through the ordinary PR workflow. GitHub's
**Approve workflows** button lets those workflows start, but does not grant the
fork access to AWS provisioning or repository secrets.

After reviewing the code, a maintainer with repository write access can use
**Actions → PR AWS tests → Run workflow**. Keep the workflow branch on `main`,
enter `pr_number`, and optionally enter the full reviewed `expected_head_sha`.
The optional SHA makes dispatch fail if the contributor has pushed since review.
This launches two disposable GPU instances for the ordinary CUDA suite and the
PhysicsNeMo container build/smoke/CPU/GPU suite. Container images stay local;
ARM testing and image publication remain in the existing main-branch workflow.

Agents and maintainers can use the same entry point through the CLI:

```bash
gh workflow run pr-aws-tests.yml --repo m2lines/Samudra --ref main \
  -f pr_number=123 -f expected_head_sha=FULL_REVIEWED_HEAD_SHA
```

Or the GitHub REST API (the `expected_head_sha` input may be omitted):

```http
POST /repos/m2lines/Samudra/actions/workflows/pr-aws-tests.yml/dispatches
Content-Type: application/json

{"ref":"main","inputs":{"pr_number":"123","expected_head_sha":"FULL_REVIEWED_HEAD_SHA"}}
```

The controller resolves and validates GitHub's current test merge commit, then
pins both suites to that SHA. A trusted hosted job records the PR number, head
SHA and merge SHA in its job name. The original `report-gpu-test-status` and
`report-container-test-status` jobs read the Actions API to verify that the
matching manual suite passed. They do not trust artifacts or output supplied by
PR code. The most recent matching attempt takes precedence, including failures,
cancellations and incomplete retries.

When the manual workflow completes, **Refresh PR AWS checks** reruns only those
original reporting jobs. They turn green when their corresponding suite passed,
without changing required check names or bypassing branch protection. Provisioning
and test failures remain failures. GitHub's historical manual-run failures remain
in Actions history; the refreshed PR checks show the current result.

Approve the ordinary PR workflows before dispatching. If they are still running
when the refresh happens, rerun the reporting jobs once they finish. PRs created
before this mechanism was merged must first merge/rebase onto a version containing
the new reporting steps; rerunning an old workflow does not replace its YAML.
If the PR or its target branch changes the merge commit, dispatch again for the
new revision. Conflicts or a merge commit GitHub has not yet calculated fail
before provisioning; resolve conflicts or retry after GitHub finishes.

Provisioning and check refresh run separately on GitHub-hosted runners. Test jobs
have a read-only token, do not persist checkout credentials, and do not restore
or save default-branch caches. Dispatch is authorization to execute the reviewed
PR's code on the project's AWS test infrastructure.

Controller regression tests (no cloud instances or project dependencies needed):

```bash
node --test .github/scripts/pr-aws-ci.test.cjs
```
