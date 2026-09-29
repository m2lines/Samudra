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

Validate changes with `actionlint` and pre-commit. A same-repository PR exercises
the hosted launch/login path for GPU tests and the x86 container build without
publishing container images. Main runs additionally exercise benchmarks and the
ARM/container CPU/GPU jobs. Fork PRs retain the callers' existing AWS guards.
