// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0

// Only run this controller from trusted default-branch code on hosted runners.
const WORKFLOW = "pr-aws-tests.yml";
const REPORTS = {
  gpu: {
    workflow: "test-gpu.yml",
    report: "report-gpu-test-status",
    test: "gpu-tests",
  },
  container: {
    workflow: "container-physicsnemo.yml",
    report: "report-container-test-status",
    test: "container-tests",
  },
};
const newestFirst = (a, b) =>
  (Date.parse(b.run_started_at) || 0) - (Date.parse(a.run_started_at) || 0) ||
  b.id - a.id;
const stamp = (number, head, merge) =>
  `PR #${number} head ${head} merge ${merge}`;
const parseStamp = (name) =>
  /^PR #(\d+) head ([a-f0-9]{40}) merge ([a-f0-9]{40})$/.exec(name);

async function jobs(github, repo, run_id) {
  return github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...repo,
    run_id,
    filter: "latest",
    per_page: 100,
  });
}

async function prepare({ github, context, core }, number, expectedHead) {
  if (!/^[1-9]\d*$/.test(number))
    throw new Error("pr_number must be a positive integer.");
  const repo = context.repo;
  const { data: repository } = await github.rest.repos.get(repo);
  if (context.ref !== `refs/heads/${repository.default_branch}`) {
    throw new Error("Dispatch this workflow from the default branch only.");
  }
  const { data: pr } = await github.rest.pulls.get({
    ...repo,
    pull_number: Number(number),
  });
  if (
    pr.state !== "open" ||
    pr.base.repo.full_name !== `${repo.owner}/${repo.repo}`
  ) {
    throw new Error("PR must be open and target this repository.");
  }
  if (expectedHead && expectedHead !== pr.head.sha) {
    throw new Error(
      "PR changed since review; expected_head_sha does not match.",
    );
  }
  if (pr.mergeable !== true || !pr.merge_commit_sha) {
    throw new Error(
      "PR merge commit is unavailable or conflicts. Retry after GitHub computes mergeability.",
    );
  }
  const { data: commit } = await github.rest.git.getCommit({
    ...repo,
    commit_sha: pr.merge_commit_sha,
  });
  if (
    commit.parents.length !== 2 ||
    commit.parents[0].sha !== pr.base.sha ||
    commit.parents[1].sha !== pr.head.sha
  ) {
    throw new Error(
      "PR merge commit is stale; retry after GitHub refreshes it.",
    );
  }
  core.setOutput("head_sha", pr.head.sha);
  core.setOutput("merge_sha", pr.merge_commit_sha);
  await core.summary
    .addHeading(`AWS tests for PR #${number}`)
    .addRaw(
      `Head: ${pr.head.sha}\n\nBase: ${pr.base.sha}\n\nTesting merge: ${pr.merge_commit_sha}\n`,
    )
    .write();
}

// A trusted job name records the resolved revisions. No artifacts, logs or
// outputs produced by the PR's test process are accepted as proof of success.
async function verify({ github, context, core }, suite, number, head) {
  const spec = REPORTS[suite];
  if (!spec) throw new Error("Unknown test suite.");
  const repo = context.repo;
  // A reporting-job rerun retains its original event's merge SHA. The base
  // branch may since have advanced: require evidence for the *current* merge,
  // while refusing to transfer results to a different PR head.
  const { data: pr } = await github.rest.pulls.get({
    ...repo,
    pull_number: Number(number),
  });
  if (
    pr.state !== "open" ||
    pr.head.sha !== head ||
    !pr.merge_commit_sha ||
    pr.mergeable !== true
  ) {
    throw new Error(
      "PR changed, closed, or has no mergeable revision; refresh its PR workflows before retrying.",
    );
  }
  const merge = pr.merge_commit_sha;
  const { data: repository } = await github.rest.repos.get(repo);
  const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
    ...repo,
    workflow_id: WORKFLOW,
    event: "workflow_dispatch",
    branch: repository.default_branch,
    per_page: 100,
  });
  // Newest matching run wins, including a failed/cancelled or unfinished retry.
  for (const run of runs.sort(newestFirst)) {
    if (run.display_title !== `AWS tests for PR #${number}`) continue;
    const runJobs = await jobs(github, repo, run.id);
    if (
      !runJobs.some(
        (job) =>
          job.name === stamp(number, head, merge) &&
          job.conclusion === "success",
      )
    )
      continue;
    const test = runJobs.find((job) => job.name === spec.test);
    if (run.status !== "completed" || test?.conclusion !== "success") {
      throw new Error(`Manual ${suite} tests have not passed: ${run.html_url}`);
    }
    core.info(`Verified ${suite} tests for merge ${merge}: ${run.html_url}`);
    await core.summary
      .addRaw(
        `Passed via [manual ${suite} tests](${run.html_url}) for merge \`${merge}\`.\n`,
      )
      .write();
    return;
  }
  throw new Error(
    `AWS-backed ${suite} tests need maintainer authorization. In Actions, select "PR AWS tests" → "Run workflow" on ${repository.default_branch}, enter PR ${number}. Or: gh workflow run ${WORKFLOW} --repo ${repo.owner}/${repo.repo} --ref ${repository.default_branch} -f pr_number=${number} -f expected_head_sha=${head}. See .github/CI-README.md. Approving or rerunning the fork workflow alone does not launch AWS tests.`,
  );
}

async function refresh({ github, context, core }, runId) {
  const repo = context.repo;
  const { data: repository } = await github.rest.repos.get(repo);
  const { data: workflow } = await github.rest.actions.getWorkflow({
    ...repo,
    workflow_id: WORKFLOW,
  });
  const { data: run } = await github.rest.actions.getWorkflowRun({
    ...repo,
    run_id: runId,
  });
  if (
    run.workflow_id !== workflow.id ||
    run.event !== "workflow_dispatch" ||
    run.head_branch !== repository.default_branch ||
    run.status !== "completed"
  ) {
    throw new Error("Not a completed default-branch PR AWS tests run.");
  }
  const runJobs = await jobs(github, repo, run.id);
  const record = runJobs.find(
    (job) => parseStamp(job.name) && job.conclusion === "success",
  );
  if (!record) {
    core.info("No validated PR revision; no checks to refresh.");
    return;
  }
  const [, number, head, merge] = parseStamp(record.name);
  const { data: pr } = await github.rest.pulls.get({
    ...repo,
    pull_number: Number(number),
  });
  if (
    pr.state !== "open" ||
    pr.head.sha !== head ||
    pr.merge_commit_sha !== merge
  ) {
    core.info("PR changed or closed; leaving its current checks untouched.");
    return;
  }
  for (const spec of Object.values(REPORTS)) {
    const runs = (
      await Promise.all(
        [head, merge].map((head_sha) =>
          github.paginate(github.rest.actions.listWorkflowRuns, {
            ...repo,
            workflow_id: spec.workflow,
            event: "pull_request",
            head_sha,
            per_page: 100,
          }),
        ),
      )
    ).flat();
    // Rerun only the newest relevant PR workflow, never old revisions or GPU jobs.
    const original = runs
      .sort(newestFirst)
      .find(
        (candidate) =>
          [head, merge].includes(candidate.head_sha) &&
          (candidate.pull_requests.length === 0 ||
            candidate.pull_requests.some(
              (pull) => pull.number === Number(number),
            )),
      );
    if (!original || original.status !== "completed") {
      core.warning(
        `No completed ${spec.workflow} run to refresh. Approve the PR workflows first, or rerun its reporting job after they finish.`,
      );
      continue;
    }
    const report = (await jobs(github, repo, original.id)).find(
      (job) => job.name === spec.report,
    );
    if (!report || report.conclusion === "skipped") continue;
    await github.rest.actions.reRunJobForWorkflowRun({
      ...repo,
      job_id: report.id,
    });
    core.info(`Refreshing ${spec.report}: ${original.html_url}`);
  }
}

module.exports = { prepare, verify, refresh, stamp, parseStamp };
