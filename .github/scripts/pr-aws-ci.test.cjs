// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { prepare, verify, refresh, stamp } = require("./pr-aws-ci.cjs");
const head = "a".repeat(40),
  merge = "b".repeat(40),
  base = "c".repeat(40);

function fixture() {
  const state = {
    pr: {
      state: "open",
      head: { sha: head },
      base: { sha: base, repo: { full_name: "owner/repo" } },
      mergeable: true,
      merge_commit_sha: merge,
    },
    parents: [{ sha: base }, { sha: head }],
    runs: [
      {
        id: 10,
        display_title: "AWS tests for PR #123",
        workflow_id: 42,
        event: "workflow_dispatch",
        head_branch: "main",
        status: "completed",
        html_url: "manual/10",
      },
    ],
    jobs: {
      10: [
        { name: stamp(123, head, merge), conclusion: "success" },
        { name: "gpu-tests", conclusion: "success" },
        { name: "container-tests", conclusion: "success" },
      ],
    },
    originals: [],
    reruns: [],
    outputs: {},
    messages: [],
  };
  const actions = {
    listWorkflowRuns: "runs",
    listJobsForWorkflowRun: "jobs",
    getWorkflow: async () => ({ data: { id: 42 } }),
    getWorkflowRun: async ({ run_id }) => ({
      data: state.runs.find((run) => run.id === run_id),
    }),
    reRunJobForWorkflowRun: async ({ job_id }) => state.reruns.push(job_id),
  };
  const github = {
    rest: {
      actions,
      repos: { get: async () => ({ data: { default_branch: "main" } }) },
      pulls: { get: async () => ({ data: state.pr }) },
      git: { getCommit: async () => ({ data: { parents: state.parents } }) },
    },
    paginate: async (method, args) =>
      method === "jobs"
        ? state.jobs[args.run_id] || []
        : args.workflow_id === "pr-aws-tests.yml"
          ? state.runs
          : state.originals.filter((run) => run.head_sha === args.head_sha),
  };
  const core = {
    info: (msg) => state.messages.push(msg),
    warning: (msg) => state.messages.push(msg),
    setOutput: (key, value) => {
      state.outputs[key] = value;
    },
  };
  core.summary = {
    addHeading() {
      return this;
    },
    addRaw() {
      return this;
    },
    async write() {},
  };
  return {
    state,
    args: {
      github,
      core,
      context: {
        repo: { owner: "owner", repo: "repo" },
        ref: "refs/heads/main",
      },
    },
  };
}

test("pins a reviewed PR to the validated merge commit", async () => {
  const { state, args } = fixture();
  await prepare(args, "123", head);
  assert.deepEqual(state.outputs, { head_sha: head, merge_sha: merge });
});

for (const [name, mutate, number, expected] of [
  ["invalid input", () => {}, "123;echo bad", head],
  [
    "non-default dispatch",
    (_, args) => {
      args.context.ref = "refs/heads/feature";
    },
    "123",
    head,
  ],
  [
    "closed PR",
    (s) => {
      s.pr.state = "closed";
    },
    "123",
    head,
  ],
  [
    "wrong repository",
    (s) => {
      s.pr.base.repo.full_name = "someone/else";
    },
    "123",
    head,
  ],
  ["changed reviewed head", () => {}, "123", base],
  [
    "conflict",
    (s) => {
      s.pr.mergeable = false;
    },
    "123",
    head,
  ],
  [
    "pending mergeability",
    (s) => {
      s.pr.mergeable = null;
    },
    "123",
    head,
  ],
  [
    "stale merge parents",
    (s) => {
      s.parents[0].sha = head;
    },
    "123",
    head,
  ],
])
  test(`refuses provisioning: ${name}`, async () => {
    const { state, args } = fixture();
    mutate(state, args);
    await assert.rejects(prepare(args, number, expected));
    assert.deepEqual(state.outputs, {});
  });

test("accepts exact revision and each suite independently", async () => {
  const { state, args } = fixture();
  state.jobs[10][2].conclusion = "failure";
  await verify(args, "gpu", 123, head);
  await assert.rejects(verify(args, "container", 123, head), /have not passed/);
});

test("rejects results for another PR or head", async () => {
  const { args } = fixture();
  await assert.rejects(
    verify(args, "gpu", 124, head),
    /maintainer authorization/,
  );
  await assert.rejects(verify(args, "gpu", 123, base), /PR changed/);
});

test("base advancement invalidates evidence for the old merge", async () => {
  const { state, args } = fixture();
  state.pr.merge_commit_sha = base;
  await assert.rejects(
    verify(args, "gpu", 123, head),
    /maintainer authorization/,
  );
  state.jobs[10][0].name = stamp(123, head, base);
  await verify(args, "gpu", 123, head);
});

for (const conclusion of ["failure", "cancelled", "skipped", null]) {
  test(`a newer ${conclusion} attempt supersedes a passing run`, async () => {
    const { state, args } = fixture();
    state.runs.push({ ...state.runs[0], id: 11 });
    state.jobs[11] = [
      { name: stamp(123, head, merge), conclusion: "success" },
      { name: "gpu-tests", conclusion },
    ];
    await assert.rejects(verify(args, "gpu", 123, head), /have not passed/);
  });
}

test("waits for the manual workflow to complete", async () => {
  const { state, args } = fixture();
  state.runs[0].status = "in_progress";
  await assert.rejects(verify(args, "gpu", 123, head), /have not passed/);
});

test("missing or failed trusted record cannot authorize results", async () => {
  const { state, args } = fixture();
  state.jobs[10][0].conclusion = "failure";
  await assert.rejects(
    verify(args, "gpu", 123, head),
    /maintainer authorization/,
  );
});

test("refreshes original report jobs only, using the latest matching runs", async () => {
  const { state, args } = fixture();
  state.originals = [
    {
      id: 3,
      head_sha: head,
      status: "completed",
      pull_requests: [{ number: 123 }],
    },
    {
      id: 2,
      head_sha: head,
      status: "completed",
      pull_requests: [{ number: 123 }],
    },
  ];
  state.jobs[3] = [
    { name: "report-gpu-test-status", id: 30, conclusion: "failure" },
    { name: "report-container-test-status", id: 31, conclusion: "failure" },
    { name: "ec2", id: 32 },
  ];
  await refresh(args, 10);
  assert.deepEqual(state.reruns, [30, 31]);
});

for (const field of ["head", "merge", "closed"])
  test(`does not refresh after PR ${field} changes`, async () => {
    const { state, args } = fixture();
    if (field === "head") state.pr.head.sha = base;
    else if (field === "merge") state.pr.merge_commit_sha = base;
    else state.pr.state = "closed";
    await refresh(args, 10);
    assert.deepEqual(state.reruns, []);
  });

for (const [field, value] of [
  ["event", "pull_request"],
  ["head_branch", "untrusted"],
  ["workflow_id", 99],
  ["status", "in_progress"],
]) {
  test(`rejects untrusted completion: ${field}`, async () => {
    const { state, args } = fixture();
    state.runs[0][field] = value;
    await assert.rejects(refresh(args, 10), /Not a completed default-branch/);
    assert.deepEqual(state.reruns, []);
  });
}

test("a later rerun of an older run supersedes newer run IDs", async () => {
  const { state, args } = fixture();
  state.runs[0].run_started_at = "2026-10-06T10:00:00Z";
  state.runs.push({
    ...state.runs[0],
    id: 9,
    run_started_at: "2026-10-06T11:00:00Z",
  });
  state.jobs[9] = [
    { name: stamp(123, head, merge), conclusion: "success" },
    { name: "gpu-tests", conclusion: "failure" },
  ];
  await assert.rejects(verify(args, "gpu", 123, head), /have not passed/);
});
