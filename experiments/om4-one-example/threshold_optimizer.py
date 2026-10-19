"""Single-threshold L-BFGS experiment; archive the actual runtime source."""

import hashlib
import importlib
import inspect
import textwrap

import torch


def make_optimizer(threshold, output):
    assert threshold in (1e-10, 1e-20)
    module = importlib.import_module("torch.optim.lbfgs")
    original = textwrap.dedent(inspect.getsource(module.LBFGS.step))
    old = "if ys > 1e-10:"
    assert original.count(old) == 1, "Pinned optimizer implementation changed"
    instrumented = original.replace(old,
        'state["curvature_candidates"] = state.get("curvature_candidates", 0) + 1\n'
        '            state["last_curvature_dot"] = float(ys)\n'
        '            if ys > self.curvature_threshold:\n'
        '                state["curvature_accepted"] = state.get("curvature_accepted", 0) + 1')
    output.mkdir(parents=True, exist_ok=True)
    for name, source in (("original-lbfgs-step.txt", original),
                         ("instrumented-lbfgs-step.txt", instrumented)):
        (output / name).write_text(source)
    (output / "optimizer-source.sha256").write_text(
        "\n".join(f"{hashlib.sha256(s.encode()).hexdigest()}  {n}" for n, s in
                  (("original-lbfgs-step.txt", original), ("instrumented-lbfgs-step.txt", instrumented))) + "\n")
    namespace = dict(vars(module))
    exec(compile(instrumented, str(output / "instrumented-lbfgs-step.txt"), "exec"), namespace)

    class ThresholdLBFGS(torch.optim.LBFGS):
        curvature_threshold = threshold
        step = namespace["step"]

    return ThresholdLBFGS
