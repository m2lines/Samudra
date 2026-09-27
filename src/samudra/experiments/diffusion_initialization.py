# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate inherited evolution weights from optional initializer transfer."""


def load_pretraining_source(core, state, *, initialization):
    if initialization == "om4":
        core.load_core(state)
    elif initialization == "scratch":
        # Leave every encoder/adapter parameter at its fresh constructor value.
        # Strict loading still enforces the entire inherited dynamics contract.
        core.evolution.load_state_dict(
            {
                k.removeprefix("evolution."): v
                for k, v in state.items()
                if k.startswith("evolution.")
            },
            strict=True,
        )
    else:
        raise ValueError(f"Unknown initializer initialization: {initialization}")
