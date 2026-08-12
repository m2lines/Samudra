# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import sys

from samudra import cli, search


def test_cli_dispatches_search_subcommands(monkeypatch):
    called = []
    monkeypatch.setattr(sys, "argv", ["samudra", "search", "plan", "search.yaml"])
    monkeypatch.setattr(search, "main", lambda: called.append(list(sys.argv)))

    cli.main()

    assert called == [["samudra search", "plan", "search.yaml"]]
