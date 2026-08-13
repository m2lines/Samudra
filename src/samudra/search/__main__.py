# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one configured architecture search."""

from samudra.search import SearchConfig, build_search


def main() -> None:
    config = SearchConfig.from_yaml_and_cli()
    search = build_search(config)
    competing = sum(not candidate.fixed for candidate in config.candidates)
    anchors = len(config.candidates) - competing
    print(
        f"search={config.name} algorithm={config.algorithm.type} "
        f"executor={config.executor.type} competing={competing} anchors={anchors}",
        flush=True,
    )
    print(search.start(), flush=True)


if __name__ == "__main__":
    main()
