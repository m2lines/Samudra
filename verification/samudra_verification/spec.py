# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""What identifies a campaign, written once to `replay.json` and read by every consumer."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .records import from_json, to_json

REPLAY = "replay.json"


class Lane(StrEnum):
    """Which dependency set a campaign ran against."""

    PINNED = "pinned"  # A lock or pinned environment file (uv.lock, data/.test.yml).
    CANARY = "canary"  # Floats within the supported ranges (data/.test-canary.yml).


@dataclass(frozen=True)
class CampaignSpec:
    """Everything needed to tell campaigns apart and to replay one.

    `project` is the project whose environment ran the campaign. Each test also
    carries its own project tag, since one campaign can run several projects' tests.
    """

    project: str
    lane: Lane
    string_storage: str | None  # pandas string backend, for data campaigns.
    profile: str
    examples: int
    seed: int
    timeout_seconds: float
    command: list[str]
    cwd: str
    revision: str
    python: str
    platform: str
    dependency_file: str
    dependency_sha256: str

    @property
    def label(self) -> str:
        """A name unique to this campaign's configuration, e.g. `data-pinned-pyarrow`."""
        parts = [self.project, self.lane.value]
        if self.string_storage:
            parts.append(self.string_storage)
        return "-".join(parts)

    def write(self, directory: Path) -> None:
        (directory / REPLAY).write_text(to_json(self))

    @classmethod
    def read(cls, directory: Path) -> "CampaignSpec":
        return from_json(cls, (directory / REPLAY).read_text())
