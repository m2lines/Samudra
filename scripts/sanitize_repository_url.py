#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Remove credentials from repository URLs before recording provenance."""

# The Torch host may provide Python 3.6 even though Samudra itself requires a
# newer interpreter inside its container.

import re
import sys
from urllib.parse import urlsplit, urlunsplit

SCP_STYLE_URL = re.compile(
    r"^(?P<user>[^/@:\s]+)@(?P<host>(?:\[[^\]]+\])|(?:[^/:\s]+)):(?P<path>.+)$"
)


def sanitize_repository_url(value: str) -> str:
    """Return a repository identity without URL credentials or query secrets."""
    if (
        not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("repository URL must be a non-empty single-line value")

    scp_match = SCP_STYLE_URL.match(value)
    if scp_match:
        host = scp_match.group("host")
        path = scp_match.group("path").lstrip("/")
        return f"ssh://{host}/{path}"

    parsed = urlsplit(value)
    if not parsed.netloc:
        if parsed.scheme and "@" in parsed.path:
            raise ValueError("repository URL has malformed user information")
        if parsed.scheme:
            return urlunsplit((parsed.scheme, "", parsed.path, "", ""))
        return value

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("repository URL has no hostname")
    host = f"[{hostname}]" if ":" in hostname else hostname
    port = parsed.port
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def main() -> int:
    try:
        sanitized = sanitize_repository_url(sys.stdin.read())
    except ValueError:
        print("ERROR: could not sanitize repository URL", file=sys.stderr)
        return 2
    sys.stdout.write(sanitized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
