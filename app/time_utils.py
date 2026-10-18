"""Small helpers for displaying timestamps in the machine's local timezone."""

from __future__ import annotations

from datetime import datetime, timezone


def local_timestamp(value: str | None = None) -> str:
    """Format a UTC/ISO timestamp in the host's local timezone.

    BuildGuard stores timestamps in UTC so ordering remains reliable. User
    interfaces and notifications should display them in the timezone of the
    machine running BuildGuard instead of exposing the storage representation.
    """
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.now(timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
