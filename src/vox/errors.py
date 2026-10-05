"""Errors that carry a one-line message, a fix hint and an exit code.

Exit codes: 0 success, 1 user error, 2 missing dependency or model.
"""

from __future__ import annotations


class VoxError(Exception):
    exit_code = 1

    def __init__(self, message: str, hint: str | None = None, *, exit_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        if exit_code is not None:
            self.exit_code = exit_code


class UserError(VoxError):
    """Bad input: wrong arguments, unreadable file, invalid config value."""

    exit_code = 1


class MissingError(VoxError):
    """A model or external dependency (such as ffmpeg) is not installed."""

    exit_code = 2


class ServerError(VoxError):
    """The background server could not be started or reached."""

    exit_code = 1
