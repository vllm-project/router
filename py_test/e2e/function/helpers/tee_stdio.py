"""Duplicate worker subprocess output to a log file and the pytest terminal."""

from __future__ import annotations

import sys
from typing import TextIO


class TeeTextIO:
    """Line-oriented tee: same bytes go to ``log_file`` and ``stream`` (e.g. stderr)."""

    def __init__(self, log_file: TextIO, stream: TextIO | None = None) -> None:
        self._log_file = log_file
        self._stream = stream if stream is not None else sys.stderr

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._log_file.write(data)
        self._stream.write(data)
        return len(data)

    def flush(self) -> None:
        self._log_file.flush()
        self._stream.flush()

    def fileno(self) -> int:
        # subprocess / stdio checks call fileno on the stdout= sink (Windows).
        return self._log_file.fileno()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str | None:
        return getattr(self._log_file, "encoding", None)

    def close(self) -> None:
        self.flush()
        self._log_file.close()
