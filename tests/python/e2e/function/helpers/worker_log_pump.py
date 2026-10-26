"""Read worker subprocess stdout and write to a log file (+ optional terminal echo)."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path


class SubprocessLogPump:
    """``Popen(stdout=PIPE)`` lines → log file; optionally echo to ``sys.stderr``."""

    def __init__(
        self,
        proc: subprocess.Popen[str],
        log_path: Path | str,
        *,
        echo: bool = True,
        prefix: str = "",
    ) -> None:
        if proc.stdout is None:
            raise ValueError("proc.stdout must be PIPE for SubprocessLogPump")
        self._proc = proc
        self._log_path = Path(log_path)
        self._echo = echo
        self._prefix = prefix
        self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(
            target=self._pump, name=f"log-pump-{self._log_path.name}", daemon=True
        )
        self._thread.start()

    def _pump(self) -> None:
        stream = self._proc.stdout
        assert stream is not None
        for line in iter(stream.readline, ""):
            self._log_file.write(line)
            self._log_file.flush()
            if self._echo:
                sys.stderr.write(f"{self._prefix}{line}")
                sys.stderr.flush()
        stream.close()

    def close(self, *, join_timeout: float = 30.0) -> None:
        self._thread.join(timeout=join_timeout)
        try:
            self._log_file.close()
        except OSError:
            pass
