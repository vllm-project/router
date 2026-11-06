"""Root pytest entrypoint for the router Python suite.

``conftest.py`` stays thin: path setup and plugin registration only.
Importable utilities live under ``tests/python/helpers/``.
Pytest fixtures live under ``tests/python/helpers/fixtures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "py_src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest_plugins = (
    "tests.python.helpers.fixtures.hardware",
    "tests.python.helpers.fixtures.router",
    "tests.python.helpers.fixtures.worker",
    "tests.python.helpers.fixtures.bench",
)
