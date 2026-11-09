"""Repository pytest entrypoint.

``pytest_plugins`` has to be declared here. Pytest rejects that list in
``tests/python/conftest.py`` when collection starts at the repository
(``pytest .`` or ``pytest tests``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
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
