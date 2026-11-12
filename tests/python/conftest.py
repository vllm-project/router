"""Path setup for the router Python suite.

Plugin registration lives in the repository-root ``conftest.py``. Pytest only
allows ``pytest_plugins`` in that top-level file. Importable utilities live
under ``tests/python/helpers/``. Fixtures live under
``tests/python/helpers/fixtures/``.
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
