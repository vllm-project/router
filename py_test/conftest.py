import sys
from pathlib import Path

import pytest

from py_test.helpers.mark import HARDWARE_MARK_NORMALIZATION_FIXTURE

# Ensure local sources in py_src are importable ahead of any installed package
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "py_src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(name=HARDWARE_MARK_NORMALIZATION_FIXTURE)
def _normalized_hardware_marks(request):
    """Platform name for one mixed-count ``hardware_test`` variant.

    The variant's SKU and ``cards_{n}`` marks are already on the item.
    This fixture only consumes the indirect parameter so the test function
    does not need a new argument.
    """
    return request.param
