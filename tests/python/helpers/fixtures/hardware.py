"""Hardware-mark normalization fixture used by ``@hardware_test``."""

from __future__ import annotations

import pytest

from tests.python.helpers.mark import HARDWARE_MARK_NORMALIZATION_FIXTURE


@pytest.fixture(name=HARDWARE_MARK_NORMALIZATION_FIXTURE)
def _normalized_hardware_marks(request):
    """Platform name for one mixed-count ``hardware_test`` variant.

    The variant's SKU and ``cards_{n}`` marks are already on the item.
    This fixture only consumes the indirect parameter so the test function
    does not need a new argument.
    """
    return request.param
