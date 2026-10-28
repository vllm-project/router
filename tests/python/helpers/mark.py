"""Pytest marks for selecting CI hardware (NVIDIA L4, AMD MI300X).

Importing this module needs only the standard library. ``pytest`` and
``torch`` are imported inside the decorators and the platform probes so
marker lookup can run without either package installed.

Register levels, SKUs, platforms, and card counts in ``pytest.ini`` with these tags:

- ``[ci-level]``: a CI level (``ready``)
- ``[hardware-resource]`` plus ``[cuda]`` or ``[rocm]``: a SKU (``L4``, ``MI300``)
- ``[hardware-platform]``: ``cuda``, ``rocm``, or the ``gpu`` device class
- ``[hardware-cards]``: ``cards_{n}``
- ``[gpu]`` on a platform line: also attach ``pytest.mark.gpu``

``cpu`` / ``gpu`` are not ``res`` keys. Pass an accelerator and a SKU:

    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    @hardware_test(res={"rocm": "MI300"}, num_cards=2)
    @hardware_test(
        res={"cuda": "L4", "rocm": "MI300"},
        num_cards={"cuda": 4, "rocm": 2},
    )
"""

from __future__ import annotations

import configparser
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    # CUDA may list several SKUs. Kept under TYPE_CHECKING so Python 3.9 does
    # not evaluate ``str | list`` while importing this module.
    SkuSpec = str | list[str] | tuple[str, ...]

_CI_LEVEL_MARKER_TAG = "[ci-level]"
_HARDWARE_RESOURCE_MARKER_TAG = "[hardware-resource]"
_HARDWARE_CARDS_MARKER_TAG = "[hardware-cards]"
_HARDWARE_PLATFORM_MARKER_TAG = "[hardware-platform]"
_GPU_DEVICE_CLASS_TAG = "[gpu]"
_CARDS_MARK_NAME_RE = re.compile(r"^cards_(\d+)$")
# Indirect fixture in ``tests/python/conftest.py``. Mixed-count ``@hardware_test``
# parametrizes it so each collected item carries one platform's SKU and
# ``cards_{n}``.
HARDWARE_MARK_NORMALIZATION_FIXTURE = "_normalized_hardware_marks"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _marker_entries() -> tuple[str, ...]:
    """Return ``name: description`` lines from ``pytest.ini`` ``markers``."""
    parser = configparser.ConfigParser(interpolation=None)
    path = _repo_root() / "pytest.ini"
    try:
        parser.read(path, encoding="utf-8")
        raw = parser.get("pytest", "markers", fallback="")
    except (configparser.Error, OSError):
        return ()
    return tuple(line.strip() for line in raw.splitlines() if line.strip())


@lru_cache(maxsize=16)
def _marker_names_with_tag(tag: str) -> frozenset[str]:
    """Return pytest marker names whose description contains ``tag``."""
    names: set[str] = set()
    for entry in _marker_entries():
        if tag not in entry:
            continue
        name = entry.split(":", 1)[0].strip()
        if name:
            names.add(name)
    return frozenset(names)


@lru_cache(maxsize=1)
def get_level_markers() -> frozenset[str]:
    """Return CI level marker names tagged ``[ci-level]`` (``ready``, …)."""
    return _marker_names_with_tag(_CI_LEVEL_MARKER_TAG)


@lru_cache(maxsize=1)
def get_hardware_mark_list() -> frozenset[str]:
    """Return hardware SKU marker names tagged ``[hardware-resource]``."""
    return _marker_names_with_tag(_HARDWARE_RESOURCE_MARKER_TAG)


@lru_cache(maxsize=1)
def get_supported_card_counts() -> frozenset[int]:
    """Return ``n`` for each ``cards_{n}`` marker tagged ``[hardware-cards]``."""
    counts: set[int] = set()
    for name in _marker_names_with_tag(_HARDWARE_CARDS_MARKER_TAG):
        matched = _CARDS_MARK_NAME_RE.fullmatch(name)
        if matched:
            counts.add(int(matched.group(1)))
    return frozenset(counts)


@lru_cache(maxsize=1)
def get_supported_platforms() -> frozenset[str]:
    """Return platform marker names tagged ``[hardware-platform]``.

    Includes the ``gpu`` device-class mark and accelerator keys.
    ``hardware_marks`` / ``hardware_test`` ``res`` dicts reject ``gpu``.
    """
    return _marker_names_with_tag(_HARDWARE_PLATFORM_MARKER_TAG)


@lru_cache(maxsize=8)
def get_skus_for_platform(platform: str) -> frozenset[str]:
    """SKU names tagged ``[hardware-resource]`` and ``[platform]`` (e.g. ``[cuda]``)."""
    return get_hardware_mark_list() & _marker_names_with_tag(f"[{platform}]")


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def is_cuda() -> bool:
    """Return whether this process has NVIDIA CUDA, not ROCm's CUDA shim."""
    torch = _import_torch()
    if torch is None:
        return False
    hip = getattr(getattr(torch, "version", None), "hip", None)
    try:
        return bool(torch.cuda.is_available() and not hip)
    except Exception:
        return False


def is_rocm() -> bool:
    """Return whether PyTorch reports a ROCm (HIP) runtime."""
    torch = _import_torch()
    if torch is None:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def device_count() -> int:
    """Return visible accelerator count, or 0 when PyTorch cannot see one."""
    torch = _import_torch()
    if torch is None:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _require_sku(platform: str, res: str) -> None:
    allowed = get_skus_for_platform(platform)
    if res not in allowed:
        supported = ", ".join(sorted(allowed)) or (
            f"(none tagged [hardware-resource] [{platform}])"
        )
        raise ValueError(
            f"Invalid {platform} resource type: {res}. Supported: {supported}"
        )


def _as_sku_list(res: SkuSpec) -> list[str]:
    skus = [res] if isinstance(res, str) else list(res)
    if not skus:
        raise ValueError("resource list must not be empty")
    return skus


def _cards_mark(num_cards: int) -> pytest.MarkDecorator:
    """Return ``cards_{n}`` so collection can filter with ``-m cards_2``."""
    import pytest

    if not isinstance(num_cards, int) or isinstance(num_cards, bool) or num_cards < 1:
        raise ValueError(f"num_cards must be a positive int, got {num_cards!r}")
    if num_cards not in get_supported_card_counts():
        supported = ", ".join(f"cards_{n}" for n in sorted(get_supported_card_counts()))
        supported = supported or "(none tagged [hardware-cards])"
        raise ValueError(
            f"num_cards={num_cards} has no registered marker. Supported: {supported}."
        )
    return getattr(pytest.mark, f"cards_{num_cards}")


def _with_card_count_skip(
    marks: list[pytest.MarkDecorator],
    *,
    num_cards: int,
    on_platform: bool,
    reason: str,
) -> list[pytest.MarkDecorator]:
    import pytest

    if num_cards == 1 or not on_platform:
        return marks
    return marks + [
        pytest.mark.skipif(device_count() < num_cards, reason=reason),
    ]


def _cuda_marks(*, res: SkuSpec, num_cards: int):
    import pytest

    skus = _as_sku_list(res)
    for sku in skus:
        _require_sku("cuda", sku)
    marks = [getattr(pytest.mark, sku) for sku in skus]
    marks.extend([pytest.mark.cuda, _cards_mark(num_cards)])
    return _with_card_count_skip(
        marks,
        num_cards=num_cards,
        on_platform=is_cuda(),
        reason=f"Need at least {num_cards} CUDA GPUs to run the test.",
    )


def _rocm_marks(*, res: str, num_cards: int):
    import pytest

    _require_sku("rocm", res)
    marks = [getattr(pytest.mark, res), pytest.mark.rocm, _cards_mark(num_cards)]
    return _with_card_count_skip(
        marks,
        num_cards=num_cards,
        on_platform=is_rocm(),
        reason=f"Need at least {num_cards} ROCm GPUs to run the test.",
    )


@lru_cache(maxsize=1)
def _gpu_res_platforms() -> frozenset[str]:
    """Accelerator platforms tagged ``[hardware-platform] [gpu]``."""
    return (
        get_supported_platforms() & _marker_names_with_tag(_GPU_DEVICE_CLASS_TAG)
    ) - {"gpu"}


def _res_platforms() -> frozenset[str]:
    """Accelerator keys allowed in ``res``.

    A ``[hardware-platform]`` name is a ``res`` key only when some SKU is tagged
    with that platform. ``gpu`` has no SKUs.
    """
    return frozenset(
        platform
        for platform in get_supported_platforms()
        if get_skus_for_platform(platform)
    )


def _normalize_num_cards(
    res: dict[str, SkuSpec], num_cards: int | dict[str, int]
) -> dict[str, int]:
    allowed = _res_platforms()
    device_class = get_supported_platforms() - allowed
    for platform in res:
        if platform in device_class:
            raise ValueError(
                f"{platform!r} is not a res dict key; use pytest.mark.{platform} "
                f"or an accelerator key ({', '.join(sorted(allowed))})."
            )
        if platform not in allowed:
            supported = ", ".join(sorted(allowed)) or "(none)"
            raise ValueError(
                f"Unsupported platform: {platform}. res keys: {supported}."
            )
    if isinstance(num_cards, int) and not isinstance(num_cards, bool):
        return {platform: num_cards for platform in res}
    if not isinstance(num_cards, dict):
        raise ValueError(f"num_cards must be a positive int or dict, got {num_cards!r}")
    num_cards_dict = dict(num_cards)
    for platform in num_cards_dict:
        if platform not in res:
            raise ValueError(f"Platform '{platform}' in num_cards but not in res.")
    for platform in res:
        num_cards_dict.setdefault(platform, 1)
    return num_cards_dict


def _marks_for_platform(
    platform: str, resource: SkuSpec, num_cards: int
) -> list[pytest.MarkDecorator]:
    import pytest

    if platform == "cuda":
        marks = _cuda_marks(res=resource, num_cards=num_cards)
    elif platform == "rocm":
        if not isinstance(resource, str):
            raise ValueError(f"rocm resource must be a string, got {resource!r}")
        marks = _rocm_marks(res=resource, num_cards=num_cards)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    if platform in _gpu_res_platforms():
        return [pytest.mark.gpu] + marks
    return marks


def _skipif_not_platform(platform: str) -> pytest.MarkDecorator:
    import pytest

    checkers = {
        "cuda": is_cuda,
        "rocm": is_rocm,
    }
    return pytest.mark.skipif(
        not checkers[platform](), reason=f"Requires {platform} platform"
    )


def _apply_marks(func, marks: list[pytest.MarkDecorator]):
    for mark in reversed(marks):
        func = mark(func)
    return func


def hardware_marks(*, res: dict[str, SkuSpec], num_cards: int | dict[str, int] = 1):
    """Return marks for a single pytest item.

    Different per-platform ``num_cards`` cannot share one item: ``cards_2`` and
    ``cards_4`` would both match ``-m cards_2`` and ``-m cards_4``. Use
    ``@hardware_test`` (one collected variant per platform) or call this once
    per platform and attach each list to its own ``pytest.param``.
    """
    num_cards_dict = _normalize_num_cards(res, num_cards)
    if len(res) > 1 and len(set(num_cards_dict.values())) > 1:
        raise ValueError(
            "hardware_marks() cannot attach different per-platform num_cards to one "
            f"pytest item (got {num_cards_dict}). Use @hardware_test, which collects "
            "one variant per platform, or call hardware_marks() once per platform."
        )

    all_marks: list[pytest.MarkDecorator] = []
    for platform, resource in res.items():
        all_marks.extend(
            _marks_for_platform(platform, resource, num_cards_dict[platform])
        )
    return all_marks


def hardware_test(*, res: dict[str, SkuSpec], num_cards: int | dict[str, int] = 1):
    """Apply hardware marks, splitting items when card counts differ by platform.

    A single item cannot carry both ``cards_4`` (CUDA) and ``cards_2`` (ROCm):
    ``-m "L4 and cards_4 and cuda"`` would otherwise also match
    ``-m "L4 and cards_2 and cuda"``. When ``num_cards`` differs across
    platforms, this decorator parametrizes one variant per platform so each
    item only has that platform's SKU and ``cards_{n}``.
    """
    import pytest

    num_cards_dict = _normalize_num_cards(res, num_cards)
    mixed_counts = len(res) > 1 and len(set(num_cards_dict.values())) > 1
    if not mixed_counts:
        all_marks = hardware_marks(res=res, num_cards=num_cards)

        def apply_union_marks(func):
            return _apply_marks(func, all_marks)

        return apply_union_marks

    params = []
    for platform, resource in res.items():
        marks = _marks_for_platform(platform, resource, num_cards_dict[platform])
        marks.append(_skipif_not_platform(platform))
        params.append(pytest.param(platform, marks=marks, id=platform))

    def apply_split_params(func):
        func = pytest.mark.usefixtures(HARDWARE_MARK_NORMALIZATION_FIXTURE)(func)
        return pytest.mark.parametrize(
            HARDWARE_MARK_NORMALIZATION_FIXTURE, params, indirect=True
        )(func)

    return apply_split_params
