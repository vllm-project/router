#!/usr/bin/env python3
"""Fail if a pytest or Rust test file is missing the marks CI uses to select it.

Python (``py_test/**/test_*.py``, not ``py_test/e2e``): every pytest module
needs a ``[ci-level]`` mark (``ready``) and a machine mark. A module counts
when it imports pytest or defines ``test_*`` / ``Test*`` items. Comments and
strings do not count as marks. ``pytest.mark.cpu`` is written directly. SKUs
and other platforms must come from ``hardware_test`` or ``hardware_marks`` so
``cards_{n}`` is attached.

Rust (``tests/*.rs`` integration crates, and ``src/**/*.rs`` that contain a
test module): the file must name one Cargo feature tagged ``[ci-level]`` and
one tagged ``[hardware-platform]`` (currently ``ready`` and ``cpu``).
"""

from __future__ import annotations

import importlib.util
import io
import re
import sys
import tokenize
from functools import lru_cache
from pathlib import Path
from types import ModuleType

HARDWARE_HELPERS = ("hardware_test", "hardware_marks")
_ALLOWED_DIRECT_MARK_FILES = frozenset({"py_test/helpers/mark.py"})
_DIRECT_PLATFORM_ALLOWLIST = frozenset({"cpu"})

HELPER_RE = re.compile(r"(?:" + "|".join(HARDWARE_HELPERS) + r")\s*\(")
PY_TEST_FILE_RE = re.compile(r"^py_test/(?:.*/)?(?:test_[^/]*\.py$|[^/]*_test\.py$)")
# Pytest collects module-level test_* functions and Test* classes without
# importing pytest. Indented test_* methods on other classes are not items.
_TEST_ITEM_RE = re.compile(
    r"(?m)^(?:async[ \t]+def[ \t]+test_\w+|def[ \t]+test_\w+|class[ \t]+Test\w+\b)"
)
RUST_INTEGRATION_RE = re.compile(r"^tests/[^/]+\.rs$")
RUST_TEST_CFG_RE = re.compile(r"#\[\s*(?:cfg\(\s*test\b|cfg\(\s*all\(\s*test\b|test\b)")

MISSING_LEVEL = "Level"
MISSING_MACHINE = "Machine"
DIRECT_SKU = "Direct SKU"
BARE_PLATFORM = "Bare platform"
_PROBLEM_TEXT = {
    DIRECT_SKU: "disallowed SKU mark",
    BARE_PLATFORM: "disallowed platform mark",
    MISSING_LEVEL: "missing level mark",
    MISSING_MACHINE: "missing machine mark",
}


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(_repo_root())
        except ValueError:
            return _normalize(path)
    return _normalize(str(candidate))


@lru_cache(maxsize=1)
def _mark_module() -> ModuleType:
    path = _repo_root() / "py_test" / "helpers" / "mark.py"
    spec = importlib.util.spec_from_file_location("_router_mark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mark_name_re(names: tuple[str, ...]) -> re.Pattern[str]:
    if not names:
        return re.compile(r"(?!)")
    return re.compile(
        r"mark\.(?:" + "|".join(re.escape(name) for name in names) + r")\b"
    )


def _feature_re(names: tuple[str, ...]) -> re.Pattern[str]:
    if not names:
        return re.compile(r"(?!)")
    body = "|".join(re.escape(name) for name in names)
    return re.compile(rf"""feature\s*=\s*["'](?:{body})["']""")


@lru_cache(maxsize=1)
def _cargo_feature_names() -> frozenset[str]:
    text = (_repo_root() / "Cargo.toml").read_text(encoding="utf-8")
    in_features = False
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_features = stripped == "[features]"
            continue
        if not in_features or not stripped or stripped.startswith("#"):
            continue
        name = stripped.split("=", 1)[0].strip()
        if name:
            names.add(name)
    return frozenset(names)


@lru_cache(maxsize=1)
def level_names() -> tuple[str, ...]:
    return tuple(sorted(_mark_module().get_level_markers()))


@lru_cache(maxsize=1)
def platform_names() -> tuple[str, ...]:
    return tuple(sorted(_mark_module().get_supported_platforms()))


@lru_cache(maxsize=1)
def sku_names() -> tuple[str, ...]:
    return tuple(sorted(_mark_module().get_hardware_mark_list()))


@lru_cache(maxsize=1)
def bare_platform_names() -> tuple[str, ...]:
    return tuple(sorted(set(platform_names()) - _DIRECT_PLATFORM_ALLOWLIST))


@lru_cache(maxsize=1)
def rust_level_features() -> tuple[str, ...]:
    return tuple(sorted(set(level_names()) & _cargo_feature_names()))


@lru_cache(maxsize=1)
def rust_machine_features() -> tuple[str, ...]:
    """Cargo features that name a machine, not the ``gpu`` device class."""
    platforms = set(platform_names()) - {"gpu"}
    return tuple(sorted(platforms & _cargo_feature_names()))


def _read(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        file_path = _repo_root() / path
    if not file_path.is_file():
        return None
    return file_path.read_text(encoding="utf-8")


def _blank_keeping_newlines(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def _source_without_comments_and_strings(contents: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(contents).readline))
    except (tokenize.TokenError, SyntaxError):
        return contents
    blanked = {tokenize.COMMENT, tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        token_type = getattr(tokenize, name, None)
        if token_type is not None:
            blanked.add(token_type)
    lines = contents.splitlines(keepends=True)
    for tok in tokens:
        if tok.type not in blanked or tok.start >= tok.end:
            continue
        (start_line, start_col), (end_line, end_col) = tok.start, tok.end
        if start_line == end_line:
            index = start_line - 1
            line = lines[index]
            lines[index] = (
                line[:start_col]
                + _blank_keeping_newlines(line[start_col:end_col])
                + line[end_col:]
            )
            continue
        first = lines[start_line - 1]
        lines[start_line - 1] = first[:start_col] + _blank_keeping_newlines(
            first[start_col:]
        )
        for index in range(start_line, end_line - 1):
            lines[index] = _blank_keeping_newlines(lines[index])
        lines[end_line - 1] = (
            _blank_keeping_newlines(lines[end_line - 1][:end_col])
            + lines[end_line - 1][end_col:]
        )
    return "".join(lines)


def _strip_rust_comments(contents: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", contents, flags=re.S)
    return re.sub(r"//.*?$", "", without_blocks, flags=re.M)


def _is_pytest_module(scanned: str) -> bool:
    if "import pytest" in scanned or "pytestmark" in scanned or "@pytest" in scanned:
        return True
    return _TEST_ITEM_RE.search(scanned) is not None


def _python_problems(rel: str, contents: str) -> list[str]:
    if not PY_TEST_FILE_RE.match(rel) or rel.startswith("py_test/e2e/"):
        return []
    scanned = _source_without_comments_and_strings(contents)
    if not _is_pytest_module(scanned):
        return []
    missing: list[str] = []
    if rel not in _ALLOWED_DIRECT_MARK_FILES:
        if _mark_name_re(sku_names()).search(scanned):
            missing.append(DIRECT_SKU)
        if _mark_name_re(bare_platform_names()).search(scanned):
            missing.append(BARE_PLATFORM)
    if not _mark_name_re(level_names()).search(scanned):
        missing.append(MISSING_LEVEL)
    has_machine = _mark_name_re(platform_names()).search(scanned) or HELPER_RE.search(
        scanned
    )
    if not has_machine:
        missing.append(MISSING_MACHINE)
    return missing


def _is_rust_test(rel: str, contents: str) -> bool:
    if rel.startswith("tests/common/") or rel.startswith("tests/fixtures/"):
        return False
    if RUST_INTEGRATION_RE.match(rel):
        return True
    return (
        rel.startswith("src/")
        and rel.endswith(".rs")
        and bool(RUST_TEST_CFG_RE.search(contents))
    )


def _rust_problems(rel: str, contents: str) -> list[str]:
    if not _is_rust_test(rel, contents):
        return []
    code = _strip_rust_comments(contents)
    missing: list[str] = []
    if not _feature_re(rust_level_features()).search(code):
        missing.append(MISSING_LEVEL)
    if not _feature_re(rust_machine_features()).search(code):
        missing.append(MISSING_MACHINE)
    return missing


def problems_for(path: str) -> list[str]:
    rel = _relative(path)
    contents = _read(path)
    if contents is None:
        return []
    if rel.endswith(".py"):
        return _python_problems(rel, contents)
    if rel.endswith(".rs"):
        return _rust_problems(rel, contents)
    return []


def _format_problems(problems: list[str]) -> str:
    return " and ".join(_PROBLEM_TEXT.get(problem, problem) for problem in problems)


def main(argv: list[str]) -> int:
    findings = {path: found for path in argv if (found := problems_for(path))}
    if not findings:
        return 0
    lines = "\n".join(
        f"  - {_relative(path)} [{_format_problems(found)}]"
        for path, found in findings.items()
    )
    print(
        "error: test files failed the mark check.\n\n"
        f"Python level marks: {', '.join(level_names()) or '(none)'}\n"
        "Python machine: pytest.mark.cpu, or hardware_test(...) / hardware_marks(...)\n"
        "Do not write pytest.mark.<SKU> "
        f"({', '.join(sku_names()) or 'none'}) or pytest.mark.<platform> "
        f"({', '.join(bare_platform_names()) or 'none'}). "
        "pytest.mark.cpu stays direct because CPU has no SKU.\n"
        "Rust features: one level "
        f"({', '.join(rust_level_features()) or 'none'}) and one machine "
        f"({', '.join(rust_machine_features()) or 'none'}), "
        'for example feature = "ready" and feature = "cpu".\n\n'
        "The following files failed:\n"
        f"{lines}\n\n"
        "To skip: SKIP=check-mark git commit ..."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
