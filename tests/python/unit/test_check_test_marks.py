"""Regression checks for Rust coverage exclusion enforcement."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "check_test_marks", ROOT / "tools/pre_commit/check_test_marks.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

GATE = '#[cfg(all(test, feature = "cpu", feature = "ready"))]'
MARKER = "#[cfg_attr(coverage_nightly, coverage(off))]"


@pytest.mark.parametrize(
    "item",
    [
        "mod tests {}",
        "mod tests;",
        "pub(crate) mod custom_tests {}",
        "fn helper() {}",
        "pub(super) async fn helper() {}",
        "impl Sample {}",
    ],
)
def test_each_test_only_item_requires_its_own_marker(item):
    assert checker._rust_coverage_problems(f"{GATE}\n{item}") == [
        "missing coverage exclusion at line 1"
    ]
    assert checker._rust_coverage_problems(f"{GATE}\n{MARKER}\n{item}") == []
    assert checker._rust_coverage_problems(f"{MARKER}\n{GATE}\n{item}") == []


def test_marked_module_does_not_hide_unmarked_sibling():
    source = f"{GATE}\n{MARKER}\nmod first {{}}\n{GATE}\nmod second {{}}"
    assert checker._rust_coverage_problems(source) == [
        "missing coverage exclusion at line 4"
    ]


def test_multiline_attributes_and_comments():
    source = """
#[cfg(all(
    feature = "cpu",
    test,
    feature = "ready",
))]
// Still part of the same declaration's attributes.
#[cfg_attr(
    coverage_nightly,
    coverage(off),
)]
mod tests;
"""
    assert checker._rust_coverage_problems(source) == []


@pytest.mark.parametrize(
    "item",
    [
        "use super::helper;",
        "field: bool,",
        "field: None,",
        "if enabled { helper(); }",
        "helper();",
        "std::thread_local! { static HOOK: bool = false; }",
    ],
)
def test_non_coverage_items_do_not_require_markers(item):
    assert checker._rust_coverage_problems(f"#[cfg(test)]\n{item}") == []


@pytest.mark.parametrize(
    "source",
    [
        f"// {MARKER}\n{GATE}\nmod tests {{}}",
        f"/* outer /* inner */ {MARKER} */\n{GATE}\nmod tests {{}}",
        f'#[doc = r#"{MARKER}"#]\n{GATE}\nmod tests {{}}',
        f'{GATE}\n#[cfg_attr(feature = "coverage_nightly", coverage(off))]\nmod tests {{}}',
    ],
)
def test_comments_strings_and_wrong_cfg_do_not_satisfy_marker(source):
    assert len(checker._rust_coverage_problems(source)) == 1


def test_fake_items_inside_literals_are_ignored():
    source = f'const TEXT: &str = r##"{GATE}\nmod tests {{}}"##;'
    assert checker._rust_coverage_problems(source) == []


@pytest.mark.parametrize(
    "gate",
    ["not(test)", 'any(test, feature = "cpu")', 'all(not(test), feature = "cpu")'],
)
def test_production_cfg_gates_are_not_test_only(gate):
    assert checker._rust_coverage_problems(f"#[cfg({gate})]\nfn service() {{}}") == []


def test_existing_hook_entrypoint_enforces_only_source_items(monkeypatch):
    source = f"{GATE}\nmod tests {{}}"
    monkeypatch.setattr(checker, "_read", lambda _: source)
    assert checker.problems_for("src/example.rs") == [
        "missing coverage exclusion at line 1"
    ]
    assert checker.problems_for("tests/rust/example.rs") == []


def test_entrypoint_checks_helpers_with_reordered_cfg_terms():
    source = '#[cfg(all(feature = "cpu", test, feature = "ready"))]\nfn helper() {}'
    assert checker._rust_problems("src/helper.rs", source) == [
        "missing coverage exclusion at line 1"
    ]


def test_cli_fails_and_reports_a_missing_exclusion(monkeypatch, capsys):
    monkeypatch.setattr(checker, "_read", lambda _: f"{GATE}\nmod tests {{}}")
    assert checker.main(["src/example.rs"]) == 1
    output = capsys.readouterr().out
    assert "src/example.rs [missing coverage exclusion at line 1]" in output


def test_current_source_tree_has_no_missing_exclusions():
    for path in (ROOT / "src").rglob("*.rs"):
        assert (
            checker._rust_coverage_problems(path.read_text(encoding="utf-8")) == []
        ), path


def test_removing_real_marker_is_detected():
    source = (ROOT / "src/otel_http.rs").read_text(encoding="utf-8")
    assert checker._rust_coverage_problems(source) == []
    assert len(checker._rust_coverage_problems(source.replace(MARKER, "", 1))) == 1
