#!/usr/bin/env python3
"""Fail if a tests/rust/*.rs file is not registered in Cargo.toml.

Cargo only auto-discovers tests/*.rs (or tests/<name>/main.rs). Nested
integration tests under tests/rust/ need an explicit [[test]] entry or
`cargo test --test '*'` will skip them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARGO_TOML = ROOT / "Cargo.toml"
TESTS_RUST = ROOT / "tests" / "rust"
NAME_OR_PATH = re.compile(r'^(name|path)\s*=\s*"([^"]+)"\s*$')


def rust_integration_files() -> set[str]:
    if not TESTS_RUST.is_dir():
        return set()
    return {f"tests/rust/{path.name}" for path in TESTS_RUST.glob("*.rs") if path.is_file()}


def cargo_test_targets(text: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if current and "name" in current and "path" in current:
            targets.append((current["name"], current["path"]))
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if line == "[[test]]":
            flush()
            current = {}
            continue
        if current is None:
            continue
        if line.startswith("["):
            flush()
            if line == "[[test]]":
                current = {}
            continue
        match = NAME_OR_PATH.match(line)
        if match:
            current[match.group(1)] = match.group(2)
    flush()
    return targets


def check(cargo_text: str, files: set[str]) -> list[str]:
    errors: list[str] = []
    rust_targets = [
        (name, path) for name, path in cargo_test_targets(cargo_text) if path.startswith("tests/rust/")
    ]
    registered = {path for _, path in rust_targets}

    for path in sorted(files - registered):
        stem = Path(path).stem
        errors.append(
            f"missing [[test]] for {path}. Add:\n"
            f"[[test]]\n"
            f'name = "{stem}"\n'
            f'path = "{path}"'
        )

    for path in sorted(registered - files):
        errors.append(f"{path} is listed in Cargo.toml but the file does not exist")

    for name, path in rust_targets:
        if path not in files:
            continue
        stem = Path(path).stem
        if name != stem:
            errors.append(f'{path} is registered as name = "{name}"; expected "{stem}"')

    return errors


def main() -> int:
    errors = check(CARGO_TOML.read_text(encoding="utf-8"), rust_integration_files())
    if not errors:
        return 0
    print("Rust integration tests under tests/rust/ must be listed in Cargo.toml.")
    print("cargo test --test '*' does not auto-discover nested files.")
    for error in errors:
        print(f"\n{error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
