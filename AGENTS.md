# AGENTS.md - vLLM Router Development Guide

This document provides guidance for AI agents working on the vLLM Router codebase.

## Build Commands

### Rust
```bash
cargo build --release      # Release build
cargo build                # Debug build
make build                 # Via Makefile
```

### Python
```bash
pip install -e .[dev]      # Dev install with dependencies
python -m build            # Build wheel
pip install setuptools-rust wheel build  # Build tools
```

## Lint Commands

### Rust
```bash
cargo fmt --check          # Check formatting
cargo clippy --all-targets --all-features -- -D warnings  # Lint
make check                 # Both via Makefile
```

### Python
```bash
python -m black --check py_src/ py_test/      # Format check
python -m ruff check py_src/ py_test/         # Lint
python -m mypy py_src/                        # Type check
pip install pre-commit && pre-commit install  # Pre-commit hooks
```

## Test Commands

### Python Tests (Pytest)
```bash
pytest py_test/                           # All tests
pytest py_test/test_launch_router.py      # Specific file
pytest py_test/test_launch_router.py::TestLaunchRouter::test_launch_router_common  # Specific test
pytest py_test/unit/                      # Unit tests only
pytest py_test/integration/               # Integration tests
pytest py_test/e2e/                       # E2E tests
pytest -k "test_launch" py_test/          # Pattern match
pytest py_test/ -v                        # Verbose
pytest py_test/ --cov=vllm_router         # With coverage
```

### Rust Tests (Cargo)
```bash
cargo test                    # All tests
cargo test --lib              # Lib tests only
cargo test --bins             # Binary tests
cargo test --test test_consistent_hash_policy    # Specific file
cargo test test_consistent_hash_policy_creation  # Specific test
cargo test --test '*'         # Integration tests
RUST_BACKTRACE=1 cargo test   # With backtrace
```

### Shortcuts
```bash
make test              # Rust tests
make pre-commit        # Format, check, benchmark
make bench             # Full benchmarks
make bench-quick       # Quick benchmarks
```

## Code Style - Rust

**Imports:** Group: std, external, local. Use `crate::` for local modules.
```rust
use std::sync::Arc;
use axum::{extract::Request, Json};
use serde::{Deserialize, Serialize};
use crate::config::types::RouterConfig;
```

**Naming:** Structs/Enums/Traits `PascalCase`, functions/variables `snake_case`, constants `SCREAMING_SNAKE_CASE`.

**Error Handling:** Use `thiserror` for error types, `anyhow` for context, `?` for propagation.
```rust
#[derive(Debug, thiserror::Error)]
#[error("Configuration error: {0}")]
pub struct ConfigurationError(String);
```

**Concurrency:** Use `Arc`, `tokio::sync` primitives, `dashmap` for concurrent maps.

## Code Style - Python

**Imports:** stdlib, third-party, local. Absolute imports, alphabetical within groups.
```python
import argparse
import logging
from typing import List, Optional
import setproctitle
from aiohttp import ClientSession
from vllm_router.mini_lb import MiniLoadBalancer
```

**Naming:** Classes `PascalCase`, functions/variables `snake_case`, constants `SCREAMING_SNAKE_CASE`.

**Type Hints:** Use for all public functions. `Optional[T]` over `Union[T, None]`.

**Testing:** Test files `test_*.py`, classes `Test*`, functions `test_*`. Use fixtures from `py_test/fixtures/`.

## Project Structure

```
vllm-router/
├── src/              # Rust (core, routers, policies, protocols, tokenizer)
├── py_src/           # Python (vllm_router package)
├── py_test/          # Python tests (unit/, integration/, e2e/, fixtures/)
├── tests/            # Rust integration tests
├── benches/          # Rust benchmarks
├── scripts/          # Utility scripts
├── Cargo.toml        # Rust config
├── pyproject.toml    # Python config
├── Makefile          # Dev shortcuts
└── .pre-commit-config.yaml  # Pre-commit hooks
```

## Development Workflow

1. Make changes following style guidelines
2. Run `make fmt` to format
3. Run `make check` to lint
4. Run relevant tests
5. Run `make pre-commit` before committing