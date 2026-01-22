# Contributing to vLLM Router

This guide covers development setup, build processes, and testing for vLLM Router.

## Development Environment Setup

### Prerequisites

- **Rust** (1.70+) and Cargo
- **Protocol Buffers compiler** (protoc)
- **Python** (3.8+) with pip
- **Git**

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-router.git
cd vllm-router

# Set up development environment (installs deps, builds, runs tests)
make dev-setup
```

This single command will:
1. Check and install dependencies
2. Build Rust binary and Python package
3. Run all tests

### Manual Setup

If you prefer step-by-step setup:

```bash
# Check dependencies
make check-deps

# Install missing dependencies
make install-deps

# Build everything
make build

# Run tests
make test
```

## Make Targets Reference

### Installation & Setup

```bash
make check-deps           # Check if all dependencies are installed
make install-rust         # Install Rust toolchain
make install-protobuf     # Install Protocol Buffers compiler
make install-python-deps  # Install Python build dependencies
make install-deps         # Install all system dependencies
make install              # Complete installation (deps + build + install)
```

### Build Targets

```bash
make build-rust           # Build Rust binary in release mode
make build-python         # Build Python wheel package
make build                # Build both Rust and Python
make install-python       # Install Python package
make rebuild-python       # Rebuild and reinstall Python package (for development)
```

**Build outputs:**
- Rust binary: `./target/release/vllm-router`
- Python wheel: `dist/*.whl`

### Development Targets

```bash
make test                 # Run all Rust tests
make check                # Run cargo check and clippy
make fmt                  # Format code with rustfmt
make dev-setup            # Complete dev environment setup
make pre-commit           # Run all pre-commit checks (fmt, check, test, bench-quick)
```

### Benchmarking

```bash
make bench                # Run full benchmark suite
make bench-quick          # Run quick benchmarks only
make bench-baseline       # Save current performance as baseline
make bench-compare        # Compare with saved baseline
make bench-ci             # Run benchmarks suitable for CI
make bench-report         # Open benchmark HTML report
make bench-clean          # Clean benchmark results
```

### Maintenance

```bash
make clean                # Clean Rust build artifacts
make clean-python         # Clean Python build artifacts
make clean-all            # Clean all artifacts (Rust, Python, benchmarks)
make docs                 # Generate and open Rust documentation
```

### sccache (Optional)

For faster builds, install and use sccache:

```bash
make setup-sccache        # Install and configure sccache
make sccache-stats        # Show sccache statistics
make sccache-clean        # Clear sccache cache
make sccache-stop         # Stop sccache server
```

## Development Workflow

### Daily Development

```bash
# 1. Make code changes

# 2. Format code
make fmt

# 3. Check for issues
make check

# 4. Run tests
make test

# 5. For Python changes, rebuild
make rebuild-python
```

### Before Committing

Run pre-commit checks to ensure code quality:

```bash
make pre-commit
```

This runs:
- Code formatting (`cargo fmt`)
- Linting (`cargo clippy`)
- All tests (`cargo test`)
- Quick benchmarks

### Testing Changes

#### Rust Tests

```bash
# Run all tests
make test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture
```

#### Python Tests

```bash
# After rebuilding Python package
make rebuild-python

# Run Python tests (if available)
pytest py_test/
```

## Build System Details

### Rust Build

The project uses Cargo for Rust builds:

```bash
# Debug build (faster compilation, slower runtime)
cargo build

# Release build (slower compilation, optimized runtime)
cargo build --release
```

**Build configuration** (`Cargo.toml`):
- LTO: thin
- Codegen units: 1 (for release)
- Includes gRPC code generation via `build.rs`

### Python Build

The project uses `setuptools-rust` to build Python bindings:

```bash
# Build wheel
python -m build

# Install wheel
pip install dist/*.whl

# Force reinstall (for development)
pip install --force-reinstall dist/*.whl
```

**Build configuration** (`pyproject.toml`):
- Uses PyO3 for Rust-Python bindings
- Generates `vllm-router` CLI entry point

### Protocol Buffers

gRPC definitions are compiled during build via `build.rs`:

```rust
// src/proto/vllm_scheduler.proto is compiled to Rust code
tonic_build::compile_protos("src/proto/vllm_scheduler.proto")?;
```

## Code Quality

### Formatting

```bash
# Format Rust code
make fmt

# Check formatting without modifying
cargo fmt -- --check
```

### Linting

```bash
# Run clippy
make check

# Run clippy with all warnings
cargo clippy -- -W clippy::all
```

### Testing

```bash
# Run all tests
make test

# Run tests with coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html
```

## Performance Testing

### Quick Benchmarks

For rapid iteration:

```bash
make bench-quick
```

### Full Benchmarks

For comprehensive performance analysis:

```bash
make bench
```

### Benchmark Comparison

To track performance changes:

```bash
# Save baseline
make bench-baseline

# Make changes...

# Compare with baseline
make bench-compare
```

### View Results

```bash
# Open HTML report
make bench-report

# Or manually
open target/criterion/request_processing/report/index.html
```

## Troubleshooting

### Build Failures

```bash
# Clean and rebuild
make clean-all
make build
```

### Rust Analyzer Issues (VSCode)

Add to `.vscode/settings.json`:

```json
{
  "rust-analyzer.linkedProjects": ["/absolute/path/to/vllm-router/Cargo.toml"]
}
```

### Python Package Issues

**Error: `externally-managed-environment` (macOS)**

The Makefile automatically handles this by using the `--user` flag. If you prefer a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
make install
```

**Other issues:**

```bash
# Clean Python artifacts
make clean-python

# Rebuild
make rebuild-python
```

### Dependency Issues

```bash
# Check what's missing
make check-deps

# Reinstall dependencies
make install-deps
```

## CI/CD Pipeline

The project uses Buildkite for CI/CD (`.buildkite/pipeline.yml`):

### Build & Test Steps

1. **Fast Checks**: Format and lint checks
2. **Build**: Rust binary and Python wheels
3. **Tests**: Unit, integration, and Python tests
4. **Benchmarks**: Performance regression tests
5. **Docker Build**: Container image creation

### Release Pipeline

Triggered on version tags (`v*`):

1. **Build Artifacts**: Wheels and source distribution
2. **PyPI Publishing**: Automatic upload to PyPI
3. **Container Images**: Docker image publishing

## Project Structure

```
vllm-router/
├── src/                    # Rust source code
│   ├── main.rs            # Binary entry point
│   ├── lib.rs             # Library entry point
│   ├── core/              # Core routing logic
│   ├── policies/          # Load balancing policies
│   ├── proto/             # gRPC protocol definitions
│   └── ...
├── py_src/                # Python source code
│   └── vllm_router/       # Python package
├── tests/                 # Rust tests
├── py_test/               # Python tests
├── benches/               # Benchmarks
├── docs/                  # Documentation
├── scripts/               # Deployment scripts
├── Cargo.toml             # Rust dependencies
├── pyproject.toml         # Python package config
├── build.rs               # Build script (protobuf)
└── Makefile               # Build automation
```

## Getting Help

- Check [docs/](docs/) for detailed documentation
- Review existing issues and PRs
- Ask questions in discussions
