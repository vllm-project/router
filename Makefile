# VLLM Router Makefile
# Provides convenient shortcuts for installation, development, and deployment tasks

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS := linux
    PKG_MANAGER := $(shell command -v apt-get 2>/dev/null || command -v yum 2>/dev/null || command -v dnf 2>/dev/null)
endif
ifeq ($(UNAME_S),Darwin)
    OS := macos
    PKG_MANAGER := $(shell command -v brew 2>/dev/null)
endif

# Python command detection
PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
PIP := $(shell command -v pip3 2>/dev/null || command -v pip 2>/dev/null)

# Check if sccache is available and set RUSTC_WRAPPER accordingly
SCCACHE := $(shell which sccache 2>/dev/null)
ifdef SCCACHE
    export RUSTC_WRAPPER := $(SCCACHE)
    $(info Using sccache for compilation caching)
else
    $(info sccache not found. Install it for faster builds: cargo install sccache)
endif

.PHONY: help install install-deps install-rust install-protobuf install-python-deps \
        build build-rust build-python install-python \
        run run-example \
        test bench bench-quick bench-baseline bench-compare \
        clean clean-all \
        check-deps

help: ## Show this help message
	@echo "VLLM Router - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Installation & Setup:"
	@grep -E '^(install|check-deps|setup).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Build & Run:"
	@grep -E '^(build|run).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Development & Testing:"
	@grep -E '^(test|check|fmt|dev|pre-commit).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Benchmarking:"
	@grep -E '^(bench|perf).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Maintenance:"
	@grep -E '^(clean|sccache|docs).*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Installation & Setup
# ============================================================================

check-deps: ## Check if all required dependencies are installed
	@echo "Checking dependencies..."
	@echo "OS: $(OS)"
	@command -v cargo >/dev/null 2>&1 || { echo "❌ Rust/Cargo not found"; exit 1; }
	@echo "✓ Rust/Cargo: $$(cargo --version)"
	@command -v rustc >/dev/null 2>&1 && echo "✓ Rustc: $$(rustc --version)" || true
	@command -v protoc >/dev/null 2>&1 || { echo "❌ protoc (Protocol Buffers compiler) not found"; exit 1; }
	@echo "✓ protoc: $$(protoc --version)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "❌ Python not found"; exit 1; }
	@echo "✓ Python: $$($(PYTHON) --version)"
	@command -v $(PIP) >/dev/null 2>&1 || { echo "❌ pip not found"; exit 1; }
	@echo "✓ pip: $$($(PIP) --version)"
	@echo "✓ All dependencies are installed!"

install-rust: ## Install Rust and Cargo
	@echo "Installing Rust..."
	@if command -v cargo >/dev/null 2>&1; then \
		echo "Rust is already installed: $$(cargo --version)"; \
	else \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo "Please run: source $$HOME/.cargo/env"; \
	fi

install-protobuf: ## Install Protocol Buffers compiler
	@echo "Installing protobuf compiler for $(OS)..."
ifeq ($(OS),linux)
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y protobuf-compiler libprotobuf-dev; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y protobuf-compiler protobuf-devel; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y protobuf-compiler protobuf-devel; \
	else \
		echo "Please install protobuf-compiler manually"; \
		exit 1; \
	fi
else ifeq ($(OS),macos)
	@if command -v brew >/dev/null 2>&1; then \
		brew install protobuf; \
	else \
		echo "Please install Homebrew first: https://brew.sh"; \
		exit 1; \
	fi
endif
	@echo "✓ protobuf installed: $$(protoc --version)"

install-python-deps: ## Install Python dependencies
	@echo "Installing Python dependencies..."
	@if $(PIP) install --upgrade pip setuptools wheel 2>&1 | grep -q "externally-managed-environment"; then \
		echo ""; \
		echo "⚠️  System Python is externally managed (macOS/Homebrew)."; \
		echo ""; \
		echo "Recommended: Use a virtual environment:"; \
		echo "  python3 -m venv venv"; \
		echo "  source venv/bin/activate"; \
		echo "  make install"; \
		echo ""; \
		echo "Attempting installation with --user --break-system-packages..."; \
		$(PIP) install --user --break-system-packages setuptools-rust build; \
	else \
		$(PIP) install setuptools-rust build; \
	fi
	@echo "✓ Python build dependencies installed"

install-deps: install-protobuf install-python-deps ## Install all system dependencies
	@echo "✓ All dependencies installed successfully!"

install: install-deps build-rust build-python install-python ## Complete installation (deps + build + install)
	@echo ""
	@echo "=========================================="
	@echo "✓ vLLM Router installed successfully!"
	@echo "=========================================="
	@echo ""
	@echo "Binary location: ./target/release/vllm-router"
	@echo "Python package: vllm-router (installed)"
	@echo ""
	@echo "Quick start:"
	@echo "  make run-example    # Run with example configuration"
	@echo "  make help           # Show all available commands"
	@echo ""

# ============================================================================
# Build & Run
# ============================================================================

build-rust: ## Build Rust binary in release mode
	@echo "Building Rust binary..."
	@cargo build --release
	@echo "✓ Binary built: ./target/release/vllm-router"

build-python: ## Build Python wheel package
	@echo "Building Python package..."
	@$(PYTHON) -m build
	@echo "✓ Python package built in dist/"

build: build-rust build-python ## Build both Rust binary and Python package
	@echo "✓ Build complete!"

install-python: build-python ## Install Python package
	@echo "Installing Python package..."
	@if $(PIP) install --force-reinstall dist/*.whl 2>&1 | grep -q "externally-managed-environment"; then \
		echo "System Python is externally managed. Installing with --user --break-system-packages..."; \
		$(PIP) install --user --break-system-packages --force-reinstall dist/*.whl; \
	fi
	@echo "✓ Python package installed"

rebuild-python: ## Rebuild and reinstall Python package (for development)
	@echo "Rebuilding and reinstalling Python package..."
	@$(PYTHON) -m build
	@if $(PIP) install --force-reinstall dist/*.whl 2>&1 | grep -q "externally-managed-environment"; then \
		echo "System Python is externally managed. Installing with --user --break-system-packages..."; \
		$(PIP) install --user --break-system-packages --force-reinstall dist/*.whl; \
	fi
	@echo "✓ Python package rebuilt and installed"

run-example: build-rust ## Run router with example configuration
	@echo "Starting vLLM Router with example configuration..."
	@echo "Note: Make sure you have vLLM workers running on the specified URLs"
	@echo ""
	./target/release/vllm-router \
		--worker-urls http://localhost:8000 http://localhost:8001 \
		--policy round_robin \
		--host 0.0.0.0 \
		--port 8080

run: run-example ## Alias for run-example

# ============================================================================
# Development & Testing
# ============================================================================

test: ## Run all tests
	@echo "Running tests..."
	@cargo test

bench: ## Run full benchmark suite
	@echo "Running full benchmarks..."
	@python3 scripts/run_benchmarks.py

bench-quick: ## Run quick benchmarks only
	@echo "Running quick benchmarks..."
	@python3 scripts/run_benchmarks.py --quick

bench-baseline: ## Save current performance as baseline
	@echo "Saving performance baseline..."
	@python3 scripts/run_benchmarks.py --save-baseline main

bench-compare: ## Compare with saved baseline
	@echo "Comparing with baseline..."
	@python3 scripts/run_benchmarks.py --compare-baseline main

bench-ci: ## Run benchmarks suitable for CI (quick mode)
	@echo "Running CI benchmarks..."
	@python3 scripts/run_benchmarks.py --quick

# ============================================================================
# Benchmarking
# ============================================================================

check: ## Run cargo check and clippy
	@echo "Running cargo check..."
	@cargo check
	@echo "Running clippy..."
	@cargo clippy

fmt: ## Format code with rustfmt
	@echo "Formatting code..."
	@cargo fmt

# Development workflow shortcuts
dev-setup: install test ## Set up complete development environment
	@echo "✓ Development environment ready!"

pre-commit: fmt check test bench-quick ## Run pre-commit checks
	@echo "✓ Pre-commit checks passed!"

# Benchmark analysis shortcuts
bench-report: ## Open benchmark HTML report
	@if [ -f "target/criterion/request_processing/report/index.html" ]; then \
		echo "Opening benchmark report..."; \
		if command -v xdg-open >/dev/null 2>&1; then \
			xdg-open target/criterion/request_processing/report/index.html; \
		elif command -v open >/dev/null 2>&1; then \
			open target/criterion/request_processing/report/index.html; \
		else \
			echo "Please open target/criterion/request_processing/report/index.html in your browser"; \
		fi \
	else \
		echo "No benchmark report found. Run 'make bench' first."; \
	fi

bench-clean: ## Clean benchmark results
	@echo "Cleaning benchmark results..."
	@rm -rf target/criterion

# Performance monitoring
perf-monitor: ## Run continuous performance monitoring
	@echo "Starting performance monitoring..."
	@if command -v watch >/dev/null 2>&1; then \
		watch -n 300 'make bench-quick'; \
	else \
		echo "Warning: 'watch' command not found. Install it or run 'make bench-quick' manually."; \
	fi

# sccache management targets
setup-sccache: ## Install and configure sccache
	@echo "Setting up sccache..."
	@./scripts/setup-sccache.sh

sccache-stats: ## Show sccache statistics
	@if [ -n "$(SCCACHE)" ]; then \
		echo "sccache statistics:"; \
		sccache -s; \
	else \
		echo "sccache not installed. Run 'make setup-sccache' to install it."; \
	fi

sccache-clean: ## Clear sccache cache
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Clearing sccache cache..."; \
		sccache -C; \
		echo "sccache cache cleared"; \
	else \
		echo "sccache not installed"; \
	fi

sccache-stop: ## Stop the sccache server
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Stopping sccache server..."; \
		sccache --stop-server || true; \
	else \
		echo "sccache not installed"; \
	fi

# ============================================================================
# Maintenance & Cleanup
# ============================================================================

clean: ## Clean Rust build artifacts
	@echo "Cleaning Rust build artifacts..."
	@cargo clean
	@echo "✓ Rust artifacts cleaned"

clean-python: ## Clean Python build artifacts
	@echo "Cleaning Python build artifacts..."
	@rm -rf dist/ build/ *.egg-info py_src/*.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Python artifacts cleaned"

clean-all: clean clean-python bench-clean ## Clean all build artifacts
	@echo "✓ All artifacts cleaned"

docs: ## Generate and open Rust documentation
	@echo "Generating documentation..."
	@cargo doc --open
