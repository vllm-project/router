# Installation Guide

This guide provides detailed installation instructions for vLLM Router on Linux and macOS systems.

## Quick Installation

For most users, the automated installation is recommended:

```bash
make install
```

This single command will:
1. Check for required dependencies
2. Install missing system dependencies (protobuf)
3. Build the Rust binary
4. Build and install the Python package

## Prerequisites

### Required Dependencies

- **Rust and Cargo** (1.70+)
- **Protocol Buffers compiler** (protoc)
- **Python** (3.8+) with pip

### Check Dependencies

Before installation, verify what's already installed:

```bash
make check-deps
```

## Step-by-Step Installation

### 1. Install Rust and Cargo

If Rust is not installed, use the official installer:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

Or use the make target:

```bash
make install-rust
```

### 2. Install Protocol Buffers Compiler

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y protobuf-compiler libprotobuf-dev
```

#### Linux (RHEL/CentOS/Fedora)

```bash
# Using yum
sudo yum install -y protobuf-compiler protobuf-devel

# Or using dnf
sudo dnf install -y protobuf-compiler protobuf-devel
```

#### macOS

```bash
brew install protobuf
```

Or use the make target (works on all platforms):

```bash
make install-protobuf
```

### 3. Install Python Build Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install setuptools-rust build
```

Or use the make target:

```bash
make install-python-deps
```

### 4. Build the Project

#### Build Rust Binary

```bash
cargo build --release
```

The binary will be available at `./target/release/vllm-router`.

Or use the make target:

```bash
make build-rust
```

#### Build Python Package

```bash
python -m build
```

Or use the make target:

```bash
make build-python
```

### 5. Install Python Package

```bash
pip install dist/*.whl
```

Or use the make target:

```bash
make install-python
```

## Installation Options

### Option 1: Complete Installation (Recommended)

Install everything in one command:

```bash
make install
```

### Option 2: Install Dependencies Only

If you want to install dependencies but build later:

```bash
make install-deps
```

### Option 3: Build Only (No Installation)

If dependencies are already installed:

```bash
make build
```

## Development Installation

For development work, you may want to rebuild and reinstall frequently:

```bash
# Rebuild and reinstall Python package
make rebuild-python

# Or manually
python -m build && pip install --force-reinstall dist/*.whl
```

## Troubleshooting

### Missing protoc

**Error:** `protoc: command not found` or `protobuf-compiler not found`

**Solution:**
```bash
make install-protobuf
```

### Missing Rust

**Error:** `cargo: command not found`

**Solution:**
```bash
make install-rust
source $HOME/.cargo/env
```

### Build Failures

If you encounter build errors, try cleaning and rebuilding:

```bash
make clean-all
make install
```

### Python Installation Issues

**Error: `externally-managed-environment` (macOS/Homebrew)**

If you see this error on macOS, the Makefile will automatically use the `--user` flag. Alternatively:

```bash
# Option 1: Use a virtual environment (recommended for development)
python3 -m venv venv
source venv/bin/activate
make install

# Option 2: Install with --user flag (automatic in Makefile)
pip install --user setuptools-rust build

# Option 3: Use pipx for the CLI tool only
brew install pipx
pipx install dist/*.whl
```

**Other permission errors:**

If you get permission errors when installing Python packages:

1. Use a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
make install
```

2. Install for user only:
```bash
pip install --user dist/*.whl
```

## Verifying Installation

After installation, verify the binary works:

```bash
# Check binary version
./target/release/vllm-router --help

# Check Python package
vllm-router --help
```

## Next Steps

- See [Basic Routing](model-routing/basic-routing.md) for usage examples
- See [Configuration](model-monitoring/configuration.md) for advanced settings
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup
