# vLLM Router Systemd Service Installation Guide

## Overview

The vLLM Router can be installed as a systemd service for automatic startup and management. This guide explains how to install and configure the service.

## Prerequisites

- **Operating System**: Linux with systemd (Ubuntu, Debian, CentOS, RHEL, etc.)
- **Root Access**: For system-wide installation (recommended)
- **vLLM Router**: Built or installed binary

## Installation Methods

### Method 1: Using the Installation Script (Recommended)

The easiest way to install the systemd service is using the provided installation script.

#### System-wide Installation

```bash
# Build the router first
cargo build --release

# Install the service
sudo ./scripts/install_service.sh install --binary-path $(pwd)/target/release/vllm-router

# Enable the service to start on boot
sudo ./scripts/install_service.sh enable

# Start the service
sudo ./scripts/install_service.sh start
```

#### User-level Installation

Install the service for the current user without requiring root access:

```bash
# Install for current user
./scripts/install_service.sh install --user --binary-path $(pwd)/target/release/vllm-router

# Enable service to start on user login
systemctl --user enable vllm-router

# Start the service
systemctl --user start vllm-router
```

### Method 2: Using Makefile

The Makefile provides convenient shortcuts for service management:

```bash
# Build and install service
make install-service

# Enable and start
make service-enable
make service-start

# Check status
make service-status
```

### Method 3: Manual Installation

If you prefer manual installation:

```bash
# 1. Create user (optional but recommended)
sudo useradd -r -s /bin/false -d /var/lib/vllm-router vllm-router

# 2. Create directories
sudo mkdir -p /var/lib/vllm-router
sudo mkdir -p /etc/vllm-router
sudo chown vllm-router:vllm-router /var/lib/vllm-router

# 3. Build and install binary
cargo build --release
sudo cp target/release/vllm-router /usr/local/bin/

# 4. Copy systemd service file
sudo cp systemd/vllm-router.service /etc/systemd/system/

# 5. Update service file with correct paths
sudo sed -i 's|/usr/local/bin/vllm-router|/usr/local/bin/vllm-router|g' /etc/systemd/system/vllm-router.service

# 6. Reload systemd
sudo systemctl daemon-reload

# 7. Enable and start
sudo systemctl enable vllm-router
sudo systemctl start vllm-router
```

## Configuration

### Creating the Configuration File

The systemd service expects a configuration file at `/etc/vllm-router/config.toml` (or the path specified during installation).

Example `/etc/vllm-router/config.toml`:

```toml
# Server configuration
host = "0.0.0.0"
port = 8080

# Worker URLs
worker_urls = [
    "http://worker1:8000",
    "http://worker2:8000",
]

# Load balancing policy
policy = "consistent_hash"

# Data parallelism
intra_node_data_parallel_size = 8

# Logging
log_level = "info"

# Metrics
metrics_enabled = true
metrics_port = 9090
```

### Custom Installation Paths

Use the `--config-path` option to specify a custom configuration location:

```bash
./scripts/install_service.sh install \
    --binary-path /usr/local/bin/vllm-router \
    --config-path /opt/vllm-router/config.toml
```

## Service Management

### Common Commands

```bash
# Check service status
systemctl status vllm-router

# Start the service
sudo systemctl start vllm-router

# Stop the service
sudo systemctl stop vllm-router

# Restart the service
sudo systemctl restart vllm-router

# Enable service to start on boot
sudo systemctl enable vllm-router

# Disable service from auto-start
sudo systemctl disable vllm-router

# View service logs
sudo journalctl -u vllm-router -f

# View last 100 lines of logs
sudo journalctl -u vllm-router -n 100
```

### Using the Installation Script

```bash
# Check status
./scripts/install_service.sh status

# Start/stop/restart
./scripts/install_service.sh start
./scripts/install_service.sh stop
./scripts/install_service.sh restart

# Enable/disable
./scripts/install_service.sh enable
./scripts/install_service.sh disable
```

## Uninstallation

### Using the Installation Script

```bash
# Stop and disable service first
sudo ./scripts/install_service.sh stop
sudo ./scripts/install_service.sh disable

# Uninstall the service
sudo ./scripts/install_service.sh uninstall
```

### Using Makefile

```bash
make service-stop
make service-disable
make uninstall-service
```

### Manual Uninstallation

```bash
# Stop and disable
sudo systemctl stop vllm-router
sudo systemctl disable vllm-router

# Remove service file
sudo rm /etc/systemd/system/vllm-router.service

# Reload systemd
sudo systemctl daemon-reload

# Optional: Remove user and directories
sudo userdel vllm-router
sudo rm -rf /var/lib/vllm-router
sudo rm -rf /etc/vllm-router

# Optional: Remove binary
sudo rm /usr/local/bin/vllm-router
```

## Troubleshooting

### Service Fails to Start

1. Check service logs:
   ```bash
   sudo journalctl -u vllm-router -n 50 --no-pager
   ```

2. Verify binary exists and is executable:
   ```bash
   ls -l /usr/local/bin/vllm-router
   /usr/local/bin/vllm-router --version
   ```

3. Check configuration file:
   ```bash
   cat /etc/vllm-router/config.toml
   ```

4. Check file permissions:
   ```bash
   ls -la /var/lib/vllm-router
   ls -la /etc/vllm-router
   ```

### Permission Issues

If you encounter permission errors, ensure:

1. The vllm-router user exists:
   ```bash
   id vllm-router
   ```

2. Directories have correct ownership:
   ```bash
   sudo chown -R vllm-router:vllm-router /var/lib/vllm-router
   sudo chown -R vllm-router:vllm-router /etc/vllm-router
   ```

### Logs Not Appearing

If logs don't appear in journald, check:

1. Systemd logging configuration in service file
2. Run binary manually to see direct output:
   ```bash
   sudo -u vllm-router /usr/local/bin/vllm-router --config /etc/vllm-router/config.toml
   ```

## Advanced Configuration

### Running as Specific User

To run as a different user, modify the `User=` and `Group=` directives in `/etc/systemd/system/vllm-router.service`:

```ini
[Service]
User=myuser
Group=mygroup
```

### Environment Variables

Add environment variables to the service file:

```ini
[Service]
Environment="RUST_LOG=info"
Environment="MY_VAR=value"
```

Or use an environment file:

```ini
[Service]
EnvironmentFile=/etc/vllm-router/service.env
```

### Resource Limits

Adjust resource limits in `/etc/systemd/system/vllm-router.service`:

```ini
[Service]
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=4G
```

## Testing

### Verify Service Installation

```bash
# Check if service file exists
ls -l /etc/systemd/system/vllm-router.service

# Verify service is recognized by systemd
systemctl cat vllm-router

# Test service syntax
systemd-analyze verify vllm-router.service
```

### Run Tests

```bash
# Run unit tests
pytest py_test/unit/test_systemd_installation.py

# Run with coverage
pytest py_test/unit/test_systemd_installation.py --cov=py_test/unit --cov-report=html
```

## Production Considerations

### Security

1. **User Isolation**: Run as dedicated user (not root)
2. **File Permissions**: Restrictive permissions on config files
3. **Network**: Firewall rules for exposed ports
4. **Resource Limits**: Set appropriate limits in systemd

### High Availability

1. **Restart Policy**: Service uses `Restart=always` for auto-restart
2. **Monitoring**: Use monitoring tools (Prometheus, Grafana)
3. **Alerting**: Set up alerts on service failures
4. **Logging**: Centralized log aggregation

### Performance

1. **Resource Limits**: Tune based on your workload
2. **Connection Limits**: Configure appropriate values
3. **Metrics**: Enable metrics for performance monitoring
4. **Logging**: Adjust log level appropriately (warn/error in production)

## Additional Resources

- [Systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [vLLM Router README](../../README.md)
- [Issue Tracker](https://github.com/blablador/vllm-router-auth/issues)