#!/usr/bin/env bash
set -e

print_usage() {
    cat << EOF
Usage: $0 [install|uninstall|enable|disable|start|stop|restart|status] [options]

Commands:
  install    Install systemd service
  uninstall  Remove systemd service
  enable     Enable service to start on boot
  disable    Disable service from auto-start
  start      Start the service
  stop       Stop the service
  restart    Restart the service
  status     Show service status

Install options:
  --user         Install for current user (default: system-wide)
  --binary-path  Path to vllm-router binary (default: auto-detect)
  --config-path  Path to config file (default: /etc/vllm-router/config.toml)
  --no-user      Skip creating dedicated user (requires root user)

Example:
  $0 install --binary-path /usr/local/bin/vllm-router
  $0 start
  $0 status
EOF
}

check_root() {
    if [[ "$EUID" -ne 0 ]]; then
        echo "Error: This command requires root privileges."
        echo "Please run with sudo or as root."
        exit 1
    fi
}

detect_binary() {
    local binary
    
    if [[ -n "$BINARY_PATH" ]]; then
        binary="$BINARY_PATH"
    else
        if command -v vllm-router &> /dev/null; then
            binary=$(command -v vllm-router)
        elif [[ -f "target/release/vllm-router" ]]; then
            binary="${PWD}/target/release/vllm-router"
        else
            echo "Error: Could not find vllm-router binary."
            echo "Please specify --binary-path or build the project first."
            exit 1
        fi
    fi
    
    echo "$binary"
}

install_service() {
    local user_mode="$1"
    local binary_path
    binary_path=$(detect_binary)
    
    echo "Installing vLLM Router systemd service..."
    echo "Binary: $binary_path"
    echo "Config: $CONFIG_PATH"
    
    if [[ "$user_mode" == "user" ]]; then
        echo "Installing for current user..."
        
        mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
        mkdir -p "${XDG_DATA_HOME:-$HOME/.local/share}/vllm-router"
        
        sed "s|/usr/local/bin/vllm-router|$binary_path|g" \
            systemd/vllm-router.service > "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/vllm-router.service"
        
        sed -i "s|/etc/vllm-router/config.toml|$CONFIG_PATH|g" \
            "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/vllm-router.service"
        
        systemctl --user daemon-reload
        echo "Service installed successfully. Run: systemctl --user enable vllm-router"
    else
        check_root
        
        if ! id "vllm-router" &>/dev/null && [[ "$CREATE_USER" == "true" ]]; then
            echo "Creating vllm-router user..."
            useradd -r -s /bin/false -d /var/lib/vllm-router vllm-router
            mkdir -p /var/lib/vllm-router
            chown vllm-router:vllm-router /var/lib/vllm-router
        fi
        
        mkdir -p /etc/vllm-router
        
        sed "s|/usr/local/bin/vllm-router|$binary_path|g" \
            systemd/vllm-router.service > /tmp/vllm-router.service
        
        sed -i "s|/etc/vllm-router/config.toml|$CONFIG_PATH|g" \
            /tmp/vllm-router.service
        
        mv /tmp/vllm-router.service /etc/systemd/system/vllm-router.service
        
        systemctl daemon-reload
        echo "Service installed successfully. Run: sudo systemctl enable vllm-router"
    fi
}

uninstall_service() {
    local user_mode="$1"
    
    echo "Uninstalling vLLM Router systemd service..."
    
    if [[ "$user_mode" == "user" ]]; then
        systemctl --user stop vllm-router 2>/dev/null || true
        systemctl --user disable vllm-router 2>/dev/null || true
        rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/vllm-router.service"
        systemctl --user daemon-reload
        echo "Service uninstalled."
    else
        check_root
        systemctl stop vllm-router 2>/dev/null || true
        systemctl disable vllm-router 2>/dev/null || true
        rm -f /etc/systemd/system/vllm-router.service
        systemctl daemon-reload
        echo "Service uninstalled."
    fi
}

service_command() {
    local cmd="$1"
    
    if [[ "$USER_MODE" == "user" ]]; then
        systemctl --user "$cmd" vllm-router
    else
        if [[ "$cmd" != "status" ]]; then
            check_root
        fi
        systemctl "$cmd" vllm-router
    fi
}

USER_MODE="system"
BINARY_PATH=""
CONFIG_PATH="/etc/vllm-router/config.toml"
CREATE_USER="true"

if [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            USER_MODE="user"
            shift
            ;;
        --no-user)
            CREATE_USER="false"
            shift
            ;;
        --binary-path)
            BINARY_PATH="$2"
            shift 2
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

case "$COMMAND" in
    install)
        install_service "$USER_MODE"
        ;;
    uninstall)
        uninstall_service "$USER_MODE"
        ;;
    enable|disable|start|stop|restart|status)
        service_command "$COMMAND"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac